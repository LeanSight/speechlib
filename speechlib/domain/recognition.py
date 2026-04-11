"""
Funcion pura de reconocimiento de speakers contra una libreria de voces.

assign_speakers es la operacion central del refactor: reemplaza la logica
duplicada que vivia repartida entre core_analysis (loop de speaker_recognition
+ fallback "unknown" → tag) y relabel_vtt (dos ramas: --rttm y --all-speakers).

Estilo GOOS-sin-mocks: la funcion no toca audio, ni filesystem, ni pyannote.
Recibe embeddings ya calculados como dict[tag → ndarray] y retorna un nuevo
Transcript. Toda la I/O se queda en la capa de application services.

Invariante critico: cuando ningun voice supera threshold, el segmento
conserva su SpeakerIdentity con recognized_name=None — el diarization_tag
NUNCA se pierde y label() siempre cae al SPEAKER_XX original. El bug del
relabel --all-speakers es estructuralmente imposible aqui.
"""

from typing import Optional

import numpy as np

from .transcript import SpeakerIdentity, Transcript


def select_segments_for_embedding(
    segments: list,
    *,
    limit_s: float,
    min_segment_s: float,
) -> list:
    """Selecciona segmentos representativos para computar el embedding promedio
    de un speaker tag.

    Bug fix: el flujo legacy iteraba en orden de documento y se detenia al
    sumar `limit_s`. En recordings largos, los primeros turnos pueden ser
    contaminados (crosstalk inicial, ruido) y no representan al speaker.
    Esta funcion ordena por duracion descendente: los turnos largos son
    tipicamente los mas limpios y discriminativos.

    Funcion pura del dominio: cero I/O. Recibe los segmentos como [start, end, ...]
    y retorna la sublista que se debe procesar (en el orden largo->corto, hasta
    alcanzar el limit_s acumulado, descartando los menores a min_segment_s).

    Args:
        segments: lista de [start_s, end_s, ...] (extras ignorados).
        limit_s: duracion maxima acumulada (en segundos) a procesar.
        min_segment_s: descarta segmentos individuales menores a este umbral.

    Returns:
        Sublista de segmentos seleccionados, ordenados de mayor a menor duracion.
    """
    eligible = [s for s in segments if (s[1] - s[0]) >= min_segment_s]
    eligible.sort(key=lambda s: -(s[1] - s[0]))
    selected: list = []
    accumulated = 0.0
    for seg in eligible:
        if accumulated >= limit_s:
            break
        selected.append(seg)
        accumulated += seg[1] - seg[0]
    return selected


def average_embeddings(embeddings: list[np.ndarray]) -> Optional[np.ndarray]:
    """Promedia una lista de embeddings, filtrando los que contengan NaN.

    Funcion pura del dominio: cero I/O. Reemplaza la logica duplicada que
    vivia inline en core_analysis._compute_averaged_embeddings_per_tag y
    en speaker_recognition.speaker_recognition (legacy borrado en Slice 15).

    Args:
        embeddings: lista de ndarray (cualquier shape; se aplanan).

    Returns:
        ndarray promedio de los embeddings VALIDOS (sin NaN), o None si
        no hay ninguno valido.
    """
    valid: list[np.ndarray] = []
    for emb in embeddings:
        arr = np.asarray(emb).flatten()
        if not np.isnan(arr).any():
            valid.append(arr)
    if not valid:
        return None
    return np.mean(valid, axis=0)


def assign_extra_speakers(
    speaker_map: dict[str, str],
    extra_names: list[str],
    segment_counts: dict[str, int],
) -> dict[str, str]:
    """Asigna nombres de speakers sin sample a tags no matcheados.

    Tags no matcheados son aquellos donde speaker_map[tag] == tag (no reconocidos).
    Se asignan en orden descendente de segment_counts (el que más habla recibe
    el primer nombre extra).

    Funcion pura del dominio: cero I/O.
    """
    if not extra_names:
        return speaker_map

    unmatched = [tag for tag, name in speaker_map.items() if tag == name]
    unmatched.sort(key=lambda t: -segment_counts.get(t, 0))

    result = dict(speaker_map)
    for tag, name in zip(unmatched, extra_names):
        result[tag] = name
    return result


def filter_voice_library(
    library: dict[str, np.ndarray],
    allowed_names: set[str] | None,
) -> dict[str, np.ndarray]:
    """Filtra la voice library a solo los speakers esperados.

    Si allowed_names es None, retorna la library completa (sin filtro).
    Funcion pura del dominio: cero I/O.
    """
    if allowed_names is None:
        return library
    return {k: v for k, v in library.items() if k in allowed_names}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity entre dos embeddings. Funcion pura, sin scipy."""
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_score_matrix(
    embeddings_by_tag: dict[str, np.ndarray],
    voice_library: dict[str, np.ndarray],
    threshold: float,
    min_margin: float,
) -> dict:
    """Construye matriz de scores para diagnóstico. Puro, JSON-serializable."""
    tags = {}
    for tag, emb in embeddings_by_tag.items():
        scores = {name: round(float(cosine_similarity(emb, v)), 4)
                  for name, v in voice_library.items()}
        name, _ = _best_match(emb, voice_library, threshold, min_margin)
        tags[tag] = {
            "scores": scores,
            "decision": name if name else tag,
        }
    return {
        "threshold": threshold,
        "min_margin": min_margin,
        "tags": tags,
    }


def build_suggestions(
    embeddings_by_tag: dict[str, np.ndarray],
    voice_library: dict[str, np.ndarray],
    threshold: float,
    min_margin: float,
    *,
    top_n: int = 3,
) -> dict:
    """Construye sugerencias de speaker recognition para revisión humana.

    A diferencia de build_score_matrix (formato diagnóstico completo), éste
    retorna solo los top-N candidatos ordenados descendentemente por score
    más un "recommended" que es el nombre del best_match (o None si el match
    es ambiguo o bajo threshold). Formato JSON-serializable directo, pensado
    para el flujo suggest+confirm del subcomando `run`.

    Función pura del dominio: cero I/O.
    """
    tags: dict = {}
    for tag, emb in embeddings_by_tag.items():
        ranked = sorted(
            (
                (name, round(float(cosine_similarity(emb, v)), 4))
                for name, v in voice_library.items()
            ),
            key=lambda kv: -kv[1],
        )
        top = ranked[:top_n]
        recommended, _ = _best_match(emb, voice_library, threshold, min_margin)
        tags[tag] = {
            "top_candidates": [{"name": n, "score": s} for n, s in top],
            "recommended": recommended,
        }
    return {
        "threshold": threshold,
        "min_margin": min_margin,
        "tags": tags,
    }


def _best_match(
    embedding: np.ndarray,
    voice_library: dict[str, np.ndarray],
    threshold: float,
    min_margin: float = 0.0,
) -> tuple[Optional[str], Optional[float]]:
    """Devuelve (name, similarity) del mejor match, o (None, best_similarity)
    si ningun voice supera el threshold O si el margen top1 vs top2 es
    inferior a min_margin (match ambiguo). None es la senal de 'no
    identificado', NO el string 'unknown'."""
    scores: list[tuple[str, float]] = []
    for name, voice_emb in voice_library.items():
        scores.append((name, cosine_similarity(embedding, voice_emb)))
    if not scores:
        return None, None  # libreria vacia

    scores.sort(key=lambda kv: -kv[1])
    best_name, best_score = scores[0]

    if best_score < threshold:
        return None, best_score

    # Slice 9: rechazar matches ambiguos donde top1 no supera a top2 por
    # min_margin. Evita false positives en library con voces de calidad
    # heterogenea cuando ningun PRESENTE real esta cerca y el matcher elige
    # al "menos diferente" entre los AUSENTES.
    if min_margin > 0.0 and len(scores) >= 2:
        second_score = scores[1][1]
        if (best_score - second_score) < min_margin:
            return None, best_score

    return best_name, best_score


def assign_speakers(
    transcript: Transcript,
    embeddings_by_tag: dict[str, np.ndarray],
    voice_library: dict[str, np.ndarray],
    threshold: float,
    min_margin: float = 0.0,
) -> Transcript:
    """Asigna identidades de speaker a cada segmento del transcript.

    Para cada diarization_tag presente en el transcript:
    - Si tiene embedding en embeddings_by_tag, se compara contra voice_library.
      El resultado (con o sin match) se aplica a TODOS los segmentos del tag.
    - Si NO tiene embedding, los segmentos de ese tag quedan intactos.

    Args:
        threshold: minimo de cosine similarity para identificar.
        min_margin: minimo de diferencia top1 - top2 para no considerar el
            match ambiguo. Default 0.0 = no se aplica filtro de margen
            (backward compatible).

    Retorna un Transcript nuevo. La entrada no se mutua.
    """
    # Resolver identidad UNA vez por tag (todos los segmentos del mismo tag
    # comparten identidad). Las identidades sin embedding se marcan como None
    # para indicar "no tocar".
    new_identity_by_tag: dict[str, Optional[SpeakerIdentity]] = {}
    for tag in transcript.diarization_tags:
        embedding = embeddings_by_tag.get(tag)
        if embedding is None:
            new_identity_by_tag[tag] = None
            continue
        name, similarity = _best_match(
            embedding, voice_library, threshold, min_margin=min_margin
        )
        new_identity_by_tag[tag] = SpeakerIdentity(
            diarization_tag=tag,
            recognized_name=name,
            similarity=similarity,
        )

    new_segments = tuple(
        seg if new_identity_by_tag[seg.speaker.diarization_tag] is None
        else seg.with_speaker(new_identity_by_tag[seg.speaker.diarization_tag])
        for seg in transcript.segments
    )
    return transcript.with_segments(new_segments)
