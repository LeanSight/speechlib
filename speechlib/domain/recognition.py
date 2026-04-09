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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


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
        scores.append((name, _cosine_similarity(embedding, voice_emb)))
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
