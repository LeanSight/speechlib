"""
Re-etiqueta los speakers de un VTT existente sin re-transcribir ni re-diarizar.

Reescritura Slice 10: el corazon de la logica vive ahora en el dominio puro
(Transcript + assign_speakers) y este archivo es un thin CLI que orquesta
el I/O necesario:

  parse_vtt
    -> Transcript
    -> compute embeddings per label desde el audio
    -> assign_speakers (funcion pura del dominio)
    -> aplicar identidades a los VttBlocks
    -> write_vtt

Las dos ramas paralelas anteriores (--rttm y --all-speakers con codigo
duplicado) fueron eliminadas. assign_speakers preserva por construccion
el SPEAKER_XX cuando ningun voice supera threshold — el bug del literal
[unknown] es estructuralmente imposible.

Uso:
    python -m speechlib.tools.relabel_vtt VTT_PATH AUDIO_PATH VOICES_FOLDER \\
        [--threshold 0.45] [--min-margin 0.10] [--all-speakers]

Output: escribe VTT corregido junto al original con sufijo _relabeled.vtt
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, r"c:\workspace\#dev\ClearerVoice-Studio\clearvoice")

import numpy as np

from speechlib.audio_utils import slice_and_save
from speechlib.domain.recognition import assign_speakers
from speechlib.domain.transcript import (
    SpeakerIdentity,
    Transcript,
    TranscriptSegment,
)
from speechlib.speaker_recognition import (
    MIN_SEGMENT_DURATION_S,
    SPEAKER_SIMILARITY_MIN_MARGIN,
    SPEAKER_SIMILARITY_THRESHOLD,
    _get_inference,
    load_avg_voice_embeddings,
)
from speechlib.vtt_utils import VttBlock, parse_vtt, write_vtt


# ── Pure helpers ──────────────────────────────────────────────────────────────


def _label_is_unidentified(label: str) -> bool:
    """Inline reemplazo de is_unidentified_speaker (que se borra en Slice 11)."""
    return label == "unknown" or (
        label.startswith("SPEAKER_") and label[8:].isdigit()
    )


def build_transcript_from_vtt_blocks(
    blocks: list[VttBlock],
    audio_path: str,
    language: str,
) -> Transcript:
    """Convierte VttBlocks parseados a Transcript del dominio.

    Para cada bloque:
    - Si el label parece SPEAKER_XX o "unknown", crea SpeakerIdentity
      no identificada (recognized_name=None, diarization_tag=label).
    - Si el label es un nombre real, crea SpeakerIdentity ya identificada.

    En ambos casos, el label original viaja como diarization_tag para
    soportar el agrupamiento de assign_speakers (un embedding por label).

    Pura: sin I/O.
    """
    segments = []
    for b in blocks:
        if _label_is_unidentified(b.speaker):
            identity = SpeakerIdentity(
                diarization_tag=b.speaker,
                recognized_name=None,
            )
        else:
            identity = SpeakerIdentity(
                diarization_tag=b.speaker,
                recognized_name=b.speaker,
                similarity=None,
            )
        segments.append(
            TranscriptSegment(
                start_ms=b.start_ms,
                end_ms=b.end_ms,
                text=b.text,
                speaker=identity,
            )
        )
    return Transcript(
        segments=tuple(segments),
        audio_path=audio_path,
        language=language,
    )


def apply_transcript_labels_to_blocks(
    blocks: list[VttBlock],
    transcript: Transcript,
) -> int:
    """Aplica los labels del Transcript de vuelta a los VttBlocks por indice.

    Pura. Retorna la cantidad de bloques cuyo label cambio.
    """
    changed = 0
    for block, segment in zip(blocks, transcript.segments):
        new_label = segment.speaker.label
        if new_label != block.speaker:
            block.speaker = new_label
            changed += 1
    return changed


# ── Application service: compute embeddings from audio ───────────────────────


def compute_embeddings_per_label(
    blocks: list[VttBlock],
    audio_path: str,
    target_labels: set[str] | None = None,
    limit_s: float = 60.0,
) -> dict[str, np.ndarray]:
    """Por cada label en target_labels (o todos si None), computa el embedding
    promedio a partir de chunks de audio. Mirroring de la logica canonica de
    speaker_recognition: per-chunk embedding + mean, filtrando NaN.

    Mutable shell: I/O del audio + slicing + inference. Las decisiones viven
    en el dominio.
    """
    inference = _get_inference()
    grouped: dict[str, list[VttBlock]] = {}
    for b in blocks:
        if target_labels is not None and b.speaker not in target_labels:
            continue
        grouped.setdefault(b.speaker, []).append(b)

    result: dict[str, np.ndarray] = {}
    tmp_dir = Path(tempfile.mkdtemp(prefix="relabel_vtt_"))
    try:
        for label, label_blocks in grouped.items():
            accumulated_ms = 0
            embs: list[np.ndarray] = []
            for i, b in enumerate(label_blocks):
                if accumulated_ms >= limit_s * 1000:
                    break
                # Slice 12: filtrar turnos < MIN_SEGMENT_DURATION_S antes de
                # cortar (pyannote/embedding rompe en chunks ultra-cortos).
                if (b.end_ms - b.start_ms) < MIN_SEGMENT_DURATION_S * 1000:
                    continue
                chunk = tmp_dir / f"chunk_{i}.wav"
                try:
                    slice_and_save(audio_path, b.start_ms, b.end_ms, str(chunk))
                    arr = np.asarray(inference(str(chunk))).flatten()
                    if not np.isnan(arr).any():
                        embs.append(arr)
                except Exception:
                    pass
                finally:
                    chunk.unlink(missing_ok=True)
                accumulated_ms += b.end_ms - b.start_ms
            if embs:
                result[label] = np.mean(embs, axis=0)
    finally:
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    return result


# ── Main CLI ─────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtt_path")
    parser.add_argument("audio_path")
    parser.add_argument("voices_folder")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SPEAKER_SIMILARITY_THRESHOLD,
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=SPEAKER_SIMILARITY_MIN_MARGIN,
        help=(
            "Margen minimo top1 vs top2 para aceptar un match. "
            f"Default: {SPEAKER_SIMILARITY_MIN_MARGIN}. Pasar 0 para desactivar."
        ),
    )
    parser.add_argument(
        "--all-speakers",
        action="store_true",
        help=(
            "Re-evaluar TODOS los bloques del VTT, no solo los no identificados. "
            "Util para detectar misidentificaciones previas."
        ),
    )
    args = parser.parse_args()

    vtt_path = Path(args.vtt_path)
    audio_path = args.audio_path
    voices_folder = Path(args.voices_folder)

    print(f"\nVTT       : {vtt_path.name}")
    print(f"Audio     : {Path(audio_path).name}")
    print(f"Voices    : {voices_folder}")
    print(f"Threshold : {args.threshold}")
    print(f"Min margin: {args.min_margin}")
    if args.all_speakers:
        print("Modo      : --all-speakers (re-evalua todos los bloques)")
    else:
        print("Modo      : solo bloques no identificados (SPEAKER_XX o unknown)")

    print("\nParsando VTT...")
    header, blocks = parse_vtt(vtt_path)
    transcript = build_transcript_from_vtt_blocks(
        blocks, str(audio_path), language="es"
    )
    unknown_count = sum(1 for s in transcript.segments if not s.speaker.is_identified)
    print(f"  {len(blocks)} bloques totales, {unknown_count} no identificados")

    if args.all_speakers:
        target_labels = set(transcript.diarization_tags)
    else:
        target_labels = {
            s.speaker.diarization_tag
            for s in transcript.segments
            if not s.speaker.is_identified
        }

    if not target_labels:
        print("\nNada que re-evaluar (sin bloques no identificados).")
        suffix = "_relabeled"
        out_path = vtt_path.with_stem(vtt_path.stem + suffix)
        write_vtt(out_path, header, blocks)
        return

    print(f"\nCargando libreria de voces...")
    voice_library = load_avg_voice_embeddings(voices_folder)
    print(f"  {len(voice_library)} speakers: {sorted(voice_library)}")

    print(f"\nComputando embeddings de {len(target_labels)} labels...")
    embeddings_by_tag = compute_embeddings_per_label(
        blocks=blocks,
        audio_path=str(audio_path),
        target_labels=target_labels,
    )
    print(f"  Embeddings calculados: {len(embeddings_by_tag)} / {len(target_labels)}")

    print(f"\nAplicando assign_speakers...")
    relabeled = assign_speakers(
        transcript=transcript,
        embeddings_by_tag=embeddings_by_tag,
        voice_library=voice_library,
        threshold=args.threshold,
        min_margin=args.min_margin,
    )
    changed = apply_transcript_labels_to_blocks(blocks, relabeled)

    out_path = vtt_path.with_stem(vtt_path.stem + "_relabeled")
    write_vtt(out_path, header, blocks)

    print(f"\n{'=' * 50}")
    print(f"  Bloques re-etiquetados : {changed}")
    print(
        f"  Siguen sin identificar : "
        f"{sum(1 for s in relabeled.segments if not s.speaker.is_identified)}"
    )
    print(f"  Output                 : {out_path.name}")
    print(f"{'=' * 50}")

    from collections import Counter

    dist = Counter(b.speaker for b in blocks)
    print("\nDistribucion de speakers:")
    for speaker, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {speaker:<28} {count:>4} bloques")


if __name__ == "__main__":
    main()
