"""
Domain logic para extraer audio samples por speaker.

plan_speaker_samples es una funcion pura que decide QUE recortar de un audio:
agrupa los segmentos del Transcript por SpeakerIdentity.label (de modo que
los identificados colapsen aunque vengan de SPEAKER_XX distintos, y los
no identificados queden separados por su tag pyannote), filtra los que no
llegan a la duracion minima y selecciona los top-N mas largos por speaker.

Reemplaza la logica de extract_unknown_speakers porque cubre TANTO speakers
identificados como no identificados en una sola operacion.

Estilo GOOS-sin-mocks: cero I/O. La ejecucion (cortar el audio y escribir
WAVs) vive en speechlib/services/extract_samples.py.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

from .transcript import Transcript, TranscriptSegment


@dataclass(frozen=True)
class SampleClip:
    """Una ventana temporal a recortar del audio fuente."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class SpeakerSamplePlan:
    """Plan de extraccion para un speaker: que clips cortar."""

    speaker_label: str
    is_identified: bool
    clips: tuple[SampleClip, ...]


def _build_plan_for_speaker(
    label: str,
    is_identified: bool,
    segments: list[TranscriptSegment],
    max_clips: int,
    min_duration_ms: int,
) -> Optional[SpeakerSamplePlan]:
    """Construye el plan para un speaker individual, o None si todos sus
    clips quedan filtrados por duracion minima.

    Pura: cero I/O. Top-N por duracion descendente, reordenados por start_ms
    para output consistente.
    """
    eligible = [s for s in segments if (s.end_ms - s.start_ms) >= min_duration_ms]
    if not eligible:
        return None
    top = sorted(eligible, key=lambda s: -(s.end_ms - s.start_ms))[:max_clips]
    top.sort(key=lambda s: s.start_ms)
    clips = tuple(SampleClip(start_ms=s.start_ms, end_ms=s.end_ms) for s in top)
    return SpeakerSamplePlan(
        speaker_label=label,
        is_identified=is_identified,
        clips=clips,
    )


def plan_speaker_samples(
    transcript: Transcript,
    max_clips_per_speaker: int,
    min_clip_duration_ms: int,
    min_unidentified_clip_duration_ms: Optional[int] = None,
) -> tuple[SpeakerSamplePlan, ...]:
    """Construye los planes de extraccion para todos los speakers del transcript.

    Args:
        transcript: aggregate del que se extraen los segmentos.
        max_clips_per_speaker: cap del top-N por speaker. Si <=0, retorna ().
        min_clip_duration_ms: duracion minima por clip para identificados.
            Garantiza calidad de los samples de verificacion.
        min_unidentified_clip_duration_ms: duracion minima por clip para
            speakers NO identificados (los que van a por_nombrar/). Default
            None = usa el mismo threshold que identificados (backward compat).
            Pasar un valor menor (ej. 500) le da visibilidad a speakers con
            audio escaso para que el usuario pueda al menos saber que existen.

    Returns:
        Tuple de planes ordenado: identificados primero (alfabetico) y luego
        no identificados (alfabetico). Speakers cuyos clips quedan todos
        filtrados no aparecen en el resultado.
    """
    if max_clips_per_speaker <= 0:
        return ()

    if min_unidentified_clip_duration_ms is None:
        min_unidentified_clip_duration_ms = min_clip_duration_ms

    # Agrupar segmentos por label (no por tag): identificados con el mismo
    # nombre colapsan, no identificados quedan separados por SPEAKER_XX.
    grouped: dict[str, list[TranscriptSegment]] = {}
    identified_label: dict[str, bool] = {}
    for seg in transcript.segments:
        label = seg.speaker.label
        grouped.setdefault(label, []).append(seg)
        # is_identified es invariante por label: si algun segmento del label
        # tiene recognized_name, todos lo tienen (label = recognized_name).
        identified_label[label] = seg.speaker.is_identified

    plans: list[SpeakerSamplePlan] = []
    for label, segments in grouped.items():
        is_identified = identified_label[label]
        threshold = (
            min_clip_duration_ms if is_identified
            else min_unidentified_clip_duration_ms
        )
        plan = _build_plan_for_speaker(
            label=label,
            is_identified=is_identified,
            segments=segments,
            max_clips=max_clips_per_speaker,
            min_duration_ms=threshold,
        )
        if plan is not None:
            plans.append(plan)

    # Identificados primero (alfabetico), luego no identificados (alfabetico).
    plans.sort(key=lambda p: (not p.is_identified, p.speaker_label))
    return tuple(plans)
