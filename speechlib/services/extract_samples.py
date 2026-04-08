"""
Application service que ejecuta SpeakerSamplePlan contra un audio real.

Mutable shell del dominio puro plan_speaker_samples: orquesta el I/O
(slicing del audio + escritura de WAVs) sin razonar sobre que extraer.

Estructura de salida (Slice 16):
    <output_dir>/<nombre>/clip_NN.wav              ← identificados
    <output_dir>/por_nombrar/<SPEAKER_XX>/clip_NN.wav  ← no identificados

Workflow del usuario para enrolar:
    1. Revisar <output_dir>/por_nombrar/
    2. Escuchar cada SPEAKER_XX, decidir el nombre real
    3. Renombrar la carpeta y moverla a voices/<nombre>/
    4. Re-correr el pipeline → ese speaker queda identificado
"""

from pathlib import Path

from ..audio_utils import slice_and_save
from ..domain.sample_extraction import SpeakerSamplePlan

UNIDENTIFIED_SUBFOLDER = "por_nombrar"


def _destination_dir_for_plan(plan: SpeakerSamplePlan, output_dir: Path) -> Path:
    """Devuelve el directorio destino para los clips de un plan.

    Funcion pura: solo depende del plan y del output_dir base.
    Identificados van al raiz; no identificados a por_nombrar/.
    """
    if plan.is_identified:
        return output_dir / plan.speaker_label
    return output_dir / UNIDENTIFIED_SUBFOLDER / plan.speaker_label


def extract_speaker_samples(
    plans: tuple[SpeakerSamplePlan, ...],
    audio_path: Path,
    output_dir: Path,
) -> dict[str, list[Path]]:
    """Ejecuta los planes contra el audio fuente.

    Args:
        plans: tuple de SpeakerSamplePlan generados por plan_speaker_samples.
        audio_path: WAV fuente (usualmente el preprocessed/enhanced).
        output_dir: raiz donde se crearan los subdirectorios por speaker.

    Returns:
        {speaker_label: [paths_de_wavs_creados]} en el mismo orden que el plan.
    """
    output_dir = Path(output_dir)
    audio_path = Path(audio_path)
    written: dict[str, list[Path]] = {}

    for plan in plans:
        speaker_dir = _destination_dir_for_plan(plan, output_dir)
        speaker_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, clip in enumerate(plan.clips, start=1):
            dest = speaker_dir / f"clip_{i:02d}.wav"
            slice_and_save(
                str(audio_path),
                clip.start_ms,
                clip.end_ms,
                str(dest),
            )
            paths.append(dest)
        written[plan.speaker_label] = paths

    return written
