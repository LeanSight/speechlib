"""
Procesa múltiples carpetas de audio, identifica speakers conocidos,
extrae desconocidos a disco para que el usuario los nombre.

Uso:
    from speechlib.batch_process import batch_process

    report = batch_process(
        folders=[Path("@recordings/20260320 Patricio Renner"),
                 Path("@recordings/20260318 Ina TRE")],
        voices_folder=Path("transcript_samples/voices"),
        language="es",
        access_token=os.environ["HF_TOKEN"],
    )
    report.print_summary()
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..core_analysis import core_analysis
from ..domain.sample_extraction import plan_speaker_samples
from ..domain.transcript import Transcript
from ..services.extract_samples import extract_speaker_samples

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".opus"}


@dataclass
class BatchReport:
    folders: list[Path] = field(default_factory=list)
    processed_files: list[Path] = field(default_factory=list)
    identified_speakers: set[str] = field(default_factory=set)
    unknown_speakers: list[dict] = field(default_factory=list)
    errors: int = 0

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("  BATCH PROCESS REPORT")
        print("=" * 60)
        print(f"  Carpetas procesadas : {len(self.folders)}")
        print(f"  Archivos procesados : {len(self.processed_files)}")
        print(f"  Errores             : {self.errors}")
        print(f"  Speakers conocidos  : {sorted(self.identified_speakers)}")
        print(f"  Speakers desconocidos: {len(self.unknown_speakers)}")
        for u in self.unknown_speakers:
            print(f"    [{u['tag']}] en {u['audio'].name}  ->  {u['folder']}")
        print("=" * 60)
        if self.unknown_speakers:
            print("\n  PRÓXIMOS PASOS:")
            print("  1. Escucha los clips en voices/_unknown/")
            print("  2. Renombra la carpeta con el nombre real de la persona")
            print("     ej: mv _unknown/SPEAKER_01_recording voices/Patricio")
            print("  3. Re-ejecuta el batch — esa persona será identificada")
        print()


def batch_process(
    folders: list[Path],
    voices_folder: Path,
    language: str,
    access_token: str,
    model_size: str = "large-v3-turbo",
    unknown_output_dir: Path | None = None,
    min_unknown_duration_s: float = 2.0,
    max_unknown_clips: int = 4,
    skip_enhance: bool = False,
) -> BatchReport:
    """Procesa múltiples carpetas de audio.

    Args:
        folders: Lista de carpetas a procesar (cada una puede tener varios audios).
        voices_folder: Librería de voces conocidas.
        language: Código ISO del idioma (es, en, ...).
        access_token: HuggingFace token para pyannote.
        model_size: Tamaño del modelo Whisper.
        unknown_output_dir: Dónde guardar clips de speakers desconocidos.
                            Por defecto: voices/_unknown/ relativo a voices_folder.
        min_unknown_duration_s: Duración mínima de clip para guardar.
        max_unknown_clips: Máximo de clips por speaker desconocido.
        skip_enhance: Omitir enhance_audio (más rápido, menor calidad).

    Returns:
        BatchReport con resumen completo.
    """
    if unknown_output_dir is None:
        unknown_output_dir = Path(voices_folder).parent / "_unknown"

    report = BatchReport(folders=list(folders))

    for folder in folders:
        folder = Path(folder)
        audio_files = sorted(
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        )

        if not audio_files:
            logger.warning("Carpeta %s: sin archivos de audio", folder)
            continue

        for audio_path in audio_files:
            logger.info("Procesando: %s", audio_path)
            try:
                segments = core_analysis(
                    str(audio_path),
                    voices_folder=str(voices_folder),
                    log_folder=str(folder),
                    language=language,
                    modelSize=model_size,
                    ACCESS_TOKEN=access_token,
                    model_type="faster-whisper",
                    skip_enhance=skip_enhance,
                )

                report.processed_files.append(audio_path)

                # Slice 8: usar el dominio nuevo. core_analysis ya publico
                # transcript.json en artifacts_dir; lo cargamos para clasificar
                # speakers y extraer samples de los no identificados al
                # unknown_output_dir (con sub-dir por audio_stem para
                # disambiguar entre recordings del mismo speaker).
                audio_artifacts_dir = audio_path.parent / f".{audio_path.stem}"
                transcript_path = audio_artifacts_dir / "transcript.json"

                if transcript_path.exists():
                    transcript = Transcript.load(transcript_path)
                    for seg in transcript.segments:
                        if seg.speaker.is_identified:
                            report.identified_speakers.add(seg.speaker.recognized_name)

                    # Extraer samples solo de los NO identificados
                    plans = plan_speaker_samples(
                        transcript,
                        max_clips_per_speaker=max_unknown_clips,
                        min_clip_duration_ms=int(min_unknown_duration_s * 1000),
                    )
                    unknown_plans = tuple(p for p in plans if not p.is_identified)
                    if unknown_plans:
                        unknown_audio_dir = unknown_output_dir / audio_path.stem
                        written = extract_speaker_samples(
                            plans=unknown_plans,
                            audio_path=audio_path,
                            output_dir=unknown_audio_dir,
                        )
                        for label, paths in written.items():
                            report.unknown_speakers.append(
                                {
                                    "tag": label,
                                    "audio": audio_path,
                                    "folder": unknown_audio_dir / label,
                                }
                            )
                else:
                    # Fallback: clasificacion por inspeccion del label legacy
                    # (no deberia ocurrir si Slice 5 esta cableado)
                    for seg in segments:
                        speaker = seg[3]
                        if not speaker.startswith("SPEAKER_") and speaker != "unknown":
                            report.identified_speakers.add(speaker)

            except Exception:
                logger.exception("Error procesando %s", audio_path)
                report.errors += 1

    return report
