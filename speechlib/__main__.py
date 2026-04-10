"""CLI entry point: python -m speechlib <audio> [options]."""
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from .core_analysis import core_analysis

app = typer.Typer(
    name="speechlib",
    help="Transcribe audio with speaker diarization and recognition.",
    rich_markup_mode="rich",
)


class OutputFormat(str, Enum):
    vtt = "vtt"
    txt = "txt"


class Grouping(str, Enum):
    sentences = "sentences"
    timestamps = "timestamps"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if verbose:
        for name in ("speechlib", "faster_whisper", "pyannote"):
            logging.getLogger(name).setLevel(logging.DEBUG)
        for name in ("urllib3", "filelock", "huggingface_hub", "fsspec", "numba",
                     "httpcore", "httpx"):
            logging.getLogger(name).setLevel(logging.WARNING)
        os.environ.setdefault("SPEECHLIB_PROFILE", "1")


def _resolve_token(token: Optional[str]) -> str:
    resolved = token or os.environ.get("HF_TOKEN") or ""
    if not resolved:
        raise typer.BadParameter(
            "HuggingFace token required. Use --token or set HF_TOKEN env var.",
            param_hint="--token",
        )
    return resolved


@app.command()
def main(
    file: Annotated[Path, typer.Argument(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Audio file to transcribe",
    )],
    voices_folder: Annotated[Optional[Path], typer.Option(
        exists=True, file_okay=False, dir_okay=True, readable=True,
        help="Folder with speaker voice samples",
    )] = None,
    log_folder: Annotated[Optional[Path], typer.Option(
        help="Output folder (default: <file_dir>/output)",
    )] = None,
    language: Annotated[str, typer.Option(help="Language code")] = "es",
    model: Annotated[str, typer.Option(help="Whisper model size")] = "large-v3-turbo",
    token: Annotated[Optional[str], typer.Option(
        envvar="HF_TOKEN", help="HuggingFace token",
    )] = None,
    output_format: Annotated[OutputFormat, typer.Option(
        help="Output format",
    )] = OutputFormat.vtt,
    skip_enhance: Annotated[bool, typer.Option(help="Skip audio enhancement")] = False,
    compress: Annotated[bool, typer.Option(help="Generate compressed _limpio.m4a")] = False,
    quantization: Annotated[bool, typer.Option(help="Use int8 quantization")] = False,
    grouping: Annotated[Grouping, typer.Option(help="Grouping mode")] = Grouping.sentences,
    speakers: Annotated[Optional[str], typer.Option(
        help="Comma-separated expected attendees (filters voice library)",
    )] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Show detailed progress")] = False,
):
    """Transcribe audio with speaker diarization and recognition."""
    _setup_logging(verbose)
    resolved_token = _resolve_token(token)

    allowed_speakers = [s.strip() for s in speakers.split(",")] if speakers else None

    core_analysis(
        file_name=str(file),
        voices_folder=str(voices_folder) if voices_folder else None,
        log_folder=str(log_folder) if log_folder else None,
        language=language,
        modelSize=model,
        ACCESS_TOKEN=resolved_token,
        model_type="faster-whisper",
        quantization=quantization,
        output_format=output_format.value,
        skip_enhance=skip_enhance,
        compress=compress,
        grouping_mode=grouping.value,
        allowed_speakers=allowed_speakers,
    )


if __name__ == "__main__":
    app()
