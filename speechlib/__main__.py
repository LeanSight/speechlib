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
    invoke_without_command=True,
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


# Shared options reusable across subcommands
_file_arg = Annotated[Path, typer.Argument(
    exists=True, file_okay=True, dir_okay=False, readable=True,
    help="Audio file to transcribe",
)]
_voices_opt = Annotated[Optional[Path], typer.Option(
    exists=True, file_okay=False, dir_okay=True, readable=True,
    help="Folder with speaker voice samples",
)]
_speakers_opt = Annotated[Optional[str], typer.Option(
    help="Comma-separated expected attendees (filters voice library)",
)]
_verbose_opt = Annotated[bool, typer.Option("-v", "--verbose", help="Show detailed progress")]


def _parse_speakers(speakers: Optional[str]) -> list[str] | None:
    return [s.strip() for s in speakers.split(",")] if speakers else None


def _parse_hotwords(value: Optional[str]) -> list[str] | None:
    """Parse --hotwords value. Supports '@<path>' (read file) or CSV inline.

    File format: one term per line; empty lines and lines starting with '#'
    are ignored. Returns None if value is None/empty.
    """
    if not value:
        return None
    if value.startswith("@"):
        path = Path(value[1:])
        terms = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return terms or None
    return [s.strip() for s in value.split(",")]


def _transcribe_one(
    file: Path,
    voices_folder: Optional[Path],
    log_folder: Optional[Path],
    language: str,
    model: str,
    resolved_token: str,
    output_format: "OutputFormat",
    skip_enhance: bool,
    compress: bool,
    quantization: bool,
    grouping: "Grouping",
    speakers: Optional[str],
    initial_prompt: Optional[str],
    hotwords: Optional[str],
) -> None:
    """Single full-pipeline pass over one file (shared by `run` and `batch`)."""
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
        allowed_speakers=_parse_speakers(speakers),
        initial_prompt=initial_prompt,
        hotwords=_parse_hotwords(hotwords),
    )


@app.callback()
def callback():
    """Speechlib: transcribe audio with speaker diarization and recognition."""


@app.command()
def run(
    file: _file_arg,
    voices_folder: _voices_opt = None,
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
    skip_enhance: Annotated[bool, typer.Option(help="Skip enhancement on output")] = False,
    compress: Annotated[bool, typer.Option(help="Generate compressed _limpio.m4a")] = False,
    quantization: Annotated[bool, typer.Option(help="Use int8 quantization")] = False,
    grouping: Annotated[Grouping, typer.Option(help="Grouping mode")] = Grouping.sentences,
    speakers: _speakers_opt = None,
    initial_prompt: Annotated[Optional[str], typer.Option(
        help="Context text biasing Whisper decoding (domain terms, names, jargon)",
    )] = None,
    hotwords: Annotated[Optional[str], typer.Option(
        help="Comma-separated terms (or '@<path>' to read one-per-line from file) "
             "injected as logit bias (alternative to --initial-prompt)",
    )] = None,
    verbose: _verbose_opt = False,
):
    """Full pipeline: preprocess, diarize, recognize, transcribe, publish."""
    _setup_logging(verbose)
    resolved_token = _resolve_token(token)

    _transcribe_one(
        file, voices_folder, log_folder, language, model, resolved_token,
        output_format, skip_enhance, compress, quantization, grouping,
        speakers, initial_prompt, hotwords,
    )


@app.command()
def batch(
    files: Annotated[list[Path], typer.Argument(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Audio files to transcribe (one process warms whisper+pyannote once)",
    )],
    voices_folder: _voices_opt = None,
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
    skip_enhance: Annotated[bool, typer.Option(help="Skip enhancement on output")] = False,
    compress: Annotated[bool, typer.Option(help="Generate compressed _limpio.m4a")] = False,
    quantization: Annotated[bool, typer.Option(help="Use int8 quantization")] = False,
    grouping: Annotated[Grouping, typer.Option(help="Grouping mode")] = Grouping.sentences,
    speakers: _speakers_opt = None,
    initial_prompt: Annotated[Optional[str], typer.Option(
        help="Context text biasing Whisper decoding (domain terms, names, jargon)",
    )] = None,
    hotwords: Annotated[Optional[str], typer.Option(
        help="Comma-separated terms (or '@<path>' to read one-per-line from file) "
             "injected as logit bias (alternative to --initial-prompt)",
    )] = None,
    verbose: _verbose_opt = False,
):
    """Transcribe many files in ONE process: whisper+pyannote load once (via the
    in-process model cache) and every file gets the same per-file contract as
    `run` (a <stem>_limpio.vtt beside each input). No daemon, no IPC."""
    _setup_logging(verbose)
    resolved_token = _resolve_token(token)

    for f in files:
        _transcribe_one(
            f, voices_folder, log_folder, language, model, resolved_token,
            output_format, skip_enhance, compress, quantization, grouping,
            speakers, initial_prompt, hotwords,
        )


@app.command()
def recognize(
    file: _file_arg,
    voices_folder: _voices_opt = None,
    speakers: _speakers_opt = None,
    force: Annotated[bool, typer.Option(help="Force recompute (delete cached speaker_map)")] = False,
    verbose: _verbose_opt = False,
):
    """Re-run speaker recognition on existing diarization artifacts."""
    _setup_logging(verbose)
    from .core_analysis import run_recognition

    if voices_folder is None:
        raise typer.BadParameter("--voices-folder required for recognize.", param_hint="--voices-folder")

    result = run_recognition(
        file_name=str(file),
        voices_folder=str(voices_folder),
        allowed_speakers=_parse_speakers(speakers),
        force=force,
    )
    import json
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def confirm(
    file: _file_arg,
    verbose: _verbose_opt = False,
):
    """Apply user-edited speaker_map.json to regenerate VTT with real names.

    Reads <cache>/speaker_map.json (written by the user based on
    speaker_map_suggestions.json) and rewrites the published VTT applying
    the mapping. Unmapped clusters stay as [SPEAKER_XX].
    """
    _setup_logging(verbose)
    from .core_analysis import run_confirm

    result = run_confirm(str(file))
    import json
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def diagnose(
    file: _file_arg,
    voices_folder: _voices_opt = None,
    speakers: _speakers_opt = None,
    verbose: _verbose_opt = False,
):
    """Show speaker recognition score matrix (read-only, no artifacts modified)."""
    _setup_logging(verbose)
    from .core_analysis import run_diagnose

    if voices_folder is None:
        raise typer.BadParameter("--voices-folder required for diagnose.", param_hint="--voices-folder")

    result = run_diagnose(
        file_name=str(file),
        voices_folder=str(voices_folder),
        allowed_speakers=_parse_speakers(speakers),
    )
    import json
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
