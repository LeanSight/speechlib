"""Transcribe audio/video files via AssemblyAI (best model) with speaker labels.

Emits per-input: <stem>.srt, <stem>.json (raw response), <stem>.txt (speaker-labeled).

Usage:
    python transcribe_aai.py [--keyterms <path>] <file1> [<file2> ...]

Keyterms file: one term per line; blank lines and # comments ignored.

Env:
    ASSEMBLY_KEY  API key
"""

import argparse
import json
import os
import sys
from pathlib import Path

import assemblyai as aai


def _load_keyterms(path: Path) -> list[str]:
    terms = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def _fmt_ts(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms_ = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_:03d}"


def _to_srt(utterances) -> str:
    lines = []
    for i, u in enumerate(utterances, 1):
        spk = getattr(u, "speaker", None) or ""
        label = f"[Speaker {spk}] " if spk else ""
        lines.append(
            f"{i}\n{_fmt_ts(u.start)} --> {_fmt_ts(u.end)}\n{label}{u.text}\n"
        )
    return "\n".join(lines)


def _to_plain_text(utterances) -> str:
    return "\n".join(
        f"[Speaker {getattr(u, 'speaker', '?')}] {u.text}" for u in utterances
    )


def transcribe_one(path: Path, keyterms: list[str] | None = None) -> None:
    print(f"\n=== {path.name} ===", flush=True)
    config_kwargs = dict(
        speech_model=aai.SpeechModel.best,
        language_code="es",
        speaker_labels=True,
        punctuate=True,
        format_text=True,
    )
    if keyterms:
        config_kwargs["word_boost"] = keyterms
        config_kwargs["boost_param"] = "high"
    config = aai.TranscriptionConfig(**config_kwargs)
    transcript = aai.Transcriber(config=config).transcribe(str(path))

    if transcript.status == aai.TranscriptStatus.error:
        print(f"  ERROR: {transcript.error}", flush=True)
        raise RuntimeError(transcript.error)

    stem = path.with_suffix("")
    utts = transcript.utterances or []

    (stem.parent / f"{stem.name}.srt").write_text(_to_srt(utts), encoding="utf-8")
    (stem.parent / f"{stem.name}.txt").write_text(
        _to_plain_text(utts), encoding="utf-8"
    )
    (stem.parent / f"{stem.name}.json").write_text(
        json.dumps(transcript.json_response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dur_s = (transcript.audio_duration or 0)
    print(
        f"  OK  duration={dur_s}s  utterances={len(utts)}  "
        f"-> {stem.name}.srt / .txt / .json",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyterms", type=Path, default=None,
                        help="Path to keyterms file (one term per line)")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    key = os.environ.get("ASSEMBLY_KEY")
    if not key:
        print("ERROR: ASSEMBLY_KEY not set", file=sys.stderr)
        return 2
    aai.settings.api_key = key

    missing = [p for p in args.files if not p.is_file()]
    if missing:
        for p in missing:
            print(f"ERROR: missing file {p}", file=sys.stderr)
        return 2

    keyterms = _load_keyterms(args.keyterms) if args.keyterms else None
    if keyterms:
        print(f"Using {len(keyterms)} keyterms", flush=True)

    for p in args.files:
        transcribe_one(p, keyterms=keyterms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
