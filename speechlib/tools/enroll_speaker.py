"""
Enroll a speaker to the voices library.

Selects the best clips from a directory of speaker clips and copies them
to the voices library, using embedding-based filtering for quality.

Usage:
    python -m speechlib.tools.enroll_speaker \\
        CLIPS_DIR SPEAKER_NAME VOICES_DIR \\
        [--min-duration 2.0] \\
        [--max-clips 5]
"""

import shutil
from pathlib import Path

import soundfile as sf

from speechlib.speaker_recognition import get_embedding
from speechlib.domain.enrollment import reject_outlier_indices, select_diverse_indices


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file in seconds."""
    info = sf.info(str(audio_path))
    return info.duration


def select_clips(
    clips_dir: Path,
    speaker_name: str,
    voices_dir: Path,
    min_duration_s: float = 2.0,
    max_clips: int = 5,
) -> list[Path]:
    """Select clips by duration and copy to voices library."""
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    clips = list(clips_dir.glob("*.wav"))
    if not clips:
        return []

    valid_clips = [c for c in clips if get_audio_duration(c) >= min_duration_s]
    selected = valid_clips[:max_clips]

    speaker_dir = voices_dir / speaker_name
    speaker_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in selected:
        dst = speaker_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)

    return copied


def enroll_speaker(
    clips_dir: Path,
    speaker_name: str,
    voices_dir: Path,
    min_duration_s: float = 2.0,
    max_clips: int = 5,
) -> list[Path]:
    """Enroll a speaker: select best clips and copy to voices library."""
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    clips = sorted(clips_dir.glob("*.wav"))
    if not clips:
        return []

    durations = [get_audio_duration(c) for c in clips]
    valid_clips = [c for c, d in zip(clips, durations) if d >= min_duration_s]

    if not valid_clips:
        return []

    embeddings = [get_embedding(str(c)) for c in valid_clips]

    kept_indices = reject_outlier_indices(embeddings)
    filtered_clips = [valid_clips[i] for i in kept_indices]
    filtered_embeddings = [embeddings[i] for i in kept_indices]

    diverse_indices = select_diverse_indices(filtered_embeddings, max_clips)
    selected = [filtered_clips[i] for i in diverse_indices]

    speaker_dir = voices_dir / speaker_name
    speaker_dir.mkdir(parents=True, exist_ok=True)

    existing = list(speaker_dir.glob("*.wav"))
    start_idx = len(existing) + 1

    copied = []
    for i, src in enumerate(selected, start=start_idx):
        dst = speaker_dir / f"segment_{i:02d}.wav"
        shutil.copy2(src, dst)
        copied.append(dst)

    return copied


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips_dir", type=Path, help="Directory with speaker clips")
    parser.add_argument("speaker_name", help="Name for the speaker in voices library")
    parser.add_argument("voices_dir", type=Path, help="Voices library directory")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum clip duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=5,
        help="Maximum number of clips to enroll (default: 5)",
    )

    args = parser.parse_args()

    speaker_dir = args.voices_dir / args.speaker_name
    existing_count = len(list(speaker_dir.glob("*.wav"))) if speaker_dir.exists() else 0

    copied = enroll_speaker(
        clips_dir=args.clips_dir,
        speaker_name=args.speaker_name,
        voices_dir=args.voices_dir,
        min_duration_s=args.min_duration,
        max_clips=args.max_clips,
    )

    if existing_count > 0:
        print(
            f"Speaker '{args.speaker_name}' already exists — adding {len(copied)} clips"
        )
    else:
        print(f"Enrolled {len(copied)} clips for '{args.speaker_name}'")


if __name__ == "__main__":
    main()
