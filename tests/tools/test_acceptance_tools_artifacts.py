"""
AT: tools usan unknown_output_dir/ para extraer clips de speakers no identificados.
"""

from pathlib import Path
from unittest.mock import patch
import json
import pytest
from conftest import make_tone_wav


def _make_voices_dir(base: Path) -> Path:
    voices = base / "voices"
    (voices / "Agustin").mkdir(parents=True)
    make_tone_wav(voices / "Agustin" / "segment_01.wav", duration_s=2.0)
    return voices


def test_batch_process_unknown_clips_go_to_unknown_output_dir(tmp_path):
    """batch_process extrae unknown speakers a unknown_output_dir (default voices/_unknown/)."""
    from speechlib.tools.batch_process import batch_process
    from speechlib.domain.transcript import (
        SpeakerIdentity,
        Transcript,
        TranscriptSegment,
    )

    folder = tmp_path / "session"
    folder.mkdir()
    audio = make_tone_wav(folder / "meeting.wav", duration_s=10.0)
    voices = _make_voices_dir(tmp_path)

    # Pre-create artifacts_dir and a transcript.json with one unknown speaker
    artifacts_dir = folder / f".{audio.stem}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(
        segments=(
            TranscriptSegment(
                start_ms=0,
                end_ms=5000,
                text="hola",
                speaker=SpeakerIdentity(diarization_tag="SPEAKER_00"),
            ),
        ),
        audio_path=str(audio),
        language="es",
    )
    transcript.save(artifacts_dir / "transcript.json")

    # core_analysis mocked (requires GPU/HF): returns one unknown segment
    with patch("speechlib.tools.batch_process.core_analysis") as mock_ca:
        mock_ca.return_value = [[0.0, 5.0, "hola", "SPEAKER_00"]]
        report = batch_process(
            folders=[folder],
            voices_folder=voices,
            language="es",
            access_token="fake_token",
        )

    unknown_dir = voices.parent / "_unknown" / audio.stem
    assert unknown_dir.exists(), (
        f"unknown_output_dir not created at {unknown_dir}"
    )
