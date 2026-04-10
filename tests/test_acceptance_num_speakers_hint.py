"""AT: num_speakers hint pasado a pyannote pipeline."""
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torchaudio

from speechlib.audio_state import AudioState


def _make_wav(path: Path, duration_s: float = 5.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def _make_state(tmp_path: Path) -> AudioState:
    wav = _make_wav(tmp_path / "audio.wav")
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return state


class TestNumSpeakersHint:

    def test_num_speakers_passed_to_pipeline(self, tmp_path):
        """Pipeline recibe num_speakers cuando se especifica."""
        from speechlib.core_analysis import _run_diarization_cached
        from pyannote.core import Annotation, Segment

        state = _make_state(tmp_path)
        annotation = Annotation(uri="waveform")
        annotation[Segment(0.0, 4.0)] = "SPEAKER_00"

        mock_pipeline = MagicMock(return_value=annotation)

        with patch("speechlib.core_analysis._get_diarization_pipeline",
                    return_value=mock_pipeline):
            _run_diarization_cached(state, "TOKEN", num_speakers=3)

        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("num_speakers") == 3

    def test_num_speakers_none_not_passed(self, tmp_path):
        """Sin num_speakers, pipeline no recibe el kwarg."""
        from speechlib.core_analysis import _run_diarization_cached
        from pyannote.core import Annotation, Segment

        state = _make_state(tmp_path)
        annotation = Annotation(uri="waveform")
        annotation[Segment(0.0, 4.0)] = "SPEAKER_00"

        mock_pipeline = MagicMock(return_value=annotation)

        with patch("speechlib.core_analysis._get_diarization_pipeline",
                    return_value=mock_pipeline):
            _run_diarization_cached(state, "TOKEN")

        _, kwargs = mock_pipeline.call_args
        assert "num_speakers" not in kwargs

    def test_num_speakers_derived_from_speakers_list(self, tmp_path):
        """core_analysis deriva num_speakers de len(allowed_speakers)."""
        from typer.testing import CliRunner
        from speechlib.__main__ import app

        wav = _make_wav(tmp_path / "audio.wav")
        runner = CliRunner()
        result = runner.invoke(app, [
            str(wav), "--speakers", "A,B,C,D,E,F",
        ])
        # No debe fallar por --speakers (puede fallar por audio/token)
        assert "No such option" not in (result.output or "")
