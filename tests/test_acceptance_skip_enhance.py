"""AT: skip_enhance=True omite enhance_audio en el pipeline.

Testea core_analysis con skip_enhance flag. Mock de GPU boundaries.
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torchaudio
from pyannote.core import Annotation, Segment


def _make_wav(path: Path, duration_s: float = 1.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def _make_annotation() -> Annotation:
    a = Annotation(uri="waveform")
    a[Segment(0.0, 1.0)] = "SPEAKER_00"
    return a


def _run_core(tmp_path, skip_enhance, enhance_mock):
    """Helper: corre core_analysis con mocks de GPU."""
    from speechlib.core_analysis import core_analysis

    wav = _make_wav(tmp_path / "audio.wav")
    annotation = _make_annotation()
    mock_pipeline = MagicMock(return_value=annotation)

    with (
        patch("speechlib.core_analysis.enhance_audio", enhance_mock),
        patch("speechlib.core_analysis._run_diarization_cached", return_value=(annotation, True)),
        patch("speechlib.core_analysis._transcribe_segments", return_value=[]),
        patch("speechlib.core_analysis._group_post_transcription", return_value=[]),
        patch("speechlib.core_analysis.write_log_file"),
        patch("speechlib.core_analysis._publish_domain_artifacts"),
        patch("speechlib.core_analysis._publish_to_source_folder"),
    ):
        core_analysis(
            file_name=str(wav),
            voices_folder=None,
            log_folder=str(tmp_path / "output"),
            language="es",
            ACCESS_TOKEN="token",
            skip_enhance=skip_enhance,
        )


def test_skip_enhance_true_does_not_call_enhance(tmp_path):
    """Con skip_enhance=True, enhance_audio no se invoca."""
    mock_enhance = MagicMock()
    _run_core(tmp_path, skip_enhance=True, enhance_mock=mock_enhance)
    mock_enhance.assert_not_called()


def test_skip_enhance_false_calls_enhance(tmp_path):
    """Sin skip_enhance (default), enhance_audio se invoca."""
    mock_enhance = MagicMock(side_effect=lambda s: s.model_copy(update={"is_enhanced": True}))
    _run_core(tmp_path, skip_enhance=False, enhance_mock=mock_enhance)
    mock_enhance.assert_called_once()
