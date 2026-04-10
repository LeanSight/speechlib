"""AT: skip_enhance controla si el output _limpio.m4a se enhancee.

ASR (diarization, recognition, transcription) NUNCA usa enhance.
skip_enhance solo afecta el paso de compress para escucha humana.
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


def _run_core(tmp_path, skip_enhance, compress, enhance_mock):
    from speechlib.core_analysis import core_analysis

    wav = _make_wav(tmp_path / "audio.wav")
    annotation = _make_annotation()

    with (
        patch("speechlib.core_analysis.enhance_audio", enhance_mock),
        patch("speechlib.core_analysis._run_diarization_cached", return_value=(annotation, True)),
        patch("speechlib.core_analysis._transcribe_segments", return_value=[]),
        patch("speechlib.core_analysis._group_post_transcription", return_value=[]),
        patch("speechlib.core_analysis.write_log_file"),
        patch("speechlib.core_analysis._publish_domain_artifacts"),
        patch("speechlib.core_analysis._publish_to_source_folder"),
        patch("speechlib.core_analysis.compress_audio", return_value=None),
    ):
        core_analysis(
            file_name=str(wav),
            voices_folder=None,
            log_folder=str(tmp_path / "output"),
            language="es",
            ACCESS_TOKEN="token",
            skip_enhance=skip_enhance,
            compress=compress,
        )


def test_skip_enhance_true_does_not_call_enhance(tmp_path):
    """Con skip_enhance=True y compress=True, enhance_audio no se invoca."""
    mock_enhance = MagicMock()
    _run_core(tmp_path, skip_enhance=True, compress=True, enhance_mock=mock_enhance)
    mock_enhance.assert_not_called()


def test_skip_enhance_false_with_compress_calls_enhance(tmp_path):
    """Con skip_enhance=False y compress=True, enhance_audio se invoca para el output."""
    mock_enhance = MagicMock(side_effect=lambda s: s.model_copy(update={"is_enhanced": True}))
    _run_core(tmp_path, skip_enhance=False, compress=True, enhance_mock=mock_enhance)
    mock_enhance.assert_called_once()


def test_no_compress_no_enhance(tmp_path):
    """Sin compress, enhance nunca se invoca (no hay output que mejorar)."""
    mock_enhance = MagicMock()
    _run_core(tmp_path, skip_enhance=False, compress=False, enhance_mock=mock_enhance)
    mock_enhance.assert_not_called()
