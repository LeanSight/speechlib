"""AT: skip_enhance=True omite enhance_audio en el preprocessing.

Testea _preprocess_audio directamente con WAV real en tmp_path.
Mock solo de enhance_audio (boundary GPU ClearVoice).
"""
from pathlib import Path
from unittest.mock import patch

import torch
import torchaudio


def _make_wav(path: Path, duration_s: float = 1.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def test_skip_enhance_true_does_not_call_enhance(tmp_path):
    """Con skip_enhance=True, enhance_audio no se invoca."""
    from speechlib.core_analysis import _preprocess_audio

    wav = _make_wav(tmp_path / "audio.wav")

    with patch("speechlib.core_analysis.enhance_audio") as mock_enhance:
        state = _preprocess_audio(str(wav), skip_enhance=True)

    mock_enhance.assert_not_called()
    assert not state.is_enhanced


def test_skip_enhance_false_calls_enhance(tmp_path):
    """Sin skip_enhance (default), enhance_audio se invoca."""
    from speechlib.core_analysis import _preprocess_audio

    wav = _make_wav(tmp_path / "audio.wav")

    with patch(
        "speechlib.core_analysis.enhance_audio",
        side_effect=lambda s: s.model_copy(update={"is_enhanced": True}),
    ) as mock_enhance:
        state = _preprocess_audio(str(wav), skip_enhance=False)

    mock_enhance.assert_called_once()
    assert state.is_enhanced
