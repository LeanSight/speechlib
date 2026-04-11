"""Unit tests: speechlib/loudnorm.py"""
import torchaudio
from pathlib import Path
from conftest import make_wav, make_tone_wav
from speechlib.loudnorm import loudnorm
from speechlib.audio_state import AudioState

TARGET_LUFS = -14.0


def _state(path: Path) -> AudioState:
    return AudioState(source_path=path, working_path=path,
                      is_wav=True, is_mono=True, is_16bit=True, is_16khz=True)


def test_silent_audio_passes_through(tmp_path):
    """Audio silencioso → is_normalized=True sin modificar el archivo."""
    wav = make_wav(tmp_path / "audio.wav", framerate=16000, n_frames=16000)
    state = _state(wav)
    result = loudnorm(state)
    assert result.is_normalized is True
    assert result.working_path == state.working_path


def test_tone_is_normalized_to_target_lufs(tmp_path):
    """Tono a amplitud baja → LUFS dentro de ±1 dB del target -14."""
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.05, duration_s=2.0)
    result = loudnorm(_state(wav))

    waveform, sr = torchaudio.load(str(result.working_path))
    measured = torchaudio.functional.loudness(waveform, sr).item()
    assert abs(measured - TARGET_LUFS) < 1.0


def test_source_path_unchanged(tmp_path):
    """source_path no cambia nunca."""
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.05, duration_s=1.0)
    state = _state(wav)
    result = loudnorm(state)
    assert result.source_path == state.source_path


def test_working_path_unchanged(tmp_path):
    """loudnorm opera in-place: working_path no cambia."""
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.05, duration_s=1.0)
    state = _state(wav)
    result = loudnorm(state)
    assert result.working_path == state.working_path


def test_output_within_true_peak(tmp_path):
    """Ninguna muestra supera el true peak de -1 dBFS."""
    wav = make_tone_wav(tmp_path / "audio.wav", amplitude=0.9, duration_s=2.0)
    result = loudnorm(_state(wav))

    waveform, _ = torchaudio.load(str(result.working_path))
    true_peak_linear = 10 ** (-1.0 / 20)
    assert waveform.abs().max().item() <= true_peak_linear + 1e-4


def test_quiet_audio_with_transient_peak_is_not_clipped(tmp_path):
    """LUFS baja + pico transitorio alto: el gain aplicado es uniforme
    (cap gain), no hay hard-clipping que aplaste el transitorio."""
    import math
    import torch

    sr = 16000
    t = torch.arange(sr * 2) / sr
    quiet_amp = 0.02
    transient_amp = 0.85
    signal = quiet_amp * torch.sin(2 * math.pi * 1000 * t)
    signal[100:150] = transient_amp
    wav_path = tmp_path / "audio.wav"
    torchaudio.save(str(wav_path), signal.unsqueeze(0), sr, bits_per_sample=16)

    result = loudnorm(_state(wav_path))
    waveform, _ = torchaudio.load(str(result.working_path))

    true_peak = 10 ** (-1.0 / 20)
    assert waveform.abs().max().item() <= true_peak + 1e-4

    # Gain efectivo: peak del output dividido por peak del input, en dos
    # zonas (transitorio y quieta). Deben coincidir: gain uniforme = sin
    # clipping. Hard-clip hace el transitorio artificialmente bajo → los
    # gains divergen.
    transient_out = waveform[0, 100:150].abs().max().item()
    quiet_out = waveform[0, sr:].abs().max().item()
    gain_transient = transient_out / transient_amp
    gain_quiet = quiet_out / quiet_amp
    assert abs(gain_transient - gain_quiet) < 0.01, (
        f"gain no uniforme: transient={gain_transient:.4f} "
        f"vs quiet={gain_quiet:.4f} (hard-clipping)"
    )
