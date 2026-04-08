"""
Slice 16 AT: muestras de speakers no identificados van a por_nombrar/<id>/.

Comportamiento nuevo: extract_speaker_samples organiza la salida por
status de identificacion:
- Identificados: <output>/<nombre>/clip_NN.wav
- No identificados: <output>/por_nombrar/<SPEAKER_XX>/clip_NN.wav

Workflow del usuario: revisar por_nombrar/, escuchar cada SPEAKER_XX,
renombrar al nombre real y mover a voices/<nombre>/ para la siguiente
corrida.

Tests sin mocks: audio sintetico (senoidal) en tmp_path, value objects
del dominio, sin pyannote, sin filesystem fixtures.
"""

from pathlib import Path

import pytest


def _write_synthetic_wav(path: Path, duration_s: float, sample_rate: int = 16000) -> Path:
    import torch
    import torchaudio

    n = int(duration_s * sample_rate)
    t = torch.linspace(0, duration_s, n).unsqueeze(0)
    waveform = 0.1 * torch.sin(2 * torch.pi * 440 * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sample_rate, bits_per_sample=16)
    return path


def test_unidentified_speakers_land_in_por_nombrar_subfolder(tmp_path):
    from speechlib.domain.sample_extraction import SampleClip, SpeakerSamplePlan
    from speechlib.services.extract_samples import extract_speaker_samples

    audio = _write_synthetic_wav(tmp_path / "src.wav", duration_s=10.0)
    out_dir = tmp_path / "samples"

    plans = (
        SpeakerSamplePlan(
            speaker_label="Manuel Olguin",
            is_identified=True,
            clips=(SampleClip(start_ms=0, end_ms=2000),),
        ),
        SpeakerSamplePlan(
            speaker_label="SPEAKER_03",
            is_identified=False,
            clips=(SampleClip(start_ms=3000, end_ms=5000),),
        ),
        SpeakerSamplePlan(
            speaker_label="SPEAKER_07",
            is_identified=False,
            clips=(SampleClip(start_ms=6000, end_ms=8000),),
        ),
    )

    result = extract_speaker_samples(plans, audio, out_dir)

    # Identificado: <output>/<nombre>/
    assert (out_dir / "Manuel Olguin").is_dir()
    assert (out_dir / "Manuel Olguin" / "clip_01.wav").exists()

    # No identificados: <output>/por_nombrar/<SPEAKER_XX>/
    assert (out_dir / "por_nombrar" / "SPEAKER_03").is_dir()
    assert (out_dir / "por_nombrar" / "SPEAKER_03" / "clip_01.wav").exists()
    assert (out_dir / "por_nombrar" / "SPEAKER_07").is_dir()
    assert (out_dir / "por_nombrar" / "SPEAKER_07" / "clip_01.wav").exists()

    # Identificados NO deben aparecer dentro de por_nombrar
    assert not (out_dir / "por_nombrar" / "Manuel Olguin").exists()

    # SPEAKER_XX NO debe aparecer al raiz (solo dentro de por_nombrar)
    assert not (out_dir / "SPEAKER_03").exists()
    assert not (out_dir / "SPEAKER_07").exists()

    # Return refleja la estructura real en disco
    assert sorted(result["Manuel Olguin"]) == [out_dir / "Manuel Olguin" / "clip_01.wav"]
    assert sorted(result["SPEAKER_03"]) == [out_dir / "por_nombrar" / "SPEAKER_03" / "clip_01.wav"]
    assert sorted(result["SPEAKER_07"]) == [out_dir / "por_nombrar" / "SPEAKER_07" / "clip_01.wav"]


def test_only_identified_speakers_skip_por_nombrar(tmp_path):
    """Si todos los speakers estan identificados, no se crea por_nombrar."""
    from speechlib.domain.sample_extraction import SampleClip, SpeakerSamplePlan
    from speechlib.services.extract_samples import extract_speaker_samples

    audio = _write_synthetic_wav(tmp_path / "a.wav", duration_s=5.0)
    out_dir = tmp_path / "samples"

    plans = (
        SpeakerSamplePlan(
            speaker_label="Manuel",
            is_identified=True,
            clips=(SampleClip(0, 1000),),
        ),
        SpeakerSamplePlan(
            speaker_label="Pamela",
            is_identified=True,
            clips=(SampleClip(1000, 2000),),
        ),
    )

    extract_speaker_samples(plans, audio, out_dir)

    assert (out_dir / "Manuel").is_dir()
    assert (out_dir / "Pamela").is_dir()
    assert not (out_dir / "por_nombrar").exists()


def test_only_unidentified_speakers_all_in_por_nombrar(tmp_path):
    """Si todos son SPEAKER_XX, todos van bajo por_nombrar/."""
    from speechlib.domain.sample_extraction import SampleClip, SpeakerSamplePlan
    from speechlib.services.extract_samples import extract_speaker_samples

    audio = _write_synthetic_wav(tmp_path / "a.wav", duration_s=5.0)
    out_dir = tmp_path / "samples"

    plans = (
        SpeakerSamplePlan(
            speaker_label="SPEAKER_00",
            is_identified=False,
            clips=(SampleClip(0, 1000),),
        ),
        SpeakerSamplePlan(
            speaker_label="SPEAKER_01",
            is_identified=False,
            clips=(SampleClip(1000, 2000),),
        ),
    )

    extract_speaker_samples(plans, audio, out_dir)

    # No hay carpetas SPEAKER_XX al raiz
    assert not (out_dir / "SPEAKER_00").exists()
    assert not (out_dir / "SPEAKER_01").exists()
    # Pero si dentro de por_nombrar
    assert (out_dir / "por_nombrar" / "SPEAKER_00" / "clip_01.wav").exists()
    assert (out_dir / "por_nombrar" / "SPEAKER_01" / "clip_01.wav").exists()
