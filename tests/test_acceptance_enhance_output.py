"""AT: enhance se aplica al output _limpio.m4a, no al pipeline ASR.

Diarization y transcription siempre usan audio sin enhance.
Enhance solo se usa para generar el audio comprimido de escucha humana.
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torchaudio
from pyannote.core import Annotation, Segment


def _make_wav(path: Path, duration_s: float = 5.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def _make_annotation() -> Annotation:
    a = Annotation(uri="waveform")
    a[Segment(0.0, 4.0)] = "SPEAKER_00"
    return a


def _run_pipeline(tmp_path, *, skip_enhance, compress):
    """Corre core_analysis con mocks de GPU, captura qué state recibió cada paso."""
    from speechlib.core_analysis import core_analysis

    wav = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
    annotation = _make_annotation()

    transcribe_paths = []
    recognition_paths = []

    def capture_transcribe(state, *args, **kwargs):
        transcribe_paths.append(str(state.working_path))
        return []

    def capture_recognition(state, *args, **kwargs):
        recognition_paths.append(str(state.working_path))
        return {"SPEAKER_00": "SPEAKER_00"}

    with (
        patch("speechlib.core_analysis.enhance_audio",
              side_effect=lambda s: s.model_copy(update={"is_enhanced": True, "working_path": tmp_path / "enhanced.wav"})),
        patch("speechlib.core_analysis._run_diarization_cached", return_value=(annotation, True)),
        patch("speechlib.core_analysis._transcribe_segments", side_effect=capture_transcribe),
        patch("speechlib.core_analysis._run_speaker_recognition_cached", side_effect=capture_recognition),
        patch("speechlib.core_analysis._group_post_transcription", return_value=[]),
        patch("speechlib.core_analysis.write_log_file"),
        patch("speechlib.core_analysis._publish_domain_artifacts"),
        patch("speechlib.core_analysis._publish_to_source_folder"),
        patch("speechlib.core_analysis.compress_audio", return_value=None),
    ):
        core_analysis(
            file_name=str(wav),
            voices_folder=str(tmp_path / "voices"),
            log_folder=str(tmp_path / "output"),
            language="es",
            ACCESS_TOKEN="token",
            skip_enhance=skip_enhance,
            compress=compress,
        )

    return transcribe_paths, recognition_paths


class TestEnhanceOnlyForOutput:

    def test_transcription_never_uses_enhanced_audio(self, tmp_path):
        """Transcription siempre recibe audio sin enhance."""
        (tmp_path / "voices").mkdir()
        paths, _ = _run_pipeline(tmp_path, skip_enhance=False, compress=True)
        assert len(paths) == 1
        assert "enhanced" not in paths[0]

    def test_recognition_never_uses_enhanced_audio(self, tmp_path):
        """Speaker recognition siempre recibe audio sin enhance."""
        (tmp_path / "voices").mkdir()
        _, paths = _run_pipeline(tmp_path, skip_enhance=False, compress=True)
        assert len(paths) == 1
        assert "enhanced" not in paths[0]
