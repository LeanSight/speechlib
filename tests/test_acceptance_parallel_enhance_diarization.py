"""AT: diarization corre sobre audio pre-enhance (post-loudnorm).

Enhance produce audio para transcription, pero diarization no necesita
audio enhanced. Esto permite separar las responsabilidades aunque corran
secuencialmente (overlap GPU descartado por contención de SMs).
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torchaudio
from pyannote.core import Annotation, Segment

from speechlib.audio_state import AudioState


def _make_wav(path: Path, duration_s: float = 5.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def _make_annotation() -> Annotation:
    a = Annotation(uri="waveform")
    a[Segment(0.0, 4.0)] = "SPEAKER_00"
    return a


class TestDiarizationUsesPreEnhanceAudio:

    def test_diarization_receives_loudnorm_path_not_enhanced(self, tmp_path):
        """_run_diarization_cached recibe el state pre-enhance."""
        from speechlib.core_analysis import core_analysis

        wav = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        annotation = _make_annotation()

        diarize_state_paths = []

        def capture_diarization(state, token, **kwargs):
            diarize_state_paths.append(str(state.working_path))
            rttm = state.artifacts_dir / "diarization.rttm"
            with open(rttm, "w") as f:
                annotation.write_rttm(f)
            return annotation, False

        with (
            patch("speechlib.core_analysis._run_diarization_cached", side_effect=capture_diarization),
            patch("speechlib.core_analysis.enhance_audio", side_effect=lambda s: s.model_copy(update={"is_enhanced": True, "working_path": tmp_path / "enhanced.wav"})),
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
                skip_enhance=False,
            )

        assert len(diarize_state_paths) == 1
        assert "enhanced" not in diarize_state_paths[0]
