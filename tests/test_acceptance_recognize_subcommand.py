"""AT: speechlib recognize re-ejecuta solo recognition sobre RTTM existente."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torchaudio
from pyannote.core import Annotation, Segment
from typer.testing import CliRunner

from speechlib.audio_state import AudioState

runner = CliRunner()


def _make_wav(path: Path, duration_s: float = 10.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def _setup_artifacts(tmp_path):
    """Crea audio + RTTM + artifacts dir como si diarization ya hubiera corrido."""
    wav = _make_wav(tmp_path / "audio.wav")
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)

    annotation = Annotation(uri="waveform")
    annotation[Segment(0.0, 5.0)] = "SPEAKER_00"
    annotation[Segment(5.0, 10.0)] = "SPEAKER_01"
    rttm_path = state.artifacts_dir / "diarization.rttm"
    with open(rttm_path, "w") as f:
        annotation.write_rttm(f)

    return wav, state


class TestRecognizeSubcommand:

    def test_recognize_reuses_rttm_and_writes_speaker_map(self, tmp_path, monkeypatch):
        """recognize carga RTTM existente y escribe speaker_map.json."""
        from speechlib.__main__ import app

        wav, state = _setup_artifacts(tmp_path)
        voices = tmp_path / "voices"
        voices.mkdir()
        monkeypatch.setenv("HF_TOKEN", "test")

        with (
            patch("speechlib.core_analysis.load_avg_voice_embeddings",
                  return_value={"Alice": np.ones(192)}),
            patch("speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                  return_value={"SPEAKER_00": np.ones(192), "SPEAKER_01": np.zeros(192)}),
            patch("speechlib.core_analysis._get_diarization_pipeline") as mock_pipeline,
        ):
            result = runner.invoke(app, [
                "recognize", str(wav),
                "--voices-folder", str(voices),
                "--speakers", "Alice",
            ])

        mock_pipeline.assert_not_called()
        speaker_map_path = state.artifacts_dir / "speaker_map.json"
        assert speaker_map_path.exists()

    def test_recognize_force_deletes_old_speaker_map(self, tmp_path, monkeypatch):
        """--force borra speaker_map.json antes de re-ejecutar."""
        from speechlib.__main__ import app

        wav, state = _setup_artifacts(tmp_path)
        voices = tmp_path / "voices"
        voices.mkdir()
        monkeypatch.setenv("HF_TOKEN", "test")

        # Pre-crear speaker_map viejo
        old_map = {"SPEAKER_00": "OldName", "SPEAKER_01": "SPEAKER_01"}
        (state.artifacts_dir / "speaker_map.json").write_text(json.dumps(old_map))
        (state.artifacts_dir / "speaker_map_params.json").write_text(
            json.dumps({"allowed_speakers": ["OldName"], "threshold": 0.55, "min_margin": 0.10})
        )

        with (
            patch("speechlib.core_analysis.load_avg_voice_embeddings",
                  return_value={"NewName": np.ones(192)}),
            patch("speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                  return_value={"SPEAKER_00": np.ones(192), "SPEAKER_01": np.zeros(192)}),
        ):
            result = runner.invoke(app, [
                "recognize", str(wav),
                "--voices-folder", str(voices),
                "--speakers", "NewName",
                "--force",
            ])

        new_map = json.loads((state.artifacts_dir / "speaker_map.json").read_text())
        assert "OldName" not in new_map.values()
