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

    def test_recognize_reuses_rttm_and_writes_suggestions(self, tmp_path, monkeypatch):
        """recognize carga RTTM existente y escribe speaker_map_suggestions.json."""
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
        suggestions_path = state.artifacts_dir / "speaker_map_suggestions.json"
        assert suggestions_path.exists()
        saved = json.loads(suggestions_path.read_text(encoding="utf-8"))
        assert "tags" in saved
        for tag_data in saved["tags"].values():
            assert "top_candidates" in tag_data
            assert "recommended" in tag_data

    def test_recognize_force_deletes_old_suggestions(self, tmp_path, monkeypatch):
        """--force borra speaker_map_suggestions.json antes de re-ejecutar."""
        from speechlib.__main__ import app

        wav, state = _setup_artifacts(tmp_path)
        voices = tmp_path / "voices"
        voices.mkdir()
        monkeypatch.setenv("HF_TOKEN", "test")

        # Pre-crear suggestions viejas con un nombre que NO debe aparecer tras force
        old_suggestions = {
            "threshold": 0.55,
            "min_margin": 0.10,
            "tags": {
                "SPEAKER_00": {
                    "top_candidates": [{"name": "OldName", "score": 0.9}],
                    "recommended": "OldName",
                },
            },
        }
        (state.artifacts_dir / "speaker_map_suggestions.json").write_text(
            json.dumps(old_suggestions)
        )
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

        new_suggestions = json.loads(
            (state.artifacts_dir / "speaker_map_suggestions.json").read_text()
        )
        # OldName ya no aparece en los top_candidates post-force
        serialized = json.dumps(new_suggestions)
        assert "OldName" not in serialized
