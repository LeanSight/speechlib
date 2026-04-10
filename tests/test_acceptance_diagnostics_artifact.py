"""AT: recognition_diagnostics.json se guarda después de speaker recognition."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torchaudio

from speechlib.audio_state import AudioState


def _make_state(tmp_path: Path) -> AudioState:
    wav = tmp_path / "audio.wav"
    n = int(10.0 * 16000)
    torchaudio.save(str(wav), torch.zeros(1, n), 16000, bits_per_sample=16)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return state


class TestDiagnosticsArtifact:

    def test_diagnostics_json_saved_after_recognition(self, tmp_path):
        from speechlib.core_analysis import _run_speaker_recognition_cached

        state = _make_state(tmp_path)
        speakers = {"SPEAKER_00": [[0.0, 5.0, "SPEAKER_00"]]}

        with (
            patch("speechlib.core_analysis.load_avg_voice_embeddings",
                  return_value={"Alice": np.ones(192)}),
            patch("speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                  return_value={"SPEAKER_00": np.ones(192)}),
        ):
            _run_speaker_recognition_cached(
                state, str(tmp_path / "voices"), speakers, ["SPEAKER_00"]
            )

        diag_path = state.artifacts_dir / "recognition_diagnostics.json"
        assert diag_path.exists()

    def test_diagnostics_contains_scores_and_decisions(self, tmp_path):
        from speechlib.core_analysis import _run_speaker_recognition_cached

        state = _make_state(tmp_path)
        speakers = {"SPEAKER_00": [[0.0, 5.0, "SPEAKER_00"]]}

        with (
            patch("speechlib.core_analysis.load_avg_voice_embeddings",
                  return_value={"Alice": np.ones(192)}),
            patch("speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                  return_value={"SPEAKER_00": np.ones(192)}),
        ):
            _run_speaker_recognition_cached(
                state, str(tmp_path / "voices"), speakers, ["SPEAKER_00"]
            )

        diag = json.loads(
            (state.artifacts_dir / "recognition_diagnostics.json").read_text(encoding="utf-8")
        )
        assert "threshold" in diag
        assert "tags" in diag
        assert "SPEAKER_00" in diag["tags"]
        assert "scores" in diag["tags"]["SPEAKER_00"]
        assert "decision" in diag["tags"]["SPEAKER_00"]
