"""AT: speaker_map.json se invalida cuando cambian los parámetros de recognition.

Un sidecar speaker_map_params.json registra los params usados.
Si los params actuales difieren, speaker_map.json se descarta y se recomputa.
"""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torchaudio

from speechlib.audio_state import AudioState


def _make_wav(path: Path, duration_s: float = 10.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def _make_state(tmp_path: Path) -> AudioState:
    wav = _make_wav(tmp_path / "audio.wav")
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return state


def _make_speakers():
    return {"SPEAKER_00": [[0.0, 5.0, "SPEAKER_00"]]}


def _run_recognition(state, tmp_path, allowed_speakers=None):
    from speechlib.core_analysis import _run_speaker_recognition_cached

    voices_dir = tmp_path / "voices"
    voices_dir.mkdir(exist_ok=True)
    speakers = _make_speakers()

    with (
        patch("speechlib.core_analysis.load_avg_voice_embeddings",
              return_value={"speaker": np.ones(192)}),
        patch("speechlib.core_analysis._compute_averaged_embeddings_per_tag",
              return_value={"SPEAKER_00": np.ones(192)}),
    ):
        return _run_speaker_recognition_cached(
            state, str(voices_dir), speakers, ["SPEAKER_00"],
            allowed_speakers=allowed_speakers,
        )


class TestCacheInvalidation:

    def test_speaker_map_invalidated_when_speakers_change(self, tmp_path):
        """Cambiar --speakers invalida el cache."""
        state = _make_state(tmp_path)

        # Run 1: speakers=["Alice"]
        _run_recognition(state, tmp_path, allowed_speakers=["Alice"])
        assert (state.artifacts_dir / "speaker_map.json").exists()
        assert (state.artifacts_dir / "speaker_map_params.json").exists()

        # Run 2: speakers=["Alice", "Bob"] → debe recomputar
        with patch("speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                    return_value={"SPEAKER_00": np.ones(192)}) as mock_compute:
            with patch("speechlib.core_analysis.load_avg_voice_embeddings",
                        return_value={"speaker": np.ones(192)}):
                from speechlib.core_analysis import _run_speaker_recognition_cached
                _run_speaker_recognition_cached(
                    state, str(tmp_path / "voices"),
                    _make_speakers(), ["SPEAKER_00"],
                    allowed_speakers=["Alice", "Bob"],
                )
            mock_compute.assert_called_once()

    def test_speaker_map_reused_when_params_unchanged(self, tmp_path):
        """Mismos params → cache hit, no recomputa."""
        state = _make_state(tmp_path)

        _run_recognition(state, tmp_path, allowed_speakers=["Alice"])

        with patch("speechlib.core_analysis._compute_averaged_embeddings_per_tag") as mock_compute:
            from speechlib.core_analysis import _run_speaker_recognition_cached
            _run_speaker_recognition_cached(
                state, str(tmp_path / "voices"),
                _make_speakers(), ["SPEAKER_00"],
                allowed_speakers=["Alice"],
            )
            mock_compute.assert_not_called()

    def test_params_json_records_speakers_and_threshold(self, tmp_path):
        """speaker_map_params.json contiene speakers y threshold."""
        state = _make_state(tmp_path)
        _run_recognition(state, tmp_path, allowed_speakers=["Alice", "Bob"])

        params = json.loads(
            (state.artifacts_dir / "speaker_map_params.json").read_text(encoding="utf-8")
        )
        assert sorted(params["allowed_speakers"]) == ["Alice", "Bob"]
        assert "threshold" in params
        assert "min_margin" in params
