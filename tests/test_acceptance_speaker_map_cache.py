"""AT: speaker_map.json cache — save, load, skip embeddings, y formato.

Testea _run_speaker_recognition_cached directamente con AudioState real.
Mock solo en boundaries GPU (_get_inference, load_avg_voice_embeddings).
Tests de formato JSON migrados a test puro de SpeakerIdentity.label.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import torchaudio

from speechlib.audio_state import AudioState


def _make_wav(path: Path, duration_s: float = 5.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def _make_state(tmp_path: Path) -> AudioState:
    wav = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return state


def _make_speakers(tags: list[str], duration_s: float = 5.0) -> dict:
    """Construye dict de speakers como lo hace _build_speaker_groups."""
    return {
        tag: [[float(i * (duration_s + 1)), float(i * (duration_s + 1) + duration_s), tag]]
        for i, tag in enumerate(tags)
    }


class TestSpeakerMapCache:

    def test_speaker_map_saved_after_recognition(self, tmp_path):
        """_run_speaker_recognition_cached guarda speaker_map.json."""
        from speechlib.core_analysis import _run_speaker_recognition_cached

        state = _make_state(tmp_path)
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        speakers = _make_speakers(["SPEAKER_00"])
        mock_lib = {"speaker": np.ones(192)}

        with (
            patch("speechlib.core_analysis.load_avg_voice_embeddings", return_value=mock_lib),
            patch(
                "speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                return_value={"SPEAKER_00": np.ones(192)},
            ),
        ):
            _run_speaker_recognition_cached(
                state, str(voices_dir), speakers, ["SPEAKER_00"]
            )

        speaker_map_path = state.artifacts_dir / "speaker_map.json"
        assert speaker_map_path.exists()

    def test_cache_hit_skips_embeddings(self, tmp_path):
        """Si speaker_map.json existe, no se computan embeddings."""
        from speechlib.core_analysis import _run_speaker_recognition_cached

        state = _make_state(tmp_path)
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        # Crear cache con sidecar de params
        speaker_map_path = state.artifacts_dir / "speaker_map.json"
        speaker_map_path.write_text(
            json.dumps({"SPEAKER_00": "speaker"}), encoding="utf-8"
        )
        params_path = state.artifacts_dir / "speaker_map_params.json"
        params_path.write_text(
            json.dumps({"allowed_speakers": None, "threshold": 0.55, "min_margin": 0.10}),
            encoding="utf-8",
        )

        speakers = _make_speakers(["SPEAKER_00"])

        with patch(
            "speechlib.core_analysis._compute_averaged_embeddings_per_tag"
        ) as mock_compute:
            result = _run_speaker_recognition_cached(
                state, str(voices_dir), speakers, ["SPEAKER_00"]
            )

        mock_compute.assert_not_called()
        assert result == {"SPEAKER_00": "speaker"}

    def test_speaker_map_format_no_unknown_literal(self, tmp_path):
        """speaker_map.json NUNCA contiene el literal 'unknown'.
        Tags sin match conservan SPEAKER_XX."""
        from speechlib.core_analysis import _run_speaker_recognition_cached

        state = _make_state(tmp_path)
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        speakers = _make_speakers(["SPEAKER_00", "SPEAKER_01"])

        # Library con embedding ortogonal a ambos tags → no match
        mock_lib = {"someone": np.array([0.0, 0.0, 1.0])}
        embeddings_by_tag = {
            "SPEAKER_00": np.array([1.0, 0.0, 0.0]),
            "SPEAKER_01": np.array([0.0, 1.0, 0.0]),
        }

        with (
            patch("speechlib.core_analysis.load_avg_voice_embeddings", return_value=mock_lib),
            patch(
                "speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                return_value=embeddings_by_tag,
            ),
        ):
            result = _run_speaker_recognition_cached(
                state, str(voices_dir), speakers, ["SPEAKER_00", "SPEAKER_01"]
            )

        assert "unknown" not in result.values()
        assert result["SPEAKER_00"] == "SPEAKER_00"
        assert result["SPEAKER_01"] == "SPEAKER_01"

    def test_enhanced_flag_propagated(self, tmp_path):
        """Cuando state.is_enhanced=True, load_avg_voice_embeddings recibe enhanced=True."""
        from speechlib.core_analysis import _run_speaker_recognition_cached

        wav = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        state = AudioState(
            source_path=wav, working_path=wav,
            is_wav=True, is_enhanced=True,
        )
        state.artifacts_dir.mkdir(parents=True, exist_ok=True)

        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        (voices_dir / "speaker").mkdir()
        _make_wav(voices_dir / "speaker" / "voice.wav")

        speakers = _make_speakers(["SPEAKER_00"])

        mock_load_lib = MagicMock(return_value={"speaker": np.ones(192)})

        with (
            patch("speechlib.core_analysis.load_avg_voice_embeddings", mock_load_lib),
            patch(
                "speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                return_value={"SPEAKER_00": np.ones(192)},
            ),
        ):
            _run_speaker_recognition_cached(
                state, str(voices_dir), speakers, ["SPEAKER_00"]
            )

        _, kwargs = mock_load_lib.call_args
        assert kwargs.get("enhanced") is True
