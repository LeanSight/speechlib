"""AT: speaker_map_suggestions.json cache — save, load, skip embeddings, formato.

Post behavior change (2026-04-11): el pipeline ya no escribe speaker_map.json
automaticamente con nombres decididos. Escribe speaker_map_suggestions.json con
top_candidates + recommended, y la asignacion de nombres reales queda al
subcomando `confirm`.

Testea _run_speaker_recognition_cached directamente con AudioState real.
Mock solo en boundaries GPU (_get_inference, load_avg_voice_embeddings).
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

    def test_suggestions_saved_after_recognition(self, tmp_path):
        """_run_speaker_recognition_cached guarda speaker_map_suggestions.json."""
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
            result = _run_speaker_recognition_cached(
                state, str(voices_dir), speakers, ["SPEAKER_00"]
            )

        suggestions_path = state.artifacts_dir / "speaker_map_suggestions.json"
        assert suggestions_path.exists()
        saved = json.loads(suggestions_path.read_text(encoding="utf-8"))
        assert "tags" in saved
        assert "SPEAKER_00" in saved["tags"]
        assert "top_candidates" in saved["tags"]["SPEAKER_00"]
        assert "recommended" in saved["tags"]["SPEAKER_00"]
        # Retorno de la función ES el suggestions dict
        assert result == saved

    def test_cache_hit_skips_embeddings(self, tmp_path):
        """Si speaker_map_suggestions.json existe, no se computan embeddings."""
        from speechlib.core_analysis import _run_speaker_recognition_cached

        state = _make_state(tmp_path)
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        # Pre-cache con suggestions válidas + sidecar de params
        cached_suggestions = {
            "threshold": 0.55,
            "min_margin": 0.10,
            "tags": {
                "SPEAKER_00": {
                    "top_candidates": [{"name": "speaker", "score": 0.95}],
                    "recommended": "speaker",
                }
            },
        }
        suggestions_path = state.artifacts_dir / "speaker_map_suggestions.json"
        suggestions_path.write_text(
            json.dumps(cached_suggestions), encoding="utf-8"
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
        assert result == cached_suggestions

    def test_suggestions_recommended_is_none_for_unmatched_tags(self, tmp_path):
        """Tags sin match tienen recommended=None y top_candidates con scores bajos."""
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

        # Ningún tag tiene recommended porque el score es 0 (ortogonal)
        assert result["tags"]["SPEAKER_00"]["recommended"] is None
        assert result["tags"]["SPEAKER_01"]["recommended"] is None
        # "unknown" literal NUNCA aparece en el formato suggestions
        serialized = json.dumps(result)
        assert "unknown" not in serialized

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
