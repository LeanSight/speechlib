"""AT: build_score_matrix (puro dominio) + diagnose subcommand."""
import json
import numpy as np

from speechlib.domain.recognition import build_score_matrix


def _unit(dim: int, index: int) -> np.ndarray:
    v = np.zeros(dim)
    v[index] = 1.0
    return v


class TestBuildScoreMatrix:

    def test_returns_scores_per_tag_per_voice(self):
        embeddings_by_tag = {
            "SPEAKER_00": _unit(3, 0),
            "SPEAKER_01": _unit(3, 1),
        }
        voice_library = {
            "Alice": _unit(3, 0),
            "Bob": _unit(3, 1),
        }
        result = build_score_matrix(
            embeddings_by_tag, voice_library, threshold=0.5, min_margin=0.1
        )

        assert "SPEAKER_00" in result["tags"]
        assert "SPEAKER_01" in result["tags"]
        # SPEAKER_00 is unit(0) → cos sim with Alice=unit(0) should be 1.0
        spk0 = result["tags"]["SPEAKER_00"]
        assert spk0["scores"]["Alice"] > 0.99
        assert spk0["scores"]["Bob"] < 0.01

    def test_contains_decision_and_threshold(self):
        embeddings_by_tag = {"SPEAKER_00": _unit(3, 0)}
        voice_library = {"Alice": _unit(3, 0)}
        result = build_score_matrix(
            embeddings_by_tag, voice_library, threshold=0.5, min_margin=0.1
        )

        assert result["threshold"] == 0.5
        assert result["min_margin"] == 0.1
        assert result["tags"]["SPEAKER_00"]["decision"] == "Alice"

    def test_no_match_decision_is_tag(self):
        embeddings_by_tag = {"SPEAKER_00": _unit(3, 0)}
        voice_library = {"Bob": _unit(3, 1)}  # orthogonal → cos sim = 0
        result = build_score_matrix(
            embeddings_by_tag, voice_library, threshold=0.5, min_margin=0.1
        )

        assert result["tags"]["SPEAKER_00"]["decision"] == "SPEAKER_00"

    def test_result_is_json_serializable(self):
        embeddings_by_tag = {"SPEAKER_00": _unit(3, 0)}
        voice_library = {"Alice": _unit(3, 0)}
        result = build_score_matrix(
            embeddings_by_tag, voice_library, threshold=0.5, min_margin=0.1
        )
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
