"""AT: diarization usa pyannote community-1 con token= (no use_auth_token=).

Testea get_diarization_pipeline directamente. Mock solo de
Pipeline.from_pretrained (boundary GPU externo).
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def clear_cache():
    from speechlib.diarization import get_diarization_pipeline
    get_diarization_pipeline.cache_clear()
    yield
    get_diarization_pipeline.cache_clear()


def test_uses_community1_model():
    """Pipeline.from_pretrained recibe modelo community-1."""
    from speechlib.diarization import get_diarization_pipeline

    with patch("speechlib.diarization.Pipeline.from_pretrained") as mock_fp:
        mock_fp.return_value = MagicMock()
        get_diarization_pipeline("MY_TOKEN")

        args, kwargs = mock_fp.call_args
        assert args[0] == "pyannote/speaker-diarization-community-1"


def test_uses_token_parameter_not_use_auth_token():
    """Usa token= (pyannote 4.x), no use_auth_token= (3.x deprecated)."""
    from speechlib.diarization import get_diarization_pipeline

    with patch("speechlib.diarization.Pipeline.from_pretrained") as mock_fp:
        mock_fp.return_value = MagicMock()
        get_diarization_pipeline("MY_TOKEN")

        _, kwargs = mock_fp.call_args
        assert kwargs.get("token") == "MY_TOKEN"
        assert "use_auth_token" not in kwargs
