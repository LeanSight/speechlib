"""AT: Pipeline.from_pretrained se cachea entre llamadas (LRU cache).

Reescrito sin mocks de preprocessing — testea _get_diarization_pipeline
directamente. Mock solo en boundary externo (Pipeline.from_pretrained).
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def clear_diarization_cache():
    from speechlib.core_analysis import _get_diarization_pipeline
    _get_diarization_pipeline.cache_clear()
    yield
    _get_diarization_pipeline.cache_clear()


def test_pipeline_loaded_once_for_multiple_calls():
    """_get_diarization_pipeline llamado 2x con mismo token → from_pretrained 1x."""
    from speechlib.core_analysis import _get_diarization_pipeline

    mock_pipeline = MagicMock()
    with patch(
        "speechlib.diarization.Pipeline.from_pretrained",
        return_value=mock_pipeline,
    ) as mock_from_pretrained:
        _get_diarization_pipeline("TOKEN")
        _get_diarization_pipeline("TOKEN")

        assert mock_from_pretrained.call_count == 1


def test_cached_pipeline_returns_same_instance():
    """El cache retorna la misma instancia de pipeline."""
    from speechlib.core_analysis import _get_diarization_pipeline

    mock_pipeline = MagicMock()
    with patch(
        "speechlib.diarization.Pipeline.from_pretrained",
        return_value=mock_pipeline,
    ):
        p1 = _get_diarization_pipeline("TOKEN")
        p2 = _get_diarization_pipeline("TOKEN")

    assert p1 is p2
