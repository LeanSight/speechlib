"""AT: diarization.rttm cache — save, load, y skip de pipeline.

Testea _run_diarization_cached directamente con AudioState real en tmp_path.
Mock solo en boundary GPU (_get_diarization_pipeline).
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


def _make_state(tmp_path: Path) -> AudioState:
    wav = _make_wav(tmp_path / "audio.wav")
    state = AudioState(source_path=wav, working_path=wav, is_wav=True)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return state


def _make_real_annotation(*speakers: str) -> Annotation:
    """Annotation real de pyannote (no mock) con un segmento por speaker."""
    a = Annotation(uri="waveform")
    for i, spk in enumerate(speakers):
        a[Segment(float(i * 5), float(i * 5 + 4))] = spk
    return a


class TestRttmCache:

    def test_rttm_saved_after_diarization(self, tmp_path):
        """Primera corrida sin cache → pipeline corre y guarda RTTM."""
        from speechlib.core_analysis import _run_diarization_cached

        state = _make_state(tmp_path)
        annotation = _make_real_annotation("SPEAKER_00", "SPEAKER_01")

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = annotation

        with patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            result_ann, from_cache = _run_diarization_cached(state, "TOKEN")

        rttm_path = state.artifacts_dir / "diarization.rttm"
        assert rttm_path.exists()
        assert not from_cache
        # Verifica que la annotation retornada tiene los speakers
        tracks = list(result_ann.itertracks(yield_label=True))
        labels = {t[2] for t in tracks}
        assert "SPEAKER_00" in labels
        assert "SPEAKER_01" in labels

    def test_rttm_cache_skips_pipeline_call(self, tmp_path):
        """RTTM en disco → pipeline NO se invoca, annotation se carga de cache."""
        from speechlib.core_analysis import _run_diarization_cached

        state = _make_state(tmp_path)

        # Crear RTTM real en disco
        state.artifacts_dir.mkdir(parents=True, exist_ok=True)
        rttm_path = state.artifacts_dir / "diarization.rttm"
        annotation = _make_real_annotation("SPEAKER_00")
        with open(rttm_path, "w") as f:
            annotation.write_rttm(f)

        mock_pipeline = MagicMock()

        with patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            _, from_cache = _run_diarization_cached(state, "TOKEN")

        assert from_cache
        mock_pipeline.assert_not_called()

    def test_rttm_roundtrip_preserves_speakers(self, tmp_path):
        """Save → load roundtrip preserva tags y timestamps."""
        from speechlib.core_analysis import _run_diarization_cached

        state = _make_state(tmp_path)
        annotation = _make_real_annotation("SPEAKER_00", "SPEAKER_01")

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = annotation

        with patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            # Primera corrida: guarda RTTM
            _run_diarization_cached(state, "TOKEN")

        # Segunda corrida: lee de cache
        with patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=MagicMock(),
        ):
            loaded_ann, from_cache = _run_diarization_cached(state, "TOKEN")

        assert from_cache
        tracks = list(loaded_ann.itertracks(yield_label=True))
        labels = {t[2] for t in tracks}
        assert "SPEAKER_00" in labels
        assert "SPEAKER_01" in labels
