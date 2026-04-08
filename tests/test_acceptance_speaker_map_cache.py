"""
Slice 12 AT: speaker_map.json cache
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torchaudio
import torch


def _make_wav(path: Path, duration_s: float = 5.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), torch.zeros(1, n), sr, bits_per_sample=16)
    return path


def _make_annotation_mock(speakers: list[str], duration_s: float = 5.0):
    turns = []
    for i, spk in enumerate(speakers):
        turn = MagicMock()
        turn.start = float(i * (duration_s + 1))
        turn.end = float(i * (duration_s + 1) + duration_s)
        turns.append((turn, None, spk))

    mock_annotation = MagicMock()
    mock_annotation.itertracks.return_value = iter(turns)

    def write_rttm(file_handle):
        for spk in speakers:
            start = 0.0
            file_handle.write(
                f"SPEAKER test 1 {start} {duration_s} <NA> <NA> {spk} <NA> <NA>\n"
            )

    mock_annotation.write_rttm = write_rttm
    return mock_annotation


class TestSpeakerMapCache:
    """Tests for speaker_map.json caching"""

    def test_speaker_map_saved_after_recognition(self, tmp_path):
        """core_analysis saves speaker_map.json after running speaker recognition"""
        from speechlib.core_analysis import core_analysis

        audio = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "speaker").mkdir()
        _make_wav(voices / "speaker" / "voice.wav")

        mock_pipeline = MagicMock()
        mock_diar = MagicMock()
        mock_diar.speaker_diarization = _make_annotation_mock(["SPEAKER_00"])
        mock_pipeline.return_value = mock_diar

        with patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            with patch(
                "speechlib.core_analysis._load_rttm", side_effect=FileNotFoundError()
            ):
                core_analysis(str(audio), str(voices), str(tmp_path / "logs"), "en")

        speaker_map_path = tmp_path / ".audio" / "speaker_map.json"
        assert speaker_map_path.exists(), "speaker_map.json should be saved"

    def test_core_analysis_passes_enhanced_flag_to_voice_library_load(self, tmp_path):
        """Slice 13b: cuando enhance está activo, load_avg_voice_embeddings
        recibe enhanced=True (la libreria carga embeddings de _enhanced/).

        El path canonico ya no llama speaker_recognition() — usa assign_speakers
        del dominio puro. El flag enhanced viaja via load_avg_voice_embeddings."""
        from speechlib.core_analysis import core_analysis

        audio = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "speaker").mkdir()
        _make_wav(voices / "speaker" / "voice.wav")

        mock_pipeline = MagicMock()
        mock_diar = MagicMock()
        mock_diar.speaker_diarization = _make_annotation_mock(["SPEAKER_00"])
        mock_pipeline.return_value = mock_diar

        import numpy as np
        mock_load_lib = MagicMock(return_value={"speaker": np.ones(192)})

        with patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            with patch(
                "speechlib.core_analysis._load_rttm", side_effect=FileNotFoundError()
            ):
                with patch(
                    "speechlib.core_analysis.load_avg_voice_embeddings", mock_load_lib
                ):
                    with patch(
                        "speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                        return_value={"SPEAKER_00": np.ones(192)},
                    ):
                        with patch(
                            "speechlib.core_analysis.enhance_audio",
                            lambda s: s.model_copy(update={"is_enhanced": True}),
                        ):
                            core_analysis(
                                str(audio), str(voices), str(tmp_path / "logs"), "en",
                                skip_enhance=False,
                            )

        mock_load_lib.assert_called_once()
        _, kwargs = mock_load_lib.call_args
        assert kwargs.get("enhanced") is True, (
            f"load_avg_voice_embeddings should receive enhanced=True, got {mock_load_lib.call_args}"
        )

    def test_speaker_map_cache_skips_embeddings_computation(self, tmp_path):
        """Slice 15: speaker_map.json exists → no se computan embeddings nuevos.

        En el flujo nuevo (Slice 13b), el cache evita llamar
        _compute_averaged_embeddings_per_tag y assign_speakers."""
        from speechlib.core_analysis import core_analysis

        audio = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "speaker").mkdir()
        _make_wav(voices / "speaker" / "voice.wav")

        rttm_path = tmp_path / ".audio" / "diarization.rttm"
        rttm_path.parent.mkdir(parents=True)
        rttm_path.write_text(
            "SPEAKER test 1 0.0 5.0 <NA> <NA> SPEAKER_00 <NA> <NA>", encoding="utf-8"
        )

        speaker_map_path = tmp_path / ".audio" / "speaker_map.json"
        speaker_map_path.write_text(
            json.dumps({"SPEAKER_00": "speaker"}), encoding="utf-8"
        )

        mock_annotation = _make_annotation_mock(["SPEAKER_00"])
        mock_rttm = MagicMock(return_value={"test": mock_annotation})
        mock_compute = MagicMock(return_value={})

        with patch("speechlib.core_analysis._load_rttm", mock_rttm):
            with patch(
                "speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                mock_compute,
            ):
                core_analysis(str(audio), str(voices), str(tmp_path / "logs"), "en")

        mock_compute.assert_not_called()

    def test_speaker_map_cache_skips_diarization_and_embeddings(self, tmp_path):
        """diarization.rttm + speaker_map.json exist → neither pipeline ni
        embedding computation se invocan."""
        from speechlib.core_analysis import core_analysis

        audio = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "speaker").mkdir()
        _make_wav(voices / "speaker" / "voice.wav")

        rttm_path = tmp_path / ".audio" / "diarization.rttm"
        rttm_path.parent.mkdir(parents=True)
        rttm_path.write_text(
            "SPEAKER test 1 0.0 5.0 <NA> <NA> SPEAKER_00 <NA> <NA>", encoding="utf-8"
        )

        speaker_map_path = tmp_path / ".audio" / "speaker_map.json"
        speaker_map_path.write_text(
            json.dumps({"SPEAKER_00": "speaker"}), encoding="utf-8"
        )

        mock_annotation = _make_annotation_mock(["SPEAKER_00"])
        mock_rttm = MagicMock(return_value={"test": mock_annotation})
        mock_pipeline = MagicMock()
        mock_compute = MagicMock(return_value={})

        with patch("speechlib.core_analysis._load_rttm", mock_rttm):
            with patch(
                "speechlib.core_analysis._get_diarization_pipeline", mock_pipeline
            ):
                with patch(
                    "speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                    mock_compute,
                ):
                    core_analysis(str(audio), str(voices), str(tmp_path / "logs"), "en")

        mock_pipeline.assert_not_called()
        mock_compute.assert_not_called()

    def test_speaker_map_json_format(self, tmp_path):
        """Slice 13b: JSON tiene SPEAKER_XX como valor cuando ningun voice
        de la libreria supera threshold. JAMAS aparece el literal "unknown".

        Esto verifica el invariante anti-bug a nivel del speaker_map.json:
        ningun valor del dict puede ser "unknown" porque el dominio nuevo
        usa SpeakerIdentity.label que cae al diarization_tag."""
        from speechlib.core_analysis import core_analysis

        audio = _make_wav(tmp_path / "audio.wav", duration_s=10.0)
        voices = tmp_path / "voices"
        voices.mkdir()
        (voices / "speaker").mkdir()
        _make_wav(voices / "speaker" / "voice.wav")

        mock_pipeline = MagicMock()
        mock_diar = MagicMock()
        mock_diar.speaker_diarization = _make_annotation_mock(
            ["SPEAKER_00", "SPEAKER_01"]
        )
        mock_pipeline.return_value = mock_diar

        # Library con un embedding ortogonal a los tags → no match para nadie
        import numpy as np
        no_match_lib = {"someone": np.array([0.0, 0.0, 1.0])}
        embeddings_by_tag = {
            "SPEAKER_00": np.array([1.0, 0.0, 0.0]),
            "SPEAKER_01": np.array([0.0, 1.0, 0.0]),
        }

        with patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_pipeline,
        ):
            with patch(
                "speechlib.core_analysis._load_rttm", side_effect=FileNotFoundError()
            ):
                with patch(
                    "speechlib.core_analysis.load_avg_voice_embeddings",
                    return_value=no_match_lib,
                ):
                    with patch(
                        "speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                        return_value=embeddings_by_tag,
                    ):
                        core_analysis(str(audio), str(voices), str(tmp_path / "logs"), "en")

        speaker_map_path = tmp_path / ".audio" / "speaker_map.json"
        data = json.loads(speaker_map_path.read_text(encoding="utf-8"))

        assert "SPEAKER_00" in data
        assert "SPEAKER_01" in data
        # Invariante anti-bug: ningun valor es "unknown"
        assert "unknown" not in data.values()
        # Sin match -> el value es el SPEAKER_XX original
        assert data["SPEAKER_00"] == "SPEAKER_00"
        assert data["SPEAKER_01"] == "SPEAKER_01"
