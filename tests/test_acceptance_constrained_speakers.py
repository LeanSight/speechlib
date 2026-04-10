"""AT: constrained speaker recognition — filtrar library a speakers esperados.

GOOS puro: test con vectores numpy sintéticos, zero I/O, zero mocks.
"""
import numpy as np

from speechlib.domain.recognition import (
    assign_speakers,
    filter_voice_library,
)
from speechlib.domain.transcript import (
    SpeakerIdentity,
    Transcript,
    TranscriptSegment,
)


def _unit(dim: int, index: int) -> np.ndarray:
    """Vector unitario en la dirección `index`."""
    v = np.zeros(dim)
    v[index] = 1.0
    return v


class TestFilterVoiceLibrary:

    def test_filter_excludes_absent_speakers(self):
        library = {
            "Alice": _unit(3, 0),
            "Bob": _unit(3, 1),
            "Carlos": _unit(3, 2),
        }
        filtered = filter_voice_library(library, allowed_names={"Alice", "Bob"})
        assert set(filtered.keys()) == {"Alice", "Bob"}

    def test_filter_with_none_returns_full_library(self):
        library = {
            "Alice": _unit(3, 0),
            "Bob": _unit(3, 1),
        }
        filtered = filter_voice_library(library, allowed_names=None)
        assert set(filtered.keys()) == {"Alice", "Bob"}

    def test_filter_with_empty_set_returns_empty(self):
        library = {"Alice": _unit(3, 0)}
        filtered = filter_voice_library(library, allowed_names=set())
        assert filtered == {}

    def test_filter_ignores_names_not_in_library(self):
        library = {"Alice": _unit(3, 0)}
        filtered = filter_voice_library(library, allowed_names={"Alice", "Unknown"})
        assert set(filtered.keys()) == {"Alice"}


class TestCliSpeakersFlag:

    def test_cli_speakers_flag_accepted(self, tmp_path):
        """--speakers es un flag válido del CLI."""
        from typer.testing import CliRunner
        from speechlib.__main__ import app

        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"\x00" * 100)
        runner = CliRunner()
        result = runner.invoke(app, [
            str(wav), "--speakers", "Alice,Bob",
        ])
        # No debe fallar por flag desconocido (exit 2 = typer validation)
        # Puede fallar por otro motivo (audio inválido, no token) pero no por --speakers
        assert "No such option" not in (result.output or "")


class TestRecognitionUsesFilteredLibrary:

    def test_recognition_filters_library_when_speakers_provided(self, tmp_path):
        """_run_speaker_recognition_cached filtra la library con allowed_speakers."""
        from pathlib import Path
        from unittest.mock import patch
        import json

        import numpy as np
        import torch
        import torchaudio

        from speechlib.audio_state import AudioState
        from speechlib.core_analysis import _run_speaker_recognition_cached

        # Audio real en tmp_path
        wav = tmp_path / "audio.wav"
        n = int(10.0 * 16000)
        torchaudio.save(str(wav), torch.zeros(1, n), 16000, bits_per_sample=16)
        state = AudioState(source_path=wav, working_path=wav, is_wav=True)
        state.artifacts_dir.mkdir(parents=True, exist_ok=True)

        speakers = {"SPEAKER_00": [[0.0, 5.0, "SPEAKER_00"]]}

        # Library con 3 speakers, pero solo "Present" está en allowed
        full_lib = {
            "Present": np.ones(192),
            "Absent1": np.ones(192) * 1.1,
            "Absent2": np.ones(192) * 0.9,
        }

        with (
            patch(
                "speechlib.core_analysis.load_avg_voice_embeddings",
                return_value=full_lib,
            ),
            patch(
                "speechlib.core_analysis._compute_averaged_embeddings_per_tag",
                return_value={"SPEAKER_00": np.ones(192)},
            ),
        ):
            result = _run_speaker_recognition_cached(
                state, str(tmp_path / "voices"), speakers, ["SPEAKER_00"],
                allowed_speakers=["Present"],
            )

        # Solo "Present" o SPEAKER_00 en valores, nunca Absent1/Absent2
        for v in result.values():
            assert v not in ("Absent1", "Absent2")


class TestAssignExtraSpeakers:

    def test_unmatched_tags_get_extra_names(self):
        """Tags no matcheados reciben nombres de speakers sin sample."""
        from speechlib.domain.recognition import assign_extra_speakers

        speaker_map = {
            "SPEAKER_00": "Manuel Olguin",
            "SPEAKER_01": "SPEAKER_01",  # no matcheó
            "SPEAKER_02": "Agustin Villena",
            "SPEAKER_03": "SPEAKER_03",  # no matcheó
        }
        extra_names = ["Paula Lapostol", "Ximena Vial"]
        # SPEAKER_01 tiene más segmentos → recibe el primer nombre extra
        segment_counts = {"SPEAKER_01": 500, "SPEAKER_03": 200}

        result = assign_extra_speakers(speaker_map, extra_names, segment_counts)

        assert result["SPEAKER_00"] == "Manuel Olguin"  # sin cambio
        assert result["SPEAKER_02"] == "Agustin Villena"  # sin cambio
        assert result["SPEAKER_01"] == "Paula Lapostol"  # más segmentos → primer nombre
        assert result["SPEAKER_03"] == "Ximena Vial"

    def test_more_extras_than_unmatched_ignores_surplus(self):
        from speechlib.domain.recognition import assign_extra_speakers

        speaker_map = {"SPEAKER_00": "SPEAKER_00"}
        extra_names = ["Paula", "Ximena", "Extra"]
        segment_counts = {"SPEAKER_00": 100}

        result = assign_extra_speakers(speaker_map, extra_names, segment_counts)
        assert result["SPEAKER_00"] == "Paula"

    def test_fewer_extras_than_unmatched_leaves_remaining_as_tag(self):
        from speechlib.domain.recognition import assign_extra_speakers

        speaker_map = {
            "SPEAKER_00": "SPEAKER_00",
            "SPEAKER_01": "SPEAKER_01",
            "SPEAKER_02": "SPEAKER_02",
        }
        extra_names = ["Paula"]
        segment_counts = {"SPEAKER_00": 300, "SPEAKER_01": 200, "SPEAKER_02": 100}

        result = assign_extra_speakers(speaker_map, extra_names, segment_counts)
        assert result["SPEAKER_00"] == "Paula"
        assert result["SPEAKER_01"] == "SPEAKER_01"  # sin nombre extra
        assert result["SPEAKER_02"] == "SPEAKER_02"

    def test_no_extras_returns_unchanged(self):
        from speechlib.domain.recognition import assign_extra_speakers

        speaker_map = {"SPEAKER_00": "SPEAKER_00"}
        result = assign_extra_speakers(speaker_map, [], {})
        assert result == speaker_map


class TestFilteredLibraryPreventsFalsePositive:

    def test_absent_speaker_no_longer_wins(self):
        """Sin filtro, 'Absent' gana (cos=0.95). Con filtro, 'Present' gana."""
        dim = 8
        tag_embedding = _unit(dim, 0)

        # Absent tiene embedding casi idéntico al tag (cos=0.95)
        absent_emb = np.zeros(dim)
        absent_emb[0] = 0.95
        absent_emb[1] = 0.31  # norm ~ 1.0

        # Present tiene embedding menos similar (cos=0.7)
        present_emb = np.zeros(dim)
        present_emb[0] = 0.7
        present_emb[2] = 0.71

        full_library = {"Absent": absent_emb, "Present": present_emb}

        # Sin filtro: Absent gana (false positive)
        transcript = Transcript(
            segments=(
                TranscriptSegment(
                    start_ms=0, end_ms=5000, text="hello",
                    speaker=SpeakerIdentity(diarization_tag="SPEAKER_00"),
                ),
            ),
            audio_path="test.wav",
            language="en",
        )
        result_full = assign_speakers(
            transcript,
            embeddings_by_tag={"SPEAKER_00": tag_embedding},
            voice_library=full_library,
            threshold=0.5,
        )
        assert result_full.segments[0].speaker.recognized_name == "Absent"

        # Con filtro: solo Present disponible → Present gana
        filtered = filter_voice_library(full_library, allowed_names={"Present"})
        result_filtered = assign_speakers(
            transcript,
            embeddings_by_tag={"SPEAKER_00": tag_embedding},
            voice_library=filtered,
            threshold=0.5,
        )
        assert result_filtered.segments[0].speaker.recognized_name == "Present"
