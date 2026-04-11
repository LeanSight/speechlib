"""
Slice 2 unit tests: assign_speakers como funcion pura.

Estilo GOOS-sin-mocks: solo numpy + value objects de dominio. Cero I/O.
Verificacion de salida y estado, no de colaboracion.
"""

import numpy as np
import pytest


def _unit(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float64)
    return v / np.linalg.norm(v)


def _make_transcript(*tags: str):
    """Helper: crea un Transcript con un segmento por tag."""
    from speechlib.domain.transcript import (
        SpeakerIdentity,
        Transcript,
        TranscriptSegment,
    )

    segments = tuple(
        TranscriptSegment(
            start_ms=i * 1000,
            end_ms=(i + 1) * 1000,
            text=f"texto-{i}",
            speaker=SpeakerIdentity(diarization_tag=tag),
        )
        for i, tag in enumerate(tags)
    )
    return Transcript(segments=segments, audio_path="x.wav", language="es")


# ── Casos basicos ────────────────────────────────────────────────────────────


class TestAssignSpeakersBasic:
    def test_perfect_match_assigns_name(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00")
        embeddings = {"SPEAKER_00": _unit(1.0, 0.0)}
        library = {"Manuel": _unit(1.0, 0.0)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        spk = result.segments[0].speaker
        assert spk.recognized_name == "Manuel"
        assert spk.similarity == pytest.approx(1.0, abs=1e-9)
        assert spk.diarization_tag == "SPEAKER_00"

    def test_picks_best_match_among_multiple(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00")
        embeddings = {"SPEAKER_00": _unit(1.0, 0.1, 0.0)}
        library = {
            "Manuel": _unit(1.0, 0.0, 0.0),   # ~0.995
            "Pamela": _unit(0.0, 1.0, 0.0),   # ~0.0995
            "Agustin": _unit(0.0, 0.0, 1.0),  # 0.0
        }

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        assert result.segments[0].speaker.recognized_name == "Manuel"


# ── Casos limite (anti-bug) ──────────────────────────────────────────────────


class TestAssignSpeakersFallback:
    def test_below_threshold_keeps_diarization_tag(self):
        """Invariante critico: similarity bajo threshold → recognized_name=None,
        label=tag. Jamas el literal 'unknown'."""
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_07")
        embeddings = {"SPEAKER_07": _unit(1.0, 0.0)}
        library = {"Pamela": _unit(0.0, 1.0)}  # ortogonal → similarity = 0

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        spk = result.segments[0].speaker
        assert spk.recognized_name is None
        assert spk.label == "SPEAKER_07"
        assert spk.label != "unknown"

    def test_empty_library_keeps_all_tags(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00", "SPEAKER_01")
        embeddings = {
            "SPEAKER_00": _unit(1.0, 0.0),
            "SPEAKER_01": _unit(0.0, 1.0),
        }
        library: dict[str, np.ndarray] = {}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        for seg in result.segments:
            assert seg.speaker.recognized_name is None
            assert seg.speaker.label == seg.speaker.diarization_tag

    def test_missing_embedding_for_tag_leaves_segment_untouched(self):
        """Si no hay embedding para un tag, el segmento se preserva tal cual.
        Permite re-evaluar parcialmente sin perder identidades previas."""
        from speechlib.domain.recognition import assign_speakers
        from speechlib.domain.transcript import (
            SpeakerIdentity,
            Transcript,
            TranscriptSegment,
        )

        transcript = Transcript(
            segments=(
                TranscriptSegment(
                    0, 1000, "previo",
                    SpeakerIdentity(
                        diarization_tag="SPEAKER_00",
                        recognized_name="Manuel",
                        similarity=0.8,
                    ),
                ),
            ),
            audio_path="x.wav",
            language="es",
        )
        embeddings: dict[str, np.ndarray] = {}  # SPEAKER_00 ausente
        library = {"Manuel": _unit(1.0, 0.0)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        # Sin tocar
        assert result.segments[0].speaker.recognized_name == "Manuel"
        assert result.segments[0].speaker.similarity == 0.8

    def test_re_evaluation_can_clear_previous_identification(self):
        """Si un segmento ya tenia name pero la nueva evaluacion no supera
        threshold, debe limpiarse a None (caso de --all-speakers fix)."""
        from speechlib.domain.recognition import assign_speakers
        from speechlib.domain.transcript import (
            SpeakerIdentity,
            Transcript,
            TranscriptSegment,
        )

        transcript = Transcript(
            segments=(
                TranscriptSegment(
                    0, 1000, "x",
                    SpeakerIdentity(
                        diarization_tag="SPEAKER_03",
                        recognized_name="WrongPerson",
                        similarity=0.42,
                    ),
                ),
            ),
            audio_path="x.wav",
            language="es",
        )
        embeddings = {"SPEAKER_03": _unit(1.0, 0.0)}
        library = {"WrongPerson": _unit(0.0, 1.0)}  # ortogonal

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        spk = result.segments[0].speaker
        assert spk.recognized_name is None
        assert spk.label == "SPEAKER_03"  # invariante anti-bug


# ── Coherencia por tag (varios segmentos del mismo speaker) ──────────────────


class TestAssignSpeakersByTag:
    def test_all_segments_with_same_tag_get_same_identity(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript(
            "SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_00"
        )
        embeddings = {
            "SPEAKER_00": _unit(1.0, 0.0),
            "SPEAKER_01": _unit(0.0, 1.0),
        }
        library = {"Manuel": _unit(1.0, 0.0), "Pamela": _unit(0.0, 1.0)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        names = [s.speaker.recognized_name for s in result.segments]
        assert names == ["Manuel", "Pamela", "Manuel", "Manuel"]

    def test_threshold_exact_boundary_is_match(self):
        """similarity == threshold debe contar como match (>=, no >)."""
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00")
        # Vectores con similarity exactamente 0.5
        embeddings = {"SPEAKER_00": _unit(1.0, 0.0)}
        # cos(60°) = 0.5
        library = {"X": _unit(0.5, np.sqrt(3) / 2)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.5)

        assert result.segments[0].speaker.recognized_name == "X"


# ── Pureza ───────────────────────────────────────────────────────────────────


class TestAssignSpeakersPurity:
    def test_input_transcript_not_mutated(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00")
        embeddings = {"SPEAKER_00": _unit(1.0, 0.0)}
        library = {"X": _unit(1.0, 0.0)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        assert transcript.segments[0].speaker.recognized_name is None
        assert result.segments[0].speaker.recognized_name == "X"
        assert result is not transcript

    def test_metadata_preserved(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00")
        embeddings = {"SPEAKER_00": _unit(1.0, 0.0)}
        library = {"X": _unit(1.0, 0.0)}

        result = assign_speakers(transcript, embeddings, library, threshold=0.40)

        assert result.audio_path == transcript.audio_path
        assert result.language == transcript.language
        assert result.segments[0].start_ms == transcript.segments[0].start_ms
        assert result.segments[0].end_ms == transcript.segments[0].end_ms
        assert result.segments[0].text == transcript.segments[0].text


class TestSelectSegmentsForEmbedding:
    """Pure-domain selection of segments for averaged embedding computation.

    Bug context (Pamela Falconi regression): el flujo legacy iteraba en
    orden de documento y se detenia al sumar limit_s. Resultado empirico en
    Alicanto SPEAKER_00: los primeros 25 turnos sumaban 62s y producian un
    embedding contaminado (similarity 0.39 vs library Pamela), mientras que
    los TOP-5 turnos mas largos producian similarity 0.69. La seleccion debe
    ser por duracion descendente, no por orden de documento.
    """

    def test_returns_longest_first_until_limit_reached(self):
        from speechlib.domain.recognition import select_segments_for_embedding

        # Mezcla intencional: cortos primero, largos despues
        segments = [
            [0.0, 0.6, "S"],   # 0.6s
            [1.0, 1.8, "S"],   # 0.8s
            [2.0, 12.0, "S"],  # 10.0s  ← largo
            [13.0, 13.7, "S"], # 0.7s
            [14.0, 22.0, "S"], # 8.0s   ← largo
            [23.0, 23.6, "S"], # 0.6s
        ]

        selected = select_segments_for_embedding(
            segments, limit_s=15.0, min_segment_s=0.5
        )

        # Top-2 mas largos: 10s + 8s = 18s. Debe parar al SUPERAR 15s,
        # devolviendo 10s + 8s (porque tras 10s acumulados, aun no se llega
        # a 15s, asi que entra el 8s tambien).
        assert [s[1] - s[0] for s in selected] == pytest.approx([10.0, 8.0])

    def test_filters_segments_below_min_duration(self):
        from speechlib.domain.recognition import select_segments_for_embedding

        segments = [
            [0.0, 0.3, "S"],   # 0.3s ← descartado
            [1.0, 0.4 + 1.0, "S"],  # 0.4s ← descartado
            [2.0, 4.5, "S"],   # 2.5s
        ]

        selected = select_segments_for_embedding(
            segments, limit_s=60.0, min_segment_s=0.5
        )

        assert len(selected) == 1
        assert selected[0][1] - selected[0][0] == pytest.approx(2.5)

    def test_empty_input_returns_empty_list(self):
        from speechlib.domain.recognition import select_segments_for_embedding

        assert select_segments_for_embedding(
            [], limit_s=60.0, min_segment_s=0.5
        ) == []

    def test_priority_is_descending_duration_not_document_order(self):
        """El bug exacto: en orden de documento los primeros eran cortos
        contaminados; el fix prioriza por duracion descendente."""
        from speechlib.domain.recognition import select_segments_for_embedding

        # 10 segmentos cortos (0.6s) + 1 segmento largo (8s)
        short_segs = [[i, i + 0.6, "S"] for i in range(10)]
        long_seg = [100.0, 108.0, "S"]
        segments = short_segs + [long_seg]

        selected = select_segments_for_embedding(
            segments, limit_s=5.0, min_segment_s=0.5
        )

        # El primero seleccionado debe ser el largo, no los cortos del inicio
        assert selected[0] is long_seg


# ── Invariantes anti-bug (migrados de tests fragiles con mocks) ─────────────


class TestUnknownSpeakerLabelInvariant:
    """Speakers no reconocidos conservan SPEAKER_XX, nunca "unknown".

    Migrado de test_acceptance_unknown_speaker_labels.py (13 patches)
    a test puro de assign_speakers (0 mocks).
    """

    def test_two_unknown_speakers_keep_speaker_xx_tags(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00", "SPEAKER_01")
        embeddings = {
            "SPEAKER_00": _unit(1.0, 0.0, 0.0),
            "SPEAKER_01": _unit(0.0, 1.0, 0.0),
        }
        library = {"Alguien": _unit(0.0, 0.0, 1.0)}  # ortogonal a ambos

        result = assign_speakers(transcript, embeddings, library, threshold=0.55)

        labels = {s.speaker.label for s in result.segments}
        assert "unknown" not in labels
        assert "SPEAKER_00" in labels
        assert "SPEAKER_01" in labels

    def test_known_and_unknown_coexist(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00", "SPEAKER_01")
        embeddings = {
            "SPEAKER_00": _unit(1.0, 0.0, 0.0),
            "SPEAKER_01": _unit(0.0, 1.0, 0.0),
        }
        library = {"Agustin": _unit(1.0, 0.0, 0.0)}  # matchea SPEAKER_00

        result = assign_speakers(transcript, embeddings, library, threshold=0.55)

        labels = {s.speaker.label for s in result.segments}
        assert "Agustin" in labels
        assert "SPEAKER_01" in labels
        assert "unknown" not in labels

    def test_single_unknown_keeps_tag(self):
        from speechlib.domain.recognition import assign_speakers

        transcript = _make_transcript("SPEAKER_00")
        embeddings = {"SPEAKER_00": _unit(1.0, 0.0, 0.0)}
        library = {"Nadie": _unit(0.0, 1.0, 0.0)}  # ortogonal

        result = assign_speakers(transcript, embeddings, library, threshold=0.55)

        assert result.segments[0].speaker.label == "SPEAKER_00"
        assert "unknown" not in result.segments[0].speaker.label


class TestMinSegmentDurationConstant:
    """Constante MIN_SEGMENT_DURATION_S protege contra crash de pyannote/embedding."""

    def test_value_is_half_second(self):
        from speechlib.speaker_recognition import MIN_SEGMENT_DURATION_S
        assert MIN_SEGMENT_DURATION_S == 0.5


class TestBuildSuggestions:
    """build_suggestions: top-N candidatos + recommended, sin decidir el map.

    Funcion pura del dominio que convierte embeddings + voice library en
    estructura human-reviewable para el flujo suggest+confirm.
    """

    def test_clear_match_ranks_top_candidate_first_and_recommends_it(self):
        from speechlib.domain.recognition import build_suggestions

        embeddings = {"SPEAKER_00": _unit(1.0, 0.0, 0.0)}
        library = {
            "ana": _unit(1.0, 0.0, 0.0),
            "bruno": _unit(0.2, 1.0, 0.0),
            "carla": _unit(0.0, 0.0, 1.0),
        }

        result = build_suggestions(embeddings, library, threshold=0.5, min_margin=0.1)

        tags = result["tags"]
        assert set(tags.keys()) == {"SPEAKER_00"}
        top = tags["SPEAKER_00"]["top_candidates"]
        assert len(top) == 3
        assert top[0]["name"] == "ana"
        assert top[0]["score"] == pytest.approx(1.0, abs=1e-3)
        assert top[1]["name"] == "bruno"
        assert top[2]["name"] == "carla"
        assert top[0]["score"] > top[1]["score"] > top[2]["score"]
        assert tags["SPEAKER_00"]["recommended"] == "ana"

    def test_ambiguous_match_recommends_none(self):
        from speechlib.domain.recognition import build_suggestions

        # embedding a 45 grados entre ana y bruno → ambiguo
        embeddings = {"SPEAKER_00": _unit(1.0, 1.0, 0.0)}
        library = {
            "ana": _unit(1.0, 0.0, 0.0),
            "bruno": _unit(0.0, 1.0, 0.0),
            "carla": _unit(0.0, 0.0, 1.0),
        }

        result = build_suggestions(embeddings, library, threshold=0.5, min_margin=0.1)

        suggestion = result["tags"]["SPEAKER_00"]
        # top1 y top2 estan a margin 0 → recommended debe ser None
        assert suggestion["recommended"] is None
        # Pero los top_candidates siguen presentes con sus scores
        assert len(suggestion["top_candidates"]) == 3

    def test_below_threshold_recommends_none_but_keeps_candidates(self):
        from speechlib.domain.recognition import build_suggestions

        # embedding ortogonal a todos → ningun score pasa threshold
        embeddings = {"SPEAKER_00": _unit(0.0, 0.0, 0.0, 1.0)}
        library = {
            "ana": _unit(1.0, 0.0, 0.0, 0.0),
            "bruno": _unit(0.0, 1.0, 0.0, 0.0),
        }

        result = build_suggestions(embeddings, library, threshold=0.5, min_margin=0.1)

        suggestion = result["tags"]["SPEAKER_00"]
        assert suggestion["recommended"] is None
        assert len(suggestion["top_candidates"]) == 2
        # Todos los scores estan muy bajos (cerca de 0)
        for cand in suggestion["top_candidates"]:
            assert cand["score"] < 0.5

    def test_top_n_caps_candidate_list(self):
        from speechlib.domain.recognition import build_suggestions

        embeddings = {"SPEAKER_00": _unit(1.0, 0.0, 0.0, 0.0, 0.0)}
        library = {
            f"speaker_{i}": _unit(*([1.0 if j == i else 0.1 for j in range(5)]))
            for i in range(5)
        }

        result = build_suggestions(embeddings, library, threshold=0.5, min_margin=0.1, top_n=2)
        top = result["tags"]["SPEAKER_00"]["top_candidates"]
        assert len(top) == 2

    def test_result_is_json_serializable(self):
        """El output debe serializarse a JSON directo (sin numpy types)."""
        import json
        from speechlib.domain.recognition import build_suggestions

        embeddings = {"SPEAKER_00": _unit(1.0, 0.0)}
        library = {"ana": _unit(1.0, 0.0), "bruno": _unit(0.0, 1.0)}

        result = build_suggestions(embeddings, library, threshold=0.5, min_margin=0.1)
        # No debe explotar
        serialized = json.dumps(result)
        assert "ana" in serialized
        assert "top_candidates" in serialized
