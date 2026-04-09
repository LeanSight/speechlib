"""
Slice 5 unit tests: build_transcript_from_legacy_segments (puro).
"""

import pytest


def test_overlap_resolves_correct_tag_when_segment_within_turn():
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    legacy = [[1.0, 1.5, "x", "Manuel"]]
    annotation = [
        (0.0, 5.0, "SPEAKER_00"),  # cubre completamente el segmento
        (5.0, 10.0, "SPEAKER_01"),
    ]
    speaker_map = {"SPEAKER_00": "Manuel", "SPEAKER_01": "Pamela"}

    t = build_transcript_from_legacy_segments(legacy, annotation, speaker_map, "x.wav", "es")
    assert t.segments[0].speaker.diarization_tag == "SPEAKER_00"


def test_overlap_picks_max_overlap_when_segment_spans_two_turns():
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    legacy = [[1.0, 4.0, "x", "X"]]
    annotation = [
        (0.0, 1.5, "SPEAKER_00"),  # overlap = 0.5s
        (1.5, 5.0, "SPEAKER_01"),  # overlap = 2.5s ← gana
    ]
    speaker_map = {"SPEAKER_00": "X", "SPEAKER_01": "X"}

    t = build_transcript_from_legacy_segments(legacy, annotation, speaker_map, "x.wav", "es")
    assert t.segments[0].speaker.diarization_tag == "SPEAKER_01"


def test_segment_outside_any_turn_falls_back_to_label():
    """Si por algun motivo no hay overlap, el diarization_tag = label."""
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    legacy = [[100.0, 105.0, "x", "SPEAKER_99"]]
    annotation = [(0.0, 10.0, "SPEAKER_00")]
    speaker_map = {"SPEAKER_00": "X"}

    t = build_transcript_from_legacy_segments(legacy, annotation, speaker_map, "x.wav", "es")
    assert t.segments[0].speaker.diarization_tag == "SPEAKER_99"
    assert t.segments[0].speaker.label == "SPEAKER_99"


def test_label_unknown_string_normalized_to_diarization_tag():
    """Si por bug del legacy aparece label='unknown', NO debe propagarse al
    Transcript: el dominio nuevo lo normaliza al diarization_tag pyannote."""
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    legacy = [[1.0, 2.0, "x", "unknown"]]
    annotation = [(0.0, 5.0, "SPEAKER_07")]
    speaker_map = {"SPEAKER_07": "unknown"}  # legacy bug

    t = build_transcript_from_legacy_segments(legacy, annotation, speaker_map, "x.wav", "es")
    spk = t.segments[0].speaker
    assert spk.diarization_tag == "SPEAKER_07"
    assert spk.recognized_name is None
    assert spk.label == "SPEAKER_07"
    assert spk.label != "unknown"


def test_int_ms_conversion():
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    legacy = [[1.234, 5.678, "x", "X"]]
    annotation = [(0.0, 10.0, "SPEAKER_00")]
    speaker_map = {"SPEAKER_00": "X"}

    t = build_transcript_from_legacy_segments(legacy, annotation, speaker_map, "x.wav", "es")
    assert t.segments[0].start_ms == 1234
    assert t.segments[0].end_ms == 5678


def test_build_from_raw_turns_empty_input_returns_empty_transcript():
    from speechlib.services.transcript_builder import build_transcript_from_annotation_turns

    t = build_transcript_from_annotation_turns(
        annotation_turns=[],
        speaker_map={},
        audio_path="x.wav",
        language="es",
    )
    assert t.segments == ()
    assert t.audio_path == "x.wav"
    assert t.language == "es"


def test_build_from_raw_turns_text_is_always_empty_string():
    """No se transcribe — los samples no requieren texto."""
    from speechlib.services.transcript_builder import build_transcript_from_annotation_turns

    t = build_transcript_from_annotation_turns(
        annotation_turns=[(0.0, 1.0, "SPEAKER_00")],
        speaker_map={"SPEAKER_00": "Manuel"},
        audio_path="x.wav",
        language="es",
    )
    assert t.segments[0].text == ""


def test_build_from_raw_turns_unmapped_tag_stays_unidentified():
    """Si un tag no aparece en speaker_map, queda como SPEAKER_XX no identificado.
    Defensa: el speaker_map podria estar parcial."""
    from speechlib.services.transcript_builder import build_transcript_from_annotation_turns

    t = build_transcript_from_annotation_turns(
        annotation_turns=[(0.0, 1.0, "SPEAKER_99")],
        speaker_map={"SPEAKER_00": "Manuel"},  # SPEAKER_99 ausente
        audio_path="x.wav",
        language="es",
    )
    spk = t.segments[0].speaker
    assert spk.diarization_tag == "SPEAKER_99"
    assert spk.recognized_name is None
    assert spk.label == "SPEAKER_99"


def test_build_from_raw_turns_unknown_value_normalized():
    """Defensa contra el legacy: si el speaker_map tiene 'unknown' como valor,
    NO debe propagarse al recognized_name. Cae a None y label = tag."""
    from speechlib.services.transcript_builder import build_transcript_from_annotation_turns

    t = build_transcript_from_annotation_turns(
        annotation_turns=[(0.0, 1.0, "SPEAKER_07")],
        speaker_map={"SPEAKER_07": "unknown"},
        audio_path="x.wav",
        language="es",
    )
    spk = t.segments[0].speaker
    assert spk.recognized_name is None
    assert spk.label == "SPEAKER_07"


def test_build_from_raw_turns_speaker_xx_value_normalized():
    """Defensa: si el speaker_map mapea SPEAKER_XX -> SPEAKER_XX (auto-fallback
    del legacy), NO debe quedar como nombre. Cae a None."""
    from speechlib.services.transcript_builder import build_transcript_from_annotation_turns

    t = build_transcript_from_annotation_turns(
        annotation_turns=[(0.0, 1.0, "SPEAKER_03")],
        speaker_map={"SPEAKER_03": "SPEAKER_03"},
        audio_path="x.wav",
        language="es",
    )
    spk = t.segments[0].speaker
    assert spk.recognized_name is None
    assert spk.label == "SPEAKER_03"


def test_build_from_raw_turns_int_ms_conversion():
    """start/end en segundos float -> ms int."""
    from speechlib.services.transcript_builder import build_transcript_from_annotation_turns

    t = build_transcript_from_annotation_turns(
        annotation_turns=[(1.234, 5.678, "SPEAKER_00")],
        speaker_map={"SPEAKER_00": "X"},
        audio_path="x.wav",
        language="es",
    )
    assert t.segments[0].start_ms == 1234
    assert t.segments[0].end_ms == 5678


def test_build_from_raw_turns_preserves_order():
    """Orden de los turnos del input se preserva en los segmentos del output."""
    from speechlib.services.transcript_builder import build_transcript_from_annotation_turns

    t = build_transcript_from_annotation_turns(
        annotation_turns=[
            (10.0, 11.0, "SPEAKER_00"),
            (5.0,  6.0,  "SPEAKER_01"),  # mas temprano pero llega despues en la lista
            (0.0,  1.0,  "SPEAKER_02"),
        ],
        speaker_map={
            "SPEAKER_00": "A",
            "SPEAKER_01": "B",
            "SPEAKER_02": "C",
        },
        audio_path="x.wav",
        language="es",
    )
    starts = [s.start_ms for s in t.segments]
    assert starts == [10000, 5000, 0]  # mismo orden que el input


def test_legacy_label_starting_with_speaker_used_as_tag_directly():
    """Si el legacy label ya es un SPEAKER_XX (no identificado por core_analysis),
    el builder debe usarlo TAL CUAL como diarization_tag, ignorando el overlap.

    Razon: tras absorb_micro_segments + merge_short_turns, los segmentos
    pueden cubrir multiples turnos pyannote y el overlap puede devolver el
    tag mas grande aunque el legacy ya tenga la respuesta correcta. Para
    no identificados, el legacy es la fuente de verdad."""
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments

    legacy = [[10.0, 50.0, "x", "SPEAKER_01"]]
    annotation = [
        (0.0, 100.0, "SPEAKER_00"),  # cubre TODO el rango → max overlap
        (10.0, 12.0, "SPEAKER_01"),  # turno corto del speaker correcto
    ]
    speaker_map = {"SPEAKER_00": "Carlos", "SPEAKER_01": "SPEAKER_01"}

    t = build_transcript_from_legacy_segments(legacy, annotation, speaker_map, "x.wav", "es")
    spk = t.segments[0].speaker
    assert spk.diarization_tag == "SPEAKER_01"  # ← preserva el legacy label
    assert spk.recognized_name is None
    assert spk.label == "SPEAKER_01"


def test_returns_immutable_transcript():
    from speechlib.services.transcript_builder import build_transcript_from_legacy_segments
    from speechlib.domain.transcript import Transcript

    t = build_transcript_from_legacy_segments(
        [[0.0, 1.0, "x", "X"]],
        [(0.0, 1.0, "SPEAKER_00")],
        {"SPEAKER_00": "X"},
        "a.wav", "es",
    )
    assert isinstance(t, Transcript)
    assert isinstance(t.segments, tuple)
