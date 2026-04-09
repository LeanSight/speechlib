"""
Smell 1: apply_speaker_map_to_segments es funcion pura que reemplaza
_merge_same_speakers de core_analysis (que mutaba in-place tres
estructuras paralelas).

Hallazgo durante el analisis: el _merge_same_speakers legacy mutaba
common, speakers Y speaker_map. PERO speakers se regenera 2 lineas
despues via _regroup_speakers_from_common, y el speaker_map se usa
solo via .get(name, name) que nunca toca las claves borradas. Las
mutaciones de speakers y speaker_map eran DEAD CODE.

La unica observacion que importa: rewrite de common[i][2] de
SPEAKER_XX a su nombre mapeado (cuando existe). Eso es exactamente
lo que hace apply_speaker_map_to_segments.

Tests puros: solo listas + dicts, sin I/O.
"""

import pytest


def test_apply_speaker_map_rewrites_segment_labels():
    from speechlib.services.transcript_builder import apply_speaker_map_to_segments

    common = [
        [0.0, 1.0, "SPEAKER_00"],
        [1.0, 2.0, "SPEAKER_01"],
        [2.0, 3.0, "SPEAKER_00"],
    ]
    speaker_map = {
        "SPEAKER_00": "Manuel",
        "SPEAKER_01": "Pamela",
    }

    result = apply_speaker_map_to_segments(common, speaker_map)

    assert result == [
        [0.0, 1.0, "Manuel"],
        [1.0, 2.0, "Pamela"],
        [2.0, 3.0, "Manuel"],
    ]


def test_apply_speaker_map_collapses_duplicate_names():
    """Si dos SPEAKER_XX mapean al mismo nombre, todos los segmentos quedan
    con ese nombre. Es lo que el legacy _merge_same_speakers hacia (parte util)."""
    from speechlib.services.transcript_builder import apply_speaker_map_to_segments

    common = [
        [0.0, 1.0, "SPEAKER_00"],
        [1.0, 2.0, "SPEAKER_03"],
        [2.0, 3.0, "SPEAKER_00"],
    ]
    speaker_map = {
        "SPEAKER_00": "Manuel",
        "SPEAKER_03": "Manuel",  # mismo nombre que SPEAKER_00 — duplicado
    }

    result = apply_speaker_map_to_segments(common, speaker_map)

    assert result == [
        [0.0, 1.0, "Manuel"],
        [1.0, 2.0, "Manuel"],
        [2.0, 3.0, "Manuel"],
    ]


def test_apply_speaker_map_preserves_unmapped_labels():
    """Si un label no esta en speaker_map, se conserva tal cual.
    Defensa: legacy puede tener tags no presentes en el map."""
    from speechlib.services.transcript_builder import apply_speaker_map_to_segments

    common = [
        [0.0, 1.0, "SPEAKER_99"],  # no en map
        [1.0, 2.0, "Manuel"],       # ya es nombre, no en map
    ]
    speaker_map = {"SPEAKER_00": "Pamela"}

    result = apply_speaker_map_to_segments(common, speaker_map)

    assert result == [
        [0.0, 1.0, "SPEAKER_99"],
        [1.0, 2.0, "Manuel"],
    ]


def test_apply_speaker_map_does_not_mutate_input():
    """Funcion pura: ni common ni speaker_map se modifican."""
    from speechlib.services.transcript_builder import apply_speaker_map_to_segments

    common = [
        [0.0, 1.0, "SPEAKER_00"],
        [1.0, 2.0, "SPEAKER_01"],
    ]
    common_snapshot = [list(s) for s in common]
    speaker_map = {"SPEAKER_00": "Manuel", "SPEAKER_01": "Pamela"}
    map_snapshot = dict(speaker_map)

    apply_speaker_map_to_segments(common, speaker_map)

    assert common == common_snapshot
    assert speaker_map == map_snapshot


def test_apply_speaker_map_handles_empty_inputs():
    from speechlib.services.transcript_builder import apply_speaker_map_to_segments

    assert apply_speaker_map_to_segments([], {}) == []
    assert apply_speaker_map_to_segments([], {"SPEAKER_00": "X"}) == []
    assert apply_speaker_map_to_segments([[0, 1, "X"]], {}) == [[0, 1, "X"]]


def test_apply_speaker_map_returns_new_list():
    """El resultado es una lista NUEVA, no la misma referencia."""
    from speechlib.services.transcript_builder import apply_speaker_map_to_segments

    common = [[0.0, 1.0, "SPEAKER_00"]]
    speaker_map = {"SPEAKER_00": "Manuel"}

    result = apply_speaker_map_to_segments(common, speaker_map)
    assert result is not common
