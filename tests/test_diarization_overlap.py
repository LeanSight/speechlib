"""Test: _build_speaker_groups preserva segmentos solapados de pyannote."""
from pyannote.core import Annotation, Segment

from speechlib.core_analysis import _build_speaker_groups


def _annotation_with_overlap():
    """Annotation con SPEAKER_00 [0-5] y SPEAKER_01 [3-8] solapados."""
    a = Annotation()
    a[Segment(0.0, 5.0)] = "SPEAKER_00"
    a[Segment(3.0, 8.0)] = "SPEAKER_01"
    return a


def test_overlapping_segments_both_appear_in_result():
    """Ambos speakers solapados deben estar en el resultado."""
    common, speakers, _, _ = _build_speaker_groups(_annotation_with_overlap())

    speakers_out = {seg[2] for seg in common}
    assert "SPEAKER_00" in speakers_out
    assert "SPEAKER_01" in speakers_out


def test_overlapping_timestamps_preserved():
    """Los timestamps solapados no se modifican (salvo rounding a 0.1s)."""
    common, _, _, _ = _build_speaker_groups(_annotation_with_overlap())

    starts = {seg[0] for seg in common}
    assert 0.0 in starts
    assert 3.0 in starts
