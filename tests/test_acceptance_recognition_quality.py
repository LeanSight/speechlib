"""
Slice 9 AT: mejoras de calidad del reconocimiento.

Cambios:
1. Threshold default subido de 0.40 a 0.45 (mas conservador, menos false
   positives observados en grabaciones reales).
2. Min margin top1 vs top2: si el ganador no es CLARAMENTE mejor que el
   segundo lugar, se rechaza como ambiguo y queda como SPEAKER_XX.

Caso del bug observado en CCS Gerentes 25min:
- SPEAKER_07 → BciS - Jolyon Abello con sim=0.414 (apenas pasa 0.40)
  pero el segundo es 0.348 → margen 0.066 (ambiguo, false positive)
- Con threshold=0.45 + margen=0.10 se rechaza correctamente.
"""

import numpy as np
import pytest


def _unit(*c):
    v = np.array(c, dtype=np.float64)
    return v / np.linalg.norm(v)


def _make_transcript(*tags):
    from speechlib.domain.transcript import (
        SpeakerIdentity,
        Transcript,
        TranscriptSegment,
    )

    return Transcript(
        segments=tuple(
            TranscriptSegment(
                start_ms=i * 1000,
                end_ms=(i + 1) * 1000,
                text="x",
                speaker=SpeakerIdentity(diarization_tag=tag),
            )
            for i, tag in enumerate(tags)
        ),
        audio_path="x.wav",
        language="es",
    )


def test_default_threshold_is_055():
    from speechlib.speaker_recognition import SPEAKER_SIMILARITY_THRESHOLD
    assert SPEAKER_SIMILARITY_THRESHOLD == 0.55


def test_assign_speakers_rejects_borderline_match_with_min_margin():
    """SPEAKER_07 case: top1=0.414, top2=0.348 → margen 0.066 < 0.10
    debe rechazarse aunque pase threshold (0.40)."""
    from speechlib.domain.recognition import assign_speakers

    transcript = _make_transcript("SPEAKER_07")
    # SPEAKER_07 embedding equidistante de Jolyon y Cristian
    # cos con Jolyon = ~0.414, con Cristian = ~0.348 (margen ~0.066)
    embeddings = {
        "SPEAKER_07": _unit(0.414, 0.348, 0.5),
    }
    library = {
        "Jolyon Abello": _unit(1.0, 0.0, 0.0),
        "Cristian Ruiz": _unit(0.0, 1.0, 0.0),
    }

    result = assign_speakers(
        transcript, embeddings, library,
        threshold=0.40,
        min_margin=0.10,
    )

    spk = result.segments[0].speaker
    assert spk.recognized_name is None
    assert spk.label == "SPEAKER_07"


def test_assign_speakers_accepts_clear_match_above_min_margin():
    """SPEAKER_06 case: top1=0.709, top2=0.435 → margen 0.274 > 0.10
    debe aceptarse."""
    from speechlib.domain.recognition import assign_speakers

    transcript = _make_transcript("SPEAKER_06")
    embeddings = {"SPEAKER_06": _unit(1.0, 0.05, 0.0)}
    library = {
        "Agustin": _unit(1.0, 0.0, 0.0),    # ~1.0
        "Cristian": _unit(0.0, 1.0, 0.0),   # ~0.05
    }

    result = assign_speakers(
        transcript, embeddings, library,
        threshold=0.40,
        min_margin=0.10,
    )

    spk = result.segments[0].speaker
    assert spk.recognized_name == "Agustin"


def test_assign_speakers_min_margin_default_is_lenient():
    """Por defecto min_margin=0.0 mantiene comportamiento backward-compatible."""
    from speechlib.domain.recognition import assign_speakers

    transcript = _make_transcript("SPEAKER_00")
    embeddings = {"SPEAKER_00": _unit(0.6, 0.5, 0.0)}
    library = {
        "A": _unit(1.0, 0.0, 0.0),
        "B": _unit(0.0, 1.0, 0.0),
    }

    # Sin min_margin (default 0.0), pasa el threshold y se identifica
    result = assign_speakers(transcript, embeddings, library, threshold=0.40)
    assert result.segments[0].speaker.recognized_name == "A"
