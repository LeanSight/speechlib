"""
Slice 17 AT: dual threshold para identificados vs no identificados.

Brecha detectada en produccion (Workshop 2 part 1): SPEAKER_01 tenia 6
segmentos cortos (300-700ms), todos bajo el filtro min_clip_duration_ms=2000.
Como resultado, NO aparecio en samples/por_nombrar/ — el usuario perdio
visibilidad de que ese speaker existia.

Politica nueva: identificados merecen clips de calidad para verificacion
(>=2s), no identificados merecen visibilidad incluso con clips cortos
(default 500ms).

Tests puros: cero I/O.
"""

import pytest


def _seg(start_ms, end_ms, label, identified=False):
    from speechlib.domain.transcript import SpeakerIdentity, TranscriptSegment

    if identified:
        spk = SpeakerIdentity(
            diarization_tag="SPEAKER_99",
            recognized_name=label,
            similarity=0.8,
        )
    else:
        spk = SpeakerIdentity(diarization_tag=label)
    return TranscriptSegment(start_ms=start_ms, end_ms=end_ms, text="x", speaker=spk)


def _transcript(*segments):
    from speechlib.domain.transcript import Transcript

    return Transcript(
        segments=tuple(segments),
        audio_path="x.wav",
        language="es",
    )


def test_unidentified_uses_lower_threshold_when_provided():
    """Caso del bug del Workshop 2 part1: SPEAKER_01 con clips de 300-700ms
    debe aparecer en plans cuando min_unidentified_clip_duration_ms=500."""
    from speechlib.domain.sample_extraction import plan_speaker_samples

    transcript = _transcript(
        # Manuel: 1 clip largo (3s) → identificado, pasa cualquier threshold
        _seg(0, 3000, "Manuel", identified=True),
        # SPEAKER_01: 4 clips cortos (300-700ms) — TODOS bajo 2000ms
        _seg(5000, 5700, "SPEAKER_01"),
        _seg(8000, 8600, "SPEAKER_01"),
        _seg(10000, 10500, "SPEAKER_01"),
        _seg(12000, 12300, "SPEAKER_01"),
    )

    plans = plan_speaker_samples(
        transcript,
        max_clips_per_speaker=5,
        min_clip_duration_ms=2000,                # estricto para identificados
        min_unidentified_clip_duration_ms=500,    # permisivo para no identificados
    )

    by_label = {p.speaker_label: p for p in plans}

    # Manuel pasa el threshold estricto
    assert "Manuel" in by_label
    assert by_label["Manuel"].is_identified is True

    # SPEAKER_01 ahora aparece (antes era invisible)
    assert "SPEAKER_01" in by_label
    spk = by_label["SPEAKER_01"]
    assert spk.is_identified is False
    # Solo los clips >= 500ms (descartando el de 300ms)
    assert len(spk.clips) == 3
    durations = sorted((c.end_ms - c.start_ms for c in spk.clips), reverse=True)
    assert durations == [700, 600, 500]


def test_unidentified_clips_below_min_unidentified_threshold_still_filtered():
    """El threshold permisivo TAMBIEN aplica un piso minimo. Clips < 500ms
    siguen siendo filtrados."""
    from speechlib.domain.sample_extraction import plan_speaker_samples

    transcript = _transcript(
        _seg(0, 200, "SPEAKER_01"),   # 200ms → filtrado
        _seg(1000, 1300, "SPEAKER_01"), # 300ms → filtrado
        _seg(2000, 2400, "SPEAKER_01"), # 400ms → filtrado
    )

    plans = plan_speaker_samples(
        transcript,
        max_clips_per_speaker=5,
        min_clip_duration_ms=2000,
        min_unidentified_clip_duration_ms=500,
    )

    # Sin clips elegibles → speaker no aparece en plans
    assert plans == ()


def test_identified_speakers_keep_strict_threshold():
    """Threshold permisivo NO contamina identificados: ellos siguen
    requiriendo el min_clip_duration_ms estricto."""
    from speechlib.domain.sample_extraction import plan_speaker_samples

    transcript = _transcript(
        # Clips cortos del identificado (no deben pasar)
        _seg(0, 1000, "Manuel", identified=True),
        _seg(2000, 2800, "Manuel", identified=True),
    )

    plans = plan_speaker_samples(
        transcript,
        max_clips_per_speaker=5,
        min_clip_duration_ms=2000,
        min_unidentified_clip_duration_ms=500,
    )

    # Manuel filtrado completamente — sus clips son < 2000ms
    assert plans == ()


def test_default_min_unidentified_falls_back_to_min_clip_duration():
    """Backward compat: si no se pasa min_unidentified_clip_duration_ms,
    se usa el mismo threshold que identificados."""
    from speechlib.domain.sample_extraction import plan_speaker_samples

    transcript = _transcript(
        _seg(0, 600, "SPEAKER_01"),  # 600ms
    )

    # Sin parametro nuevo: comportamiento backward-compatible (filtrado)
    plans = plan_speaker_samples(
        transcript,
        max_clips_per_speaker=5,
        min_clip_duration_ms=2000,
    )
    assert plans == ()
