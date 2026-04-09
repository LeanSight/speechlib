"""
Slice 18 AT: samples para extraccion deben venir de los turnos RAW del RTTM,
no de los segmentos del transcript readable.

Brecha detectada en Alicanto: los segmentos del transcript.json son resultado
de group_by_sentences (post-merge), pueden ser de hasta 55s. En meetings
multi-speaker hay crosstalk inevitable en bloques tan largos. Pyannote
etiqueta al speaker dominante pero las interjecciones de otros contaminan
los samples.

Solucion: build_transcript_from_annotation_turns() construye un Transcript
"sintetico" donde cada segmento es un turno raw del RTTM (cada turno es
single-speaker por construccion de pyannote diarization). Esa version
del Transcript se usa SOLO para sample extraction; el transcript.json
readable mantiene los segmentos post-merge para legibilidad.

Tests puros: tuples + value objects, sin pyannote, sin audio.
"""


def test_helper_creates_one_segment_per_raw_turn():
    """Cada turno del RTTM se convierte en exactamente un TranscriptSegment.
    Sin merging, sin grouping — single-speaker garantizado por pyannote."""
    from speechlib.services.transcript_builder import (
        build_transcript_from_annotation_turns,
    )

    # 4 turnos pyannote: SPEAKER_00 corto, SPEAKER_01 largo, SPEAKER_00 corto otra vez, SPEAKER_02
    annotation_turns = [
        (0.0,    1.5,  "SPEAKER_00"),  # 1500ms
        (2.0,   12.0,  "SPEAKER_01"),  # 10000ms (largo pero single speaker)
        (12.5,  13.7,  "SPEAKER_00"),  # 1200ms
        (14.0,  15.5,  "SPEAKER_02"),  # 1500ms
    ]
    speaker_map = {
        "SPEAKER_00": "SPEAKER_00",       # no identificado
        "SPEAKER_01": "Manuel Olguin",    # identificado
        "SPEAKER_02": "SPEAKER_02",       # no identificado
    }

    transcript = build_transcript_from_annotation_turns(
        annotation_turns=annotation_turns,
        speaker_map=speaker_map,
        audio_path="rec.wav",
        language="es",
    )

    # Un segmento por turno, en orden
    assert len(transcript.segments) == 4

    s0, s1, s2, s3 = transcript.segments

    # Conversion a ms y identidades correctas
    assert s0.start_ms == 0
    assert s0.end_ms == 1500
    assert s0.speaker.diarization_tag == "SPEAKER_00"
    assert s0.speaker.recognized_name is None
    assert s0.speaker.label == "SPEAKER_00"

    assert s1.start_ms == 2000
    assert s1.end_ms == 12000
    assert s1.speaker.diarization_tag == "SPEAKER_01"
    assert s1.speaker.recognized_name == "Manuel Olguin"
    assert s1.speaker.label == "Manuel Olguin"

    assert s2.start_ms == 12500
    assert s2.end_ms == 13700
    assert s2.speaker.diarization_tag == "SPEAKER_00"  # mismo tag, segundo turno

    assert s3.start_ms == 14000
    assert s3.end_ms == 15500
    assert s3.speaker.diarization_tag == "SPEAKER_02"


def test_samples_from_raw_turns_avoid_crosstalk_in_long_segments():
    """Caso del bug Alicanto: la diferencia entre extraer desde transcript
    readable (segmentos post-merge largos) vs RTTM raw (turnos single-speaker).

    Verifica que pasar el Transcript construido por build_transcript_from_
    annotation_turns a plan_speaker_samples produce clips de duraciones que
    matchean los turnos RAW (no los segmentos readable).
    """
    from speechlib.domain.sample_extraction import plan_speaker_samples
    from speechlib.services.transcript_builder import (
        build_transcript_from_annotation_turns,
    )

    # SPEAKER_05 tiene 3 turnos cortos. En el transcript readable estarian
    # fusionados en un solo segmento de 50+ segundos.
    annotation_turns = [
        (0.0,   2.5,  "SPEAKER_05"),
        (3.0,   5.5,  "SPEAKER_05"),
        (6.0,  10.0,  "SPEAKER_05"),
    ]
    speaker_map = {"SPEAKER_05": "SPEAKER_05"}  # no identificado

    transcript = build_transcript_from_annotation_turns(
        annotation_turns=annotation_turns,
        speaker_map=speaker_map,
        audio_path="x.wav",
        language="es",
    )

    plans = plan_speaker_samples(
        transcript,
        max_clips_per_speaker=5,
        min_clip_duration_ms=1500,
        min_unidentified_clip_duration_ms=500,
    )

    assert len(plans) == 1
    plan = plans[0]
    # Los clips son los turnos individuales (todos >= 500ms): no hay un solo
    # clip de "duracion total" — son los 3 turnos por separado.
    assert len(plan.clips) == 3
    durations = sorted((c.end_ms - c.start_ms for c in plan.clips), reverse=True)
    assert durations == [4000, 2500, 2500]
    # Ningun clip es de "todo el rango fusionado" (10000ms o mas)
    assert max(durations) == 4000
