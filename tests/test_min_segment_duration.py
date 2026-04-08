"""
Slice 12: filtro de turnos pyannote demasiado cortos en speaker_recognition.

Los turnos < 0.5s rompen pyannote/embedding con 'max_pool1d() Invalid
computed output size: 0' o 'Calculated padded input size per channel: (2).
Kernel size: (5)'. Estos errores se capturaban silenciosamente y generaban
ruido en logs. Filtrarlos antes de cortar el audio elimina el ruido y
ahorra I/O.
"""

import numpy as np
from unittest.mock import MagicMock, patch


def test_segments_below_min_duration_are_filtered():
    """speaker_recognition no debe llamar inference para segments < MIN_SEGMENT_DURATION_S."""
    from speechlib.speaker_recognition import (
        MIN_SEGMENT_DURATION_S,
        speaker_recognition,
    )

    # MIN_SEGMENT_DURATION_S debe estar definido
    assert MIN_SEGMENT_DURATION_S > 0
    assert MIN_SEGMENT_DURATION_S <= 1.0  # razonable

    inference_calls = []

    def fake_inference(path):
        inference_calls.append(path)
        return np.ones(192)

    with (
        patch("speechlib.speaker_recognition._get_inference", return_value=fake_inference),
        patch("speechlib.speaker_recognition.load_avg_voice_embeddings", return_value={"X": np.ones(192)}),
        patch("speechlib.speaker_recognition.slice_and_save"),
        patch("os.path.exists", return_value=True),
        patch("os.remove"),
    ):
        # 3 segmentos cortos (< 0.5s) + 1 segmento valido
        segments = [
            [0.0, 0.2, "SPEAKER_00"],   # 0.2s - filtrado
            [1.0, 1.3, "SPEAKER_00"],   # 0.3s - filtrado
            [2.0, 5.0, "SPEAKER_00"],   # 3.0s - OK
            [6.0, 6.4, "SPEAKER_00"],   # 0.4s - filtrado
        ]
        speaker_recognition(
            file_name="x.wav",
            voices_folder="voices",
            segments=segments,
        )

    # Solo el segmento valido debe haber llamado inference
    assert len(inference_calls) == 1


def test_min_segment_duration_constant_value():
    from speechlib.speaker_recognition import MIN_SEGMENT_DURATION_S
    assert MIN_SEGMENT_DURATION_S == 0.5
