"""
Slice 12: filtro de turnos pyannote demasiado cortos antes de calcular embeddings.

Los turnos < 0.5s rompen pyannote/embedding con 'max_pool1d() Invalid
computed output size: 0' o 'Calculated padded input size per channel: (2).
Kernel size: (5)'. Estos errores se capturaban silenciosamente y generaban
ruido en logs. Filtrarlos antes de cortar el audio elimina el ruido y
ahorra I/O.

Slice 15: speaker_recognition() legacy fue eliminada. El filtro vive ahora
en _compute_averaged_embeddings_per_tag (core_analysis) y en
compute_embeddings_per_label (relabel_vtt). Este test ejercita el primero.
"""

import numpy as np
from unittest.mock import MagicMock, patch


def test_segments_below_min_duration_are_filtered_in_core_analysis():
    """_compute_averaged_embeddings_per_tag no debe llamar inference para
    segmentos < MIN_SEGMENT_DURATION_S."""
    from pathlib import Path
    from speechlib.audio_state import AudioState
    from speechlib.core_analysis import _compute_averaged_embeddings_per_tag
    from speechlib.speaker_recognition import MIN_SEGMENT_DURATION_S

    # MIN_SEGMENT_DURATION_S debe estar definido y razonable
    assert MIN_SEGMENT_DURATION_S > 0
    assert MIN_SEGMENT_DURATION_S <= 1.0

    inference_calls = []

    def fake_inference(path):
        inference_calls.append(path)
        return np.ones(192)

    state = AudioState(
        source_path=Path("/tmp/x.wav"),
        working_path=Path("/tmp/x.wav"),
    )

    speakers = {
        "SPEAKER_00": [
            [0.0, 0.2, "SPEAKER_00"],   # 0.2s - filtrado
            [1.0, 1.3, "SPEAKER_00"],   # 0.3s - filtrado
            [2.0, 5.0, "SPEAKER_00"],   # 3.0s - OK
            [6.0, 6.4, "SPEAKER_00"],   # 0.4s - filtrado
        ],
    }

    with (
        patch("speechlib.core_analysis._get_inference", return_value=fake_inference),
        patch("speechlib.core_analysis.slice_and_save"),
        patch("os.path.exists", return_value=True),
        patch("os.makedirs"),
        patch("os.remove"),
    ):
        result = _compute_averaged_embeddings_per_tag(state, speakers)

    # Solo el segmento valido debe haber llamado inference
    assert len(inference_calls) == 1
    # El embedding promedio para SPEAKER_00 debe existir (1 chunk valido)
    assert "SPEAKER_00" in result


def test_min_segment_duration_constant_value():
    from speechlib.speaker_recognition import MIN_SEGMENT_DURATION_S
    assert MIN_SEGMENT_DURATION_S == 0.5
