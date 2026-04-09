from functools import lru_cache

import torch
from pyannote.audio import Pipeline


@lru_cache(maxsize=1)
def get_diarization_pipeline(token: str):
    """Carga y cachea el pipeline de diarizacion pyannote/speaker-diarization-community-1.

    Slice A (Apr 2026): upgrade desde 3.1 a community-1 (released Q1 2026 con
    pyannote.audio 4.0). Mejoras documentadas:
    - DER en AliMeeting (meetings, nuestro caso): 24.5% -> 20.3% (-17%)
    - DER en AISHELL-4: 12.2% -> 11.7%
    - Reduce significativamente "speaker confusion" (turnos asignados al
      speaker incorrecto) — exactamente el modo de falla observado en Alicanto
    - Soporta num_speakers, min_speakers, max_speakers parameters
    - API y output RTTM compatibles con 3.1 (drop-in)
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    return pipeline
