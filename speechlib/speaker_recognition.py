"""
Voice library loading + embedding extraction.

Slice 15: las funciones legacy speaker_recognition(), find_best_speaker(),
detect_unknown_speakers() fueron eliminadas. La logica de matching vive
ahora en speechlib.domain.recognition.assign_speakers (funcion pura).

Lo que sobrevive aqui es solo I/O:
- _get_inference() / get_embedding() — wrappers de pyannote/embedding
- cosine_similarity() — usado por tools/diagnose_speaker y tools/enroll_speaker
- load_voice_embeddings() / load_avg_voice_embeddings() — cargan la library
- Constantes: SPEAKER_SIMILARITY_THRESHOLD, MIN_MARGIN, MIN_SEGMENT_DURATION_S
"""

import logging
import os
from pathlib import Path
import numpy as np
from pyannote.audio import Model, Inference
from .domain.recognition import cosine_similarity
import torch

logger = logging.getLogger(__name__)

SPEAKER_SIMILARITY_THRESHOLD = 0.50
SPEAKER_SIMILARITY_MIN_MARGIN = 0.10
# Threshold subido de 0.45 -> 0.50 tras descubrir falso positivo en Alicanto
# (SPEAKER_02 era falsamente identificado como "AA - Cristian Ruiz" sim=0.498).
# Pamela (0.71) y Agustin (0.72) siguen pasando holgadamente. Validado en
# tests/test_acceptance_pamela_alicanto_recognition.py.
MIN_SEGMENT_DURATION_S = 0.5  # turnos pyannote mas cortos rompen pyannote/embedding
VOICES_SKIP_PREFIX = "_"


_embedding_model = None
_inference = None


def _get_inference():
    global _embedding_model, _inference
    if _embedding_model is None:
        _embedding_model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=os.environ.get("HF_TOKEN", None)
        )
        if torch.cuda.is_available():
            _embedding_model.to(torch.device("cuda"))
        _inference = Inference(_embedding_model, window="whole")
    return _inference


def get_embedding(audio_path: str) -> np.ndarray:
    inference = _get_inference()
    embedding = inference(audio_path)
    return embedding


# cosine_similarity importada de domain.recognition (canonica, sin scipy)


def load_voice_embeddings(
    voices_folder: Path, enhanced: bool = False
) -> dict[str, list[np.ndarray]]:
    """Carga embeddings por archivo para cada speaker en voices_folder.

    Retorna {speaker_name: [embedding_por_archivo, ...]}
    Omite directorios con prefijo VOICES_SKIP_PREFIX ('_').

    Si enhanced=True, busca WAVs en _enhanced/ de cada speaker.
    Fallback a raíz si _enhanced/ no existe o está vacío.
    """
    result: dict[str, list[np.ndarray]] = {}
    voices_folder = Path(voices_folder)
    for entry in sorted(voices_folder.iterdir()):
        if not entry.is_dir() or entry.name.startswith(VOICES_SKIP_PREFIX):
            continue
        wav_dir = entry / "_enhanced" if enhanced else entry
        if not wav_dir.is_dir():
            wav_dir = entry
        wavs = sorted(wav_dir.glob("*.wav"))
        if enhanced and not wavs:
            wavs = sorted(entry.glob("*.wav"))
        embs = []
        for wav in wavs:
            try:
                embs.append(get_embedding(str(wav)))
            except Exception as e:
                print(f"Error extracting embedding from {wav}: {e}")
        if embs:
            result[entry.name] = embs
    return result


def load_avg_voice_embeddings(
    voices_folder: Path, enhanced: bool = False
) -> dict[str, np.ndarray]:
    """Carga embedding promedio por speaker en voices_folder.

    Retorna {speaker_name: avg_embedding}.
    """
    raw = load_voice_embeddings(voices_folder, enhanced=enhanced)
    return {name: np.mean(embs, axis=0) for name, embs in raw.items()}


