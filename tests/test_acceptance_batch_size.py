"""AT: transcribe_full_aligned expone TRANSCRIPTION_BATCH_SIZE como constante configurable."""
from speechlib.transcribe import TRANSCRIPTION_BATCH_SIZE


def test_batch_size_is_exposed_constant():
    """batch_size es una constante del módulo, no un magic number inline."""
    assert isinstance(TRANSCRIPTION_BATCH_SIZE, int)
    assert TRANSCRIPTION_BATCH_SIZE > 0
