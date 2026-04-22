"""
Unit tests: AudioState model (Slice 1)
"""
import pytest
from pathlib import Path
from speechlib.audio_state import AudioState


def test_audio_state_defaults():
    state = AudioState(source_path=Path("a.wav"), working_path=Path("a.wav"))
    assert state.source_path == Path("a.wav")
    assert state.working_path == Path("a.wav")
    assert state.is_wav is False
    assert state.is_mono is False
    assert state.is_16bit is False


def test_audio_state_source_is_immutable():
    state = AudioState(source_path=Path("a.wav"), working_path=Path("a.wav"))
    with pytest.raises(Exception):
        state.source_path = Path("b.wav")


def test_audio_state_model_copy_updates_working_path():
    state = AudioState(source_path=Path("a.wav"), working_path=Path("a.wav"))
    updated = state.model_copy(update={"working_path": Path("b.wav"), "is_wav": True})
    assert updated.source_path == Path("a.wav")   # source sin cambios
    assert updated.working_path == Path("b.wav")
    assert updated.is_wav is True


def test_audio_state_accepts_flags():
    state = AudioState(
        source_path=Path("a.wav"),
        working_path=Path("b.wav"),
        is_wav=True,
        is_mono=True,
        is_16bit=True,
    )
    assert state.is_wav and state.is_mono and state.is_16bit


def test_artifacts_dir_strips_trailing_spaces():
    """Filenames como 'Voz .m4a' (trailing space antes de ext) producian
    un artifacts_dir con trailing space que Windows no puede escribir."""
    state = AudioState(
        source_path=Path("/rec/Voz .m4a"),
        working_path=Path("/rec/Voz .m4a"),
    )
    assert state.artifacts_dir == Path("/rec/.Voz")
    assert not str(state.artifacts_dir).endswith(" ")


def test_artifacts_dir_normal_filename():
    state = AudioState(
        source_path=Path("/rec/meeting.m4a"),
        working_path=Path("/rec/meeting.m4a"),
    )
    assert state.artifacts_dir == Path("/rec/.meeting")


def test_artifacts_dir_cae_al_cache_sin_limpio_si_direct_no_existe(tmp_path):
    """Stem termina en `_limpio` y `.<stem>/` no existe pero `.<stem_sin_limpio>/` si:
    artifacts_dir resuelve al cache sin sufijo (caso: original borrado post-pipeline)."""
    (tmp_path / ".reunion").mkdir()
    state = AudioState(
        source_path=tmp_path / "reunion_limpio.m4a",
        working_path=tmp_path / "reunion_limpio.m4a",
    )
    assert state.artifacts_dir == tmp_path / ".reunion"


def test_artifacts_dir_usa_direct_si_existe_aunque_haya_fallback(tmp_path):
    """Si ambos caches existen, gana el directo (fresh run sobre `_limpio.m4a` standalone
    es legitimo y no debe ser hijacked por un cache sin sufijo preexistente)."""
    (tmp_path / ".reunion_limpio").mkdir()
    (tmp_path / ".reunion").mkdir()
    state = AudioState(
        source_path=tmp_path / "reunion_limpio.m4a",
        working_path=tmp_path / "reunion_limpio.m4a",
    )
    assert state.artifacts_dir == tmp_path / ".reunion_limpio"


def test_artifacts_dir_sin_limpio_no_aplica_fallback(tmp_path):
    """Stem sin sufijo `_limpio`: comportamiento legacy (default al `.<stem>/`)."""
    state = AudioState(
        source_path=tmp_path / "meeting.m4a",
        working_path=tmp_path / "meeting.m4a",
    )
    assert state.artifacts_dir == tmp_path / ".meeting"
