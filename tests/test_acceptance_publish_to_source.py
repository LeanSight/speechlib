"""
AT: publish de outputs finales al source folder con naming _limpio.

Verifica que al terminar core_analysis:
- transcript queda como {stem}_limpio.vtt junto al audio original
- audio comprimido queda como {stem}_limpio.m4a junto al audio original
- artifacts_dir mantiene sus originales intactos

Uso:
    HF_TOKEN=hf_... pytest tests/test_acceptance_publish_to_source.py -v -s -m e2e
"""

import os
import shutil
from pathlib import Path

import pytest

HF_TOKEN = os.environ.get("HF_TOKEN", "")
AUDIO = Path(__file__).parent.parent / "examples" / "obama_zach.wav"

pytestmark = pytest.mark.e2e

skip_reason = []
if not HF_TOKEN:
    skip_reason.append("HF_TOKEN no esta seteado")
if not AUDIO.exists():
    skip_reason.append(f"audio no encontrado: {AUDIO}")

needs_env = pytest.mark.skipif(bool(skip_reason), reason=" | ".join(skip_reason) or "ok")


@pytest.fixture(scope="module")
def published_result(tmp_path_factory):
    """Corre core_analysis con compress=True y verifica publish."""
    from speechlib.core_analysis import core_analysis

    log_dir = tmp_path_factory.mktemp("e2e_publish")
    tmp_audio = log_dir / AUDIO.name
    shutil.copy2(AUDIO, tmp_audio)

    core_analysis(
        str(tmp_audio),
        voices_folder=None,
        log_folder=str(log_dir),
        language="en",
        modelSize="base",
        ACCESS_TOKEN=HF_TOKEN,
        model_type="faster-whisper",
        skip_enhance=True,
        compress=True,
    )

    stem = tmp_audio.stem.strip()
    source_dir = tmp_audio.parent
    artifacts_dir = source_dir / f".{stem}"

    return source_dir, stem, artifacts_dir


@needs_env
def test_vtt_limpio_exists_in_source_folder(published_result):
    """transcript_limpio.vtt queda junto al audio original."""
    source_dir, stem, _ = published_result
    vtt_path = source_dir / f"{stem}_limpio.vtt"
    assert vtt_path.exists(), f"No se encontro {vtt_path}"
    content = vtt_path.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT"), "VTT no tiene header WEBVTT"
    assert len(content) > 20, "VTT parece vacio"


@needs_env
def test_m4a_limpio_exists_in_source_folder(published_result):
    """audio comprimido queda como {stem}_limpio.m4a junto al original."""
    source_dir, stem, _ = published_result
    m4a_path = source_dir / f"{stem}_limpio.m4a"
    assert m4a_path.exists(), f"No se encontro {m4a_path}"
    assert m4a_path.stat().st_size > 0, "M4A esta vacio"


@needs_env
def test_old_m4a_naming_not_present(published_result):
    """El naming viejo {stem}.m4a NO debe existir."""
    source_dir, stem, _ = published_result
    old_m4a = source_dir / f"{stem}.m4a"
    assert not old_m4a.exists(), f"Naming viejo aun existe: {old_m4a}"


@needs_env
def test_artifacts_dir_keeps_originals(published_result):
    """artifacts_dir mantiene transcript original intacto."""
    _, _, artifacts_dir = published_result
    vtt_original = artifacts_dir / "transcript_en.vtt"
    assert vtt_original.exists(), f"Original borrado de artifacts: {vtt_original}"


@needs_env
def test_publish_without_compress(tmp_path):
    """Sin compress=True, solo se publica el VTT (no hay M4A)."""
    from speechlib.core_analysis import core_analysis

    tmp_audio = tmp_path / AUDIO.name
    shutil.copy2(AUDIO, tmp_audio)

    core_analysis(
        str(tmp_audio),
        voices_folder=None,
        log_folder=str(tmp_path / "logs"),
        language="en",
        modelSize="base",
        ACCESS_TOKEN=HF_TOKEN,
        model_type="faster-whisper",
        skip_enhance=True,
        compress=False,
    )

    stem = tmp_audio.stem.strip()
    vtt_path = tmp_path / f"{stem}_limpio.vtt"
    m4a_path = tmp_path / f"{stem}_limpio.m4a"

    assert vtt_path.exists(), f"VTT limpio no publicado: {vtt_path}"
    assert not m4a_path.exists(), "M4A no deberia existir sin compress=True"
