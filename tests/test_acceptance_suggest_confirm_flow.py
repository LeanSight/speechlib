"""
AT: speechlib cambia a pipeline suggest+confirm.

Behavior 1: `run` con voices_folder produce VTT con tags [SPEAKER_XX]
crudos Y un speaker_map_suggestions.json con top-N candidatos/scores,
SIN aplicar speaker_map automaticamente al VTT. La decision de
asignar nombres reales queda al subcomando `confirm` (Behavior 2, TBD).

Ground truth: obama_zach.wav tiene 2 speakers (obama, zach), ambos en
examples/voices/. En el comportamiento viejo, el VTT final tenia [obama]
y [zach] aplicados. En el nuevo, tiene [SPEAKER_00] y [SPEAKER_01] crudos.

Uso:
    pytest tests/test_acceptance_suggest_confirm_flow.py -v -s
"""

import json
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES = PROJECT_ROOT / "examples"
AUDIO = EXAMPLES / "obama_zach.wav"
CACHE_FIXTURE = EXAMPLES / ".obama_zach"
VOICES = EXAMPLES / "voices"

skip_reason = []
if not AUDIO.exists():
    skip_reason.append(f"audio no encontrado: {AUDIO}")
if not (CACHE_FIXTURE / "diarization.rttm").exists():
    skip_reason.append(f"diarization.rttm no encontrado en {CACHE_FIXTURE}")
if not VOICES.exists():
    skip_reason.append(f"voices folder no encontrada: {VOICES}")

needs_artifacts = pytest.mark.skipif(bool(skip_reason), reason=" | ".join(skip_reason) or "ok")


@pytest.fixture
def run_with_voices(tmp_path):
    """Corre core_analysis con voices_folder sobre obama_zach copiado a tmp.
    Retorna (tmp_audio, tmp_cache_dir)."""
    from speechlib.core_analysis import core_analysis

    tmp_audio = tmp_path / AUDIO.name
    shutil.copy2(AUDIO, tmp_audio)

    tmp_cache = tmp_path / f".{AUDIO.stem}"
    shutil.copytree(CACHE_FIXTURE, tmp_cache)

    core_analysis(
        str(tmp_audio),
        voices_folder=str(VOICES),
        log_folder=str(tmp_path / "logs"),
        language="en",
        modelSize="base",
        ACCESS_TOKEN=None,  # reusa diarization.rttm del cache
        model_type="faster-whisper",
        skip_enhance=True,
    )
    return tmp_audio, tmp_cache


@pytest.fixture
def run_with_voices_for_confirm(tmp_path):
    """Variante de run_with_voices que el test de confirm consume.

    Separada para no compartir state con el test de Behavior 1.
    """
    from speechlib.core_analysis import core_analysis

    tmp_audio = tmp_path / AUDIO.name
    shutil.copy2(AUDIO, tmp_audio)

    tmp_cache = tmp_path / f".{AUDIO.stem}"
    shutil.copytree(CACHE_FIXTURE, tmp_cache)

    core_analysis(
        str(tmp_audio),
        voices_folder=str(VOICES),
        log_folder=str(tmp_path / "logs"),
        language="en",
        modelSize="base",
        ACCESS_TOKEN=None,
        model_type="faster-whisper",
        skip_enhance=True,
    )
    return tmp_audio, tmp_cache


@needs_artifacts
def test_run_with_voices_produces_raw_vtt_and_suggestions(run_with_voices):
    """run con voices_folder:
    1. Publica VTT con tags [SPEAKER_XX] crudos (no nombres aplicados).
    2. Escribe speaker_map_suggestions.json con top_candidates + recommended.
    3. NO escribe speaker_map.json automaticamente con nombres (identity o ausente).
    """
    tmp_audio, tmp_cache = run_with_voices

    # 1. VTT published con tags raw
    vtt_path = tmp_audio.parent / f"{tmp_audio.stem}_limpio.vtt"
    assert vtt_path.exists(), f"VTT publicado no existe: {vtt_path}"
    vtt_content = vtt_path.read_text(encoding="utf-8")
    assert "[SPEAKER_" in vtt_content, (
        "VTT deberia tener tags [SPEAKER_XX] crudos. "
        f"Contenido (primeros 500 chars): {vtt_content[:500]}"
    )
    assert "[obama]" not in vtt_content, (
        "VTT no debe aplicar 'obama' automaticamente — ese es job de `confirm`. "
        f"Contenido (primeros 500 chars): {vtt_content[:500]}"
    )
    assert "[zach]" not in vtt_content, (
        "VTT no debe aplicar 'zach' automaticamente — ese es job de `confirm`. "
        f"Contenido (primeros 500 chars): {vtt_content[:500]}"
    )

    # 2. Suggestions JSON con estructura correcta
    suggestions_path = tmp_cache / "speaker_map_suggestions.json"
    assert suggestions_path.exists(), (
        f"speaker_map_suggestions.json no escrito en {suggestions_path}"
    )
    suggestions = json.loads(suggestions_path.read_text(encoding="utf-8"))
    assert "tags" in suggestions, f"suggestions sin 'tags': {suggestions}"
    assert len(suggestions["tags"]) >= 2, (
        f"esperaba >=2 clusters (obama + zach): {suggestions}"
    )
    for tag, data in suggestions["tags"].items():
        assert "top_candidates" in data, f"{tag} sin top_candidates: {data}"
        assert "recommended" in data, f"{tag} sin recommended: {data}"
        for cand in data["top_candidates"]:
            assert "name" in cand and "score" in cand, (
                f"candidato malformado en {tag}: {cand}"
            )

    # 3. speaker_map.json NO auto-escrito con nombres reales
    map_path = tmp_cache / "speaker_map.json"
    if map_path.exists():
        m = json.loads(map_path.read_text(encoding="utf-8"))
        for tag, name in m.items():
            assert tag == name, (
                f"speaker_map.json debe ser identity (tag==name) o ausente, "
                f"pero encontre {tag} -> {name}"
            )


@needs_artifacts
def test_confirm_applies_user_speaker_map_to_vtt(run_with_voices_for_confirm):
    """Behavior 2: confirm aplica speaker_map.json (escrito por usuario)
    al VTT publicado, mapeando los tags presentes en el map a nombres
    reales y manteniendo los unmapped como [SPEAKER_XX] literal.
    """
    from typer.testing import CliRunner
    from speechlib.__main__ import app

    tmp_audio, tmp_cache = run_with_voices_for_confirm

    # Leer suggestions para descubrir los tags reales que produjo el pipeline
    suggestions = json.loads(
        (tmp_cache / "speaker_map_suggestions.json").read_text(encoding="utf-8")
    )
    all_tags = sorted(suggestions["tags"].keys())
    assert len(all_tags) >= 2, f"esperaba >=2 tags pyannote: {all_tags}"

    mapped_tag = all_tags[0]
    unmapped_tag = all_tags[1]

    # Usuario escribe su speaker_map.json: solo mapea uno
    user_map = {mapped_tag: "TestPersonAlpha"}
    map_path = tmp_cache / "speaker_map.json"
    map_path.write_text(json.dumps(user_map), encoding="utf-8")

    # Invocar `speechlib confirm <audio>`
    runner = CliRunner()
    result = runner.invoke(app, ["confirm", str(tmp_audio)])
    assert result.exit_code == 0, (
        f"confirm fallo (exit_code={result.exit_code}). Output:\n{result.output}"
    )

    # VTT publicado debe tener TestPersonAlpha aplicado
    vtt_path = tmp_audio.parent / f"{tmp_audio.stem}_limpio.vtt"
    assert vtt_path.exists(), f"VTT no publicado: {vtt_path}"
    vtt_content = vtt_path.read_text(encoding="utf-8")
    assert "[TestPersonAlpha]" in vtt_content, (
        f"VTT esperaba [TestPersonAlpha] aplicado tras confirm. "
        f"Contenido (primeros 800 chars): {vtt_content[:800]}"
    )

    # El cluster no mapeado permanece literal
    assert f"[{unmapped_tag}]" in vtt_content, (
        f"VTT esperaba [{unmapped_tag}] literal (cluster sin mapear). "
        f"Contenido (primeros 800 chars): {vtt_content[:800]}"
    )

    # speaker_map.json del usuario NO se modifica
    assert json.loads(map_path.read_text(encoding="utf-8")) == user_map
