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
