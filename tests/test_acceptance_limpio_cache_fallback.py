"""AT: confirm resuelve cache desde `<stem>_limpio.<ext>` al cache `<stem>/`.

Behavior: cuando el pipeline publica `<stem>_limpio.<ext>` al lado del source y el
usuario borra el original para ahorrar espacio, el cache `.<stem>/` sigue en disk.
`speechlib confirm <stem>_limpio.<ext>` debe encontrar el cache existente y aplicar
el speaker_map, en lugar de fallar buscando `.<stem>_limpio/`.
"""
import json
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES = PROJECT_ROOT / "examples"
AUDIO = EXAMPLES / "obama_zach.wav"
CACHE_FIXTURE = EXAMPLES / ".obama_zach"

skip_reason = []
if not AUDIO.exists():
    skip_reason.append(f"audio no encontrado: {AUDIO}")
if not (CACHE_FIXTURE / "transcript.json").exists():
    skip_reason.append(f"transcript.json no encontrado en {CACHE_FIXTURE}")

needs_artifacts = pytest.mark.skipif(
    bool(skip_reason), reason=" | ".join(skip_reason) or "ok"
)


@needs_artifacts
def test_confirm_resuelve_cache_cuando_solo_existe_limpio(tmp_path):
    """Given cache `.<stem>/` del run previo + audio comprimido
    `<stem>_limpio.<ext>` (el original fue borrado),
    When el usuario corre `speechlib confirm <stem>_limpio.<ext>`,
    Then speechlib usa el cache `.<stem>/`, aplica el speaker_map,
    y el VTT publicado contiene los nombres mapeados.
    """
    from typer.testing import CliRunner
    from speechlib.__main__ import app

    # Setup: solo existe el audio "_limpio" + cache sin sufijo (original borrado)
    limpio_audio = tmp_path / f"{AUDIO.stem}_limpio{AUDIO.suffix}"
    shutil.copy2(AUDIO, limpio_audio)

    cache_dir = tmp_path / f".{AUDIO.stem}"
    shutil.copytree(CACHE_FIXTURE, cache_dir)

    # El cache con stem _limpio NO existe (verificación de precondición)
    assert not (tmp_path / f".{limpio_audio.stem}").exists()

    # Usuario escribe su speaker_map.json usando los tags del transcript
    transcript = json.loads((cache_dir / "transcript.json").read_text(encoding="utf-8"))
    tags = sorted({seg["speaker"]["diarization_tag"] for seg in transcript["segments"]})
    assert len(tags) >= 1, f"transcript sin tags de speaker: {transcript}"

    user_map = {tags[0]: "TestPersonLimpio"}
    (cache_dir / "speaker_map.json").write_text(
        json.dumps(user_map), encoding="utf-8"
    )

    # When: invocar confirm pasando el `_limpio.<ext>`
    runner = CliRunner()
    result = runner.invoke(app, ["confirm", str(limpio_audio)])

    # Then: exit 0 y VTT publicado con el nombre mapeado
    assert result.exit_code == 0, (
        f"confirm fallo (exit_code={result.exit_code}). Output:\n{result.output}"
    )

    vtt_path = limpio_audio.parent / f"{limpio_audio.stem}_limpio.vtt"
    assert vtt_path.exists(), f"VTT no publicado: {vtt_path}"
    vtt_content = vtt_path.read_text(encoding="utf-8")
    assert "[TestPersonLimpio]" in vtt_content, (
        f"VTT esperaba [TestPersonLimpio] aplicado tras confirm. "
        f"Contenido (primeros 500 chars): {vtt_content[:500]}"
    )
