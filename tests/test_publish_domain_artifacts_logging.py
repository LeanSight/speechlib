"""
Smell 3: _publish_domain_artifacts loggea el traceback COMPLETO cuando algo
falla, en lugar del swallow silencioso del solo mensaje.

Antes: print(f"WARNING: ... ({exc}). Legacy output OK.")
Despues: usar logger.exception() que incluye el traceback completo, lo que
permite diagnosticar regresiones en la nueva feature de Slice 5.

Tests sin mocks: capturamos el output via caplog (pytest fixture), no
patcheamos nada. La funcion se invoca con argumentos invalidos para forzar
una excepcion natural.
"""

import logging


def test_publish_domain_artifacts_logs_full_traceback_on_failure(tmp_path, caplog):
    """Si _publish_domain_artifacts falla, el log debe contener tanto el
    mensaje del error como el traceback (line numbers, stack frames)."""
    from speechlib.audio_state import AudioState
    from speechlib.core_analysis import _publish_domain_artifacts

    # Setup minimal valido para audio_state pero sin annotation real
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"fake")
    state = AudioState(source_path=audio, working_path=audio)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Pasar annotation invalido (None) para que itertracks() rompa
    # de manera predecible
    with caplog.at_level(logging.WARNING):
        _publish_domain_artifacts(
            common_segments=[],
            annotation=None,  # provoca AttributeError dentro del try
            speaker_map={},
            state=state,
            language="es",
        )

    # 1. Hay al menos un log message
    assert len(caplog.records) >= 1, "Esperaba al menos un log record"

    # 2. El log captura el traceback (no solo el mensaje)
    full_text = caplog.text
    assert "domain transcript publish failed" in full_text
    # Traceback contiene "Traceback" o referencias a frames
    assert "Traceback" in full_text or "_publish_domain_artifacts" in full_text, (
        f"Esperaba traceback en log, got:\n{full_text}"
    )


def test_publish_domain_artifacts_does_not_raise_on_failure(tmp_path):
    """El swallow del except debe seguir siendo non-raising — si falla,
    el legacy output sigue funcionando. Solo cambia el HOW se loggea."""
    from speechlib.audio_state import AudioState
    from speechlib.core_analysis import _publish_domain_artifacts

    audio = tmp_path / "y.wav"
    audio.write_bytes(b"fake")
    state = AudioState(source_path=audio, working_path=audio)
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Pasar annotation invalido
    try:
        _publish_domain_artifacts(
            common_segments=[],
            annotation=None,
            speaker_map={},
            state=state,
            language="es",
        )
    except Exception as exc:
        raise AssertionError(
            f"_publish_domain_artifacts no debe propagar excepciones, "
            f"pero levantó: {type(exc).__name__}: {exc}"
        )
