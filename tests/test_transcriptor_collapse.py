"""
Smell 4: Transcriptor

(a) Docstring de 220 lineas con todos los idiomas soportados que pertenece a
    documentacion externa, no al codigo.

(b) 5 metodos transcribe (whisper, faster_whisper, custom_whisper,
    huggingface_model, assemby_ai_model) que son el mismo metodo con un
    parametro model_type distinto. Clasico caso de colapsar.

(c) El typo "assemby_ai_model" es API publica — duele. Renombrar a
    "assembly_ai_model" sin romper backward compat: agregar el correctamente
    escrito como canonical, mantener el typo como alias deprecado.

Tests sin mocks de dominio. Usamos patch al BOUNDARY (core_analysis) como
test seam — Khorikov: mockear en limites del sistema esta justificado.
"""

import warnings
from unittest.mock import patch


def test_transcriptor_has_canonical_assembly_ai_model_method():
    """El metodo correctamente escrito existe y es callable."""
    from speechlib.speechlib import Transcriptor

    assert hasattr(Transcriptor, "assembly_ai_model")
    assert callable(Transcriptor.assembly_ai_model)


def test_transcriptor_keeps_typo_alias_for_backward_compat():
    """El typo legacy sigue existiendo (no rompemos clientes)."""
    from speechlib.speechlib import Transcriptor

    assert hasattr(Transcriptor, "assemby_ai_model")
    assert callable(Transcriptor.assemby_ai_model)


def test_typo_alias_emits_deprecation_warning():
    """Llamar al alias deprecado emite DeprecationWarning para que el cliente
    vea que debe migrar al nombre correcto."""
    from speechlib.speechlib import Transcriptor

    t = Transcriptor.__new__(Transcriptor)
    t.file = "x.wav"
    t.voices_folder = None
    t.language = "es"
    t.log_folder = "/tmp"
    t.modelSize = "tiny"
    t.quantization = False
    t.ACCESS_TOKEN = "fake"

    with patch("speechlib.speechlib.core_analysis", return_value=[]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t.assemby_ai_model("api_key")

    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        f"Esperaba DeprecationWarning, got: {[w.category.__name__ for w in caught]}"
    )
    deprecation_msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("assemby_ai_model" in m and "assembly_ai_model" in m for m in deprecation_msgs), (
        f"Esperaba mensaje mencionando ambos nombres, got: {deprecation_msgs}"
    )


def test_canonical_method_does_not_emit_deprecation():
    """El metodo canonical NO debe emitir warnings."""
    from speechlib.speechlib import Transcriptor

    t = Transcriptor.__new__(Transcriptor)
    t.file = "x.wav"
    t.voices_folder = None
    t.language = "es"
    t.log_folder = "/tmp"
    t.modelSize = "tiny"
    t.quantization = False
    t.ACCESS_TOKEN = "fake"

    with patch("speechlib.speechlib.core_analysis", return_value=[]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t.assembly_ai_model("api_key")

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation_warnings == [], (
        f"assembly_ai_model NO debe emitir DeprecationWarning, got: "
        f"{[str(w.message) for w in deprecation_warnings]}"
    )


def test_all_5_transcribe_methods_dispatch_correct_model_type():
    """Cada metodo publico llama a core_analysis con el model_type correcto."""
    from speechlib.speechlib import Transcriptor

    t = Transcriptor.__new__(Transcriptor)
    t.file = "x.wav"
    t.voices_folder = None
    t.language = "es"
    t.log_folder = "/tmp"
    t.modelSize = "tiny"
    t.quantization = False
    t.ACCESS_TOKEN = "fake"

    expected_model_types = {
        "whisper": "whisper",
        "faster_whisper": "faster-whisper",
        "custom_whisper": "custom",
        "huggingface_model": "huggingface",
        "assembly_ai_model": "assemblyAI",
    }

    for method_name, expected_model_type in expected_model_types.items():
        with patch("speechlib.speechlib.core_analysis", return_value=[]) as mock_ca:
            method = getattr(t, method_name)
            # Llamar con argumentos extra segun el metodo
            if method_name == "custom_whisper":
                method("/path/custom_model")
            elif method_name == "huggingface_model":
                method("hf_id")
            elif method_name == "assembly_ai_model":
                method("api_key")
            else:
                method()

            # core_analysis fue llamado con el model_type correcto en posicion 7
            # (file, voices_folder, log_folder, language, modelSize, ACCESS_TOKEN, model_type, ...)
            args, kwargs = mock_ca.call_args
            assert args[6] == expected_model_type, (
                f"{method_name}: esperaba model_type={expected_model_type!r}, got {args[6]!r}"
            )


def test_typo_alias_delegates_to_canonical():
    """assemby_ai_model debe llamar al mismo path que assembly_ai_model."""
    from speechlib.speechlib import Transcriptor

    t = Transcriptor.__new__(Transcriptor)
    t.file = "x.wav"
    t.voices_folder = None
    t.language = "es"
    t.log_folder = "/tmp"
    t.modelSize = "tiny"
    t.quantization = False
    t.ACCESS_TOKEN = "fake"

    # Capturar args de ambos llamados
    canonical_args = None
    typo_args = None

    with patch("speechlib.speechlib.core_analysis", return_value=[]) as mock_ca:
        t.assembly_ai_model("k1")
        canonical_args = mock_ca.call_args.args

    with patch("speechlib.speechlib.core_analysis", return_value=[]) as mock_ca2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            t.assemby_ai_model("k1")
        typo_args = mock_ca2.call_args.args

    assert canonical_args == typo_args


def test_transcriptor_docstring_does_not_enumerate_languages():
    """El docstring NO debe contener la lista de idiomas (220 lineas).
    Esa info pertenece a documentacion externa."""
    from speechlib.speechlib import Transcriptor

    doc = Transcriptor.__init__.__doc__ or ""
    # Heuristica: si el docstring tuviera la lista, contendria muchos idiomas.
    # Verificamos que no esta enumerando.
    language_markers = ["Afrikaans", "Amharic", "Cantonese", "Yoruba", "Yiddish"]
    found = [m for m in language_markers if m in doc]
    assert len(found) == 0, (
        f"Docstring no debe enumerar idiomas, encontrados: {found}. "
        "Mover a documentacion externa."
    )
