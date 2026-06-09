"""Acceptance: speechlib silencia los runtime warnings ruidosos de sus deps.

Origen: devdocs/IMPROVE-runtime-warnings.md. Cada slice suprime un warning
concreto emitido por torch/torchaudio/pyannote durante el pipeline. La politica
de supresion vive en core_analysis.py, justo tras el import de compat, y se
aplica como side-effect al importar el modulo.

Patron de test (supresion via warnings): importar core_analysis registra el
filtro 'ignore'; dentro de catch_warnings(record=True) SIN simplefilter, los
filtros registrados siguen activos, asi que re-emitir el warning no lo graba.
"""
import warnings


def test_statspool_std_dof_warning_is_suppressed():
    """#4 StatsPool: 'std(): degrees of freedom is <= 0' no debe propagarse.

    pyannote emite este UserWarning internamente al diarizar segmentos de 1
    frame. speechlib no puede evitar la llamada std() de pyannote, pero si
    suprimir el ruido. (El path de embedding propio de speechlib ya esta
    guardado por MIN_SEGMENT_DURATION_S / select_segments_for_embedding.)
    """
    from speechlib.core_analysis import _configure_runtime_warnings

    with warnings.catch_warnings(record=True) as recorded:
        _configure_runtime_warnings()
        warnings.warn(
            "std(): degrees of freedom is <= 0. Correction should be strictly "
            "less than the reduction factor.",
            UserWarning,
        )

    assert not any(
        "degrees of freedom" in str(w.message) for w in recorded
    ), "el UserWarning de std() dof <= 0 deberia estar suprimido"


def test_pyannote_tf32_reproducibility_warning_is_suppressed():
    """#3 TF32: pyannote ReproducibilityWarning no debe propagarse.

    pyannote 4.x deshabilita TF32 en CUDA al mover el pipeline a device y lo
    anuncia con un ReproducibilityWarning (subclase de UserWarning). speechlib
    acepta el disable (exactitud > velocidad) y solo silencia el anuncio. La
    politica de re-habilitar TF32 (Ampere+) queda fuera de este slice.
    """
    from speechlib.core_analysis import _configure_runtime_warnings
    from pyannote.audio.utils.reproducibility import ReproducibilityWarning

    with warnings.catch_warnings(record=True) as recorded:
        _configure_runtime_warnings()
        warnings.warn("TensorFloat-32 (TF32) has been disabled", ReproducibilityWarning)

    assert not any(
        issubclass(w.category, ReproducibilityWarning) for w in recorded
    ), "el ReproducibilityWarning de TF32 deberia estar suprimido"
