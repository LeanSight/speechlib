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


def test_torchaudio_save_emits_no_bits_per_sample_warning(tmp_path):
    """#2 bits_per_sample: torchaudio.save no debe avisar (kwarg no-op removido).

    torchaudio 2.10+ enruta save() por TorchCodec AudioEncoder, que ignora
    bits_per_sample y avisa por cada llamada. PCM 16-bit es el default para un
    target .wav, asi que quitar el kwarg deja la salida intacta y elimina el
    warning en la fuente. Aqui se verifica via resample_to_16k (uno de los 5
    call sites) que NO se emite el warning y que la salida sigue siendo 16-bit.
    """
    import wave
    from conftest import make_wav
    from speechlib.resample_to_16k import resample_to_16k
    from speechlib.audio_state import AudioState

    wav = make_wav(tmp_path / "audio.wav", framerate=44100, n_frames=4410)
    state = AudioState(source_path=wav, working_path=wav,
                       is_wav=True, is_mono=True, is_16bit=True)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = resample_to_16k(state)

    assert not any(
        "bits_per_sample" in str(w.message) for w in recorded
    ), "torchaudio.save no deberia avisar por bits_per_sample"

    with wave.open(str(result.working_path), "rb") as wf:
        assert wf.getsampwidth() == 2, "la salida debe seguir siendo PCM 16-bit"


def test_triton_flop_counter_log_warning_is_silenced():
    """#1 triton: el logger de torch.utils.flop_counter no debe loguear WARNING.

    Es un logging.Logger.warning (NO warnings.warn), asi que filterwarnings no
    lo silencia: hay que subir el nivel del logger. Triton no tiene wheel oficial
    de Windows; su ausencia es legitima y el FLOP counter no se usa.
    """
    import logging
    from speechlib.core_analysis import _configure_runtime_warnings

    _configure_runtime_warnings()

    logger = logging.getLogger("torch.utils.flop_counter")
    assert logger.getEffectiveLevel() >= logging.ERROR, (
        "el logger torch.utils.flop_counter deberia estar en nivel >= ERROR"
    )
