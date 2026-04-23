"""AT: hotwords fluye desde CLI hasta batched.transcribe().

Behavior: el usuario puede pasar `--hotwords "term1,term2,..."` a `speechlib run`
para sesgar la decodificación de faster-whisper a nivel de logit bias puntual
(no como contexto de prompt). La lista se entrega como kwarg `hotwords` al
llamado de `batched.transcribe()`.

Frontera externa de entrada: CLI. Frontera externa de salida: faster-whisper
(verificada vía fake que captura los kwargs recibidos). El kwarg `hotwords`
es estándar de faster-whisper BatchedInferencePipeline desde v1.x.
"""
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from speechlib.transcribe import transcribe_full_aligned
from speechlib.__main__ import app


def _make_mock_model():
    mock = MagicMock()
    mock.supported_languages = ["es", "en"]
    return mock


def _make_mock_pipeline():
    mock = MagicMock()
    seg = MagicMock()
    seg.start = 0.0
    seg.end = 3.0
    seg.text = "hola mundo"
    seg.words = []
    mock.transcribe.return_value = ([seg], MagicMock())
    return mock


def test_transcribe_full_aligned_forwards_hotwords():
    """Given una lista hotwords, When se llama transcribe_full_aligned,
    Then batched.transcribe recibe los términos como string space-joined
    (faster-whisper's hotwords kwarg espera str, no list)."""
    mock_model = _make_mock_model()
    mock_pipeline = _make_mock_pipeline()
    diarization_segs = [[0.0, 3.0, "SPEAKER_00"]]

    hotwords = ["Patricio", "Alejandra", "Aguas Andinas", "Esri"]

    with (
        patch("speechlib.transcribe._get_faster_whisper_model", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_pipeline),
    ):
        transcribe_full_aligned(
            "audio.wav", diarization_segs, "es", "large-v3-turbo", False,
            hotwords=hotwords,
        )

    kwargs = mock_pipeline.transcribe.call_args.kwargs
    expected = "Patricio Alejandra Aguas Andinas Esri"
    assert kwargs.get("hotwords") == expected, (
        f"esperaba hotwords={expected!r} (string) en batched.transcribe kwargs, "
        f"recibí: {kwargs!r}"
    )


def test_transcribe_full_aligned_default_hotwords_is_none():
    """Default: cuando el usuario no pasa hotwords, batched.transcribe
    lo recibe como None (comportamiento original preservado)."""
    mock_model = _make_mock_model()
    mock_pipeline = _make_mock_pipeline()
    diarization_segs = [[0.0, 3.0, "SPEAKER_00"]]

    with (
        patch("speechlib.transcribe._get_faster_whisper_model", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_pipeline),
    ):
        transcribe_full_aligned(
            "audio.wav", diarization_segs, "es", "large-v3-turbo", False,
        )

    kwargs = mock_pipeline.transcribe.call_args.kwargs
    assert kwargs.get("hotwords") is None, (
        f"esperaba hotwords=None por default, recibí: {kwargs}"
    )


def test_cli_run_exposes_hotwords_flag():
    """El subcomando `run` acepta --hotwords y lo documenta en --help."""
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0, f"help fallo: {result.output}"
    assert "--hotwords" in result.output, (
        f"--hotwords no aparece en `run --help`. Output:\n{result.output}"
    )
