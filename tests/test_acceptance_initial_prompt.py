"""AT: initial_prompt fluye desde CLI hasta batched.transcribe().

Behavior: el usuario puede pasar `--initial-prompt "<texto>"` a `speechlib run`
para sesgar la decodificación de faster-whisper hacia términos de dominio
(nombres propios, jerga, siglas). El texto se entrega como kwarg `initial_prompt`
al llamado de `batched.transcribe()`.

Frontera externa de entrada: CLI. Frontera externa de salida: faster-whisper
(verificada vía fake que captura los kwargs recibidos). El kwarg
`initial_prompt` es estándar de faster-whisper:
https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
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


def test_transcribe_full_aligned_forwards_initial_prompt():
    """Given un initial_prompt, When se llama transcribe_full_aligned,
    Then batched.transcribe recibe ese texto como kwarg initial_prompt."""
    mock_model = _make_mock_model()
    mock_pipeline = _make_mock_pipeline()
    diarization_segs = [[0.0, 3.0, "SPEAKER_00"]]

    prompt = "Reunión BCI Seguros. Términos: LLM, POC, Jolyon."

    with (
        patch("speechlib.transcribe._get_faster_whisper_model", return_value=mock_model),
        patch("speechlib.transcribe.BatchedInferencePipeline", return_value=mock_pipeline),
    ):
        transcribe_full_aligned(
            "audio.wav", diarization_segs, "es", "large-v3-turbo", False,
            initial_prompt=prompt,
        )

    kwargs = mock_pipeline.transcribe.call_args.kwargs
    assert kwargs.get("initial_prompt") == prompt, (
        f"esperaba initial_prompt={prompt!r} en batched.transcribe kwargs, "
        f"recibí: {kwargs}"
    )


def test_transcribe_full_aligned_default_initial_prompt_is_none():
    """Default: cuando el usuario no pasa initial_prompt, batched.transcribe
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
    assert kwargs.get("initial_prompt") is None, (
        f"esperaba initial_prompt=None por default, recibí: {kwargs}"
    )


def test_cli_run_exposes_initial_prompt_flag():
    """El subcomando `run` acepta --initial-prompt y lo documenta en --help."""
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0, f"help fallo: {result.output}"
    assert "--initial-prompt" in result.output, (
        f"--initial-prompt no aparece en `run --help`. Output:\n{result.output}"
    )
