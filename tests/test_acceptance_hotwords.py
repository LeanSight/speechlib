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


def test_cli_run_reads_hotwords_from_file_with_at_prefix(tmp_path, monkeypatch):
    """Given un archivo de keyterms con comentarios y líneas vacías,
    When el usuario corre `speechlib run <audio> --hotwords @<path>`,
    Then los términos del archivo llegan a core_analysis como list[str],
    ignorando líneas vacías y comentarios (#).

    Este es el behavior nuevo: el prefijo '@' indica "leer de archivo"
    en lugar de interpretar el valor como CSV inline."""
    keyterms_file = tmp_path / "keyterms.txt"
    keyterms_file.write_text(
        "# participantes\n"
        "Patricio\n"
        "Alejandra\n"
        "\n"
        "# stack\n"
        "Aguas Andinas\n",
        encoding="utf-8",
    )
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake audio content")

    captured = {}

    def fake_core_analysis(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("speechlib.__main__.core_analysis", fake_core_analysis)
    monkeypatch.setenv("HF_TOKEN", "dummy-token")

    runner = CliRunner()
    result = runner.invoke(app, [
        "run", str(audio),
        "--hotwords", f"@{keyterms_file}",
    ])

    assert result.exit_code == 0, (
        f"CLI falló (exit={result.exit_code}): {result.output}"
    )
    assert captured.get("hotwords") == ["Patricio", "Alejandra", "Aguas Andinas"], (
        f"esperaba hotwords=['Patricio', 'Alejandra', 'Aguas Andinas'] "
        f"(leídos del archivo), recibí: {captured.get('hotwords')!r}"
    )


def test_parse_hotwords_reads_file_with_at_prefix(tmp_path):
    """Unit: _parse_hotwords('@<path>') lee una línea por término,
    ignora líneas vacías y comentarios (#)."""
    from speechlib.__main__ import _parse_hotwords
    f = tmp_path / "kt.txt"
    f.write_text("# header\nPatricio\nAlejandra\n\n# stack\nAguas Andinas\n",
                 encoding="utf-8")
    assert _parse_hotwords(f"@{f}") == ["Patricio", "Alejandra", "Aguas Andinas"]


def test_parse_hotwords_csv_splits_by_comma_and_strips():
    """Unit: _parse_hotwords('a, b,c') sigue funcionando como CSV inline."""
    from speechlib.__main__ import _parse_hotwords
    assert _parse_hotwords("Patricio, Alejandra,Aguas Andinas") == [
        "Patricio", "Alejandra", "Aguas Andinas",
    ]


def test_parse_hotwords_none_when_value_is_none_or_empty():
    """Unit: _parse_hotwords(None) y _parse_hotwords('') → None."""
    from speechlib.__main__ import _parse_hotwords
    assert _parse_hotwords(None) is None
    assert _parse_hotwords("") is None


def test_cli_run_hotwords_csv_still_works_as_before(tmp_path, monkeypatch):
    """Regresión: --hotwords "a,b,c" (sin prefijo @) sigue funcionando
    como CSV inline — no debe interpretarse como path."""
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake audio content")

    captured = {}

    def fake_core_analysis(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("speechlib.__main__.core_analysis", fake_core_analysis)
    monkeypatch.setenv("HF_TOKEN", "dummy-token")

    runner = CliRunner()
    result = runner.invoke(app, [
        "run", str(audio),
        "--hotwords", "Patricio,Alejandra,Aguas Andinas",
    ])

    assert result.exit_code == 0, (
        f"CLI falló (exit={result.exit_code}): {result.output}"
    )
    assert captured.get("hotwords") == ["Patricio", "Alejandra", "Aguas Andinas"], (
        f"esperaba hotwords parseado como CSV, recibí: {captured.get('hotwords')!r}"
    )
