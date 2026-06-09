"""AT: `batch` transcribe varios files en UN proceso (warma whisper+pyannote una
sola vez), llamando core_analysis una vez por archivo — mismo contrato que `run`.

El path ASR real (carga de modelos) va detrás de @pytest.mark.e2e; aquí se fakea
core_analysis para verificar el contrato de orquestación sin GPU.
"""
from typer.testing import CliRunner

from speechlib.__main__ import app


def test_batch_calls_core_analysis_once_per_file_in_one_process(tmp_path, monkeypatch):
    """
    Given  dos media transcribibles
    When   corre `speechlib batch a b`
    Then   core_analysis corre exactamente una vez por archivo, en orden — UN
           proceso recorre los files (los modelos cacheados se cargan una vez).
    """
    a = tmp_path / "a.m4a"
    a.write_bytes(b"fake a")
    b = tmp_path / "b.m4a"
    b.write_bytes(b"fake b")

    calls = []

    def fake_core_analysis(**kwargs):
        calls.append(kwargs["file_name"])

    monkeypatch.setattr("speechlib.__main__.core_analysis", fake_core_analysis)
    monkeypatch.setenv("HF_TOKEN", "dummy-token")

    result = CliRunner().invoke(app, ["batch", str(a), str(b), "--language", "es"])

    assert result.exit_code == 0, f"CLI falló (exit={result.exit_code}): {result.output}"
    assert calls == [str(a), str(b)], "core_analysis corre una vez por archivo, en orden"


def test_batch_forwards_run_parameters_per_file(tmp_path, monkeypatch):
    """
    Given  un batch con --model y --speakers explícitos
    When   corre
    Then   cada llamada a core_analysis recibe los mismos kwargs que `run`
           (modelSize, allowed_speakers parseado, faster-whisper).
    """
    a = tmp_path / "a.m4a"
    a.write_bytes(b"fake a")

    captured = []

    def fake_core_analysis(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr("speechlib.__main__.core_analysis", fake_core_analysis)
    monkeypatch.setenv("HF_TOKEN", "dummy-token")

    result = CliRunner().invoke(
        app, ["batch", str(a), "--model", "large-v3", "--speakers", "Ana, Beto"]
    )

    assert result.exit_code == 0, f"CLI falló (exit={result.exit_code}): {result.output}"
    assert captured[0]["modelSize"] == "large-v3"
    assert captured[0]["allowed_speakers"] == ["Ana", "Beto"]
    assert captured[0]["model_type"] == "faster-whisper"
