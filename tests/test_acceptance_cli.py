"""AT: CLI valida argumentos antes de iniciar trabajo costoso.

Typer valida paths y tokens upfront. Errores claros, sin GPU wasted.
"""
from pathlib import Path
from typer.testing import CliRunner

runner = CliRunner()


def _invoke(*args: str):
    from speechlib.__main__ import app
    return runner.invoke(app, ["run"] + list(args))


class TestPreflightValidation:

    def test_missing_audio_file_exits_with_error(self, tmp_path):
        result = _invoke(str(tmp_path / "nonexistent.wav"))
        assert result.exit_code != 0

    def test_voices_folder_not_found_exits_with_error(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"\x00" * 100)
        result = _invoke(str(wav), "--voices-folder", str(tmp_path / "no_voices"))
        assert result.exit_code != 0

    def test_missing_hf_token_exits_with_error(self, tmp_path, monkeypatch):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"\x00" * 100)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        result = _invoke(str(wav), "--token", "")
        assert result.exit_code != 0

    def test_valid_args_pass_preflight(self, tmp_path, monkeypatch):
        """Args válidos pasan preflight (core_analysis puede fallar después)."""
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"\x00" * 100)
        voices = tmp_path / "voices"
        voices.mkdir()
        monkeypatch.setenv("HF_TOKEN", "test_token")
        # Llega a core_analysis que falla por audio inválido, pero
        # preflight validation NO es la causa del error
        result = _invoke(str(wav), "--voices-folder", str(voices))
        assert result.exit_code != 2  # 2 = typer validation error
