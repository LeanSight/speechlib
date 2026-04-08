"""
AT: batch_process transcribe múltiples carpetas de audio, identifica speakers
conocidos (Agustin), extrae desconocidos (Patricio, mujeres de Ina TRE)
a voices/_unknown/ y retorna un reporte estructurado.

Tests unitarios (rápidos, con mocks de diarización/transcripción).
El E2E real sobre las grabaciones se hace por separado.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from conftest import make_tone_wav


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_audio_folder(base: Path, name: str, n_files: int = 1) -> Path:
    folder = base / name
    folder.mkdir(parents=True)
    for i in range(n_files):
        make_tone_wav(folder / f"recording_{i:02d}.wav", duration_s=5.0)
    return folder


def _make_voices_dir(tmp_path: Path) -> Path:
    voices = tmp_path / "voices"
    (voices / "Agustin").mkdir(parents=True)
    make_tone_wav(voices / "Agustin" / "segment_01.wav", duration_s=2.0)
    return voices


# ── AT: batch_process retorna reporte correcto ─────────────────────────────────


def test_batch_process_returns_report_per_folder(tmp_path):
    """batch_process retorna un BatchReport con una entrada por carpeta."""
    from speechlib.tools.batch_process import batch_process

    folder_a = _make_audio_folder(tmp_path, "session_a")
    folder_b = _make_audio_folder(tmp_path, "session_b")
    voices = _make_voices_dir(tmp_path)

    with patch("speechlib.tools.batch_process.core_analysis") as mock_ca:
        mock_ca.return_value = [[0.0, 2.0, "hello", "Agustin"]]
        report = batch_process(
            folders=[folder_a, folder_b],
            voices_folder=voices,
            language="es",
            access_token="fake_token",
        )

    assert len(report.folders) == 2
    assert folder_a in report.folders
    assert folder_b in report.folders


def test_batch_process_processes_all_audio_files(tmp_path):
    """core_analysis se llama una vez por archivo de audio encontrado."""
    from speechlib.tools.batch_process import batch_process

    folder = _make_audio_folder(tmp_path, "session", n_files=3)
    voices = _make_voices_dir(tmp_path)

    with patch("speechlib.tools.batch_process.core_analysis") as mock_ca:
        mock_ca.return_value = [[0.0, 1.0, "ok", "Agustin"]]
        report = batch_process(
            folders=[folder],
            voices_folder=voices,
            language="es",
            access_token="fake_token",
        )

    assert mock_ca.call_count == 3


def test_batch_process_finds_audio_extensions(tmp_path):
    """batch_process detecta .wav, .mp3, .m4a, .mp4."""
    from speechlib.tools.batch_process import batch_process

    folder = tmp_path / "mixed"
    folder.mkdir()
    make_tone_wav(folder / "a.wav", duration_s=2.0)
    # Crear archivos dummy de otros formatos (no son WAV reales pero testea el glob)
    (folder / "b.mp3").write_bytes(b"dummy")
    (folder / "c.m4a").write_bytes(b"dummy")
    (folder / "notes.txt").write_bytes(b"ignore")

    voices = _make_voices_dir(tmp_path)

    with patch("speechlib.tools.batch_process.core_analysis") as mock_ca:
        mock_ca.return_value = []
        report = batch_process(
            folders=[folder],
            voices_folder=voices,
            language="es",
            access_token="fake_token",
        )

    # Debe haber intentado procesar wav + mp3 + m4a (3 archivos), no el .txt
    assert mock_ca.call_count == 3


def test_batch_report_lists_unknown_speakers(tmp_path):
    """El reporte incluye speakers desconocidos extraidos del transcript.json
    publicado por core_analysis (Slice 5 cabling).

    Slice 8: ya no se mockea extract_unknown_speakers; el flujo nuevo carga
    el Transcript del aggregate publicado y usa el sample extraction service.
    """
    from speechlib.domain.transcript import (
        SpeakerIdentity,
        Transcript,
        TranscriptSegment,
    )
    from speechlib.tools.batch_process import batch_process

    folder = _make_audio_folder(tmp_path, "session")
    voices = _make_voices_dir(tmp_path)
    unknown_dir = tmp_path / "_unknown"

    # Pre-publicar el transcript.json que core_analysis (mockeado) "habria"
    # generado, en el artifacts_dir esperado.
    audio_path = next(folder.glob("*.wav"))
    artifacts_dir = audio_path.parent / f".{audio_path.stem}"
    artifacts_dir.mkdir(exist_ok=True)
    transcript = Transcript(
        segments=(
            TranscriptSegment(
                start_ms=0, end_ms=2500, text="hola",
                speaker=SpeakerIdentity(
                    diarization_tag="SPEAKER_00",
                    recognized_name="Agustin",
                    similarity=0.7,
                ),
            ),
            TranscriptSegment(
                start_ms=2500, end_ms=5000, text="soy nuevo",
                speaker=SpeakerIdentity(diarization_tag="SPEAKER_01"),
            ),
        ),
        audio_path=str(audio_path),
        language="es",
    )
    transcript.save(artifacts_dir / "transcript.json")

    with patch("speechlib.tools.batch_process.core_analysis") as mock_ca:
        mock_ca.return_value = [
            [0.0, 2.5, "hola", "Agustin"],
            [2.5, 5.0, "soy nuevo", "SPEAKER_01"],
        ]
        report = batch_process(
            folders=[folder],
            voices_folder=voices,
            language="es",
            access_token="fake_token",
            unknown_output_dir=unknown_dir,
            min_unknown_duration_s=2.0,
            max_unknown_clips=2,
        )

    assert len(report.unknown_speakers) >= 1
    # Estructura de salida nueva: <unknown>/<audio_stem>/<speaker_label>/clip_NN.wav
    assert (unknown_dir / audio_path.stem / "SPEAKER_01").is_dir()


def test_batch_report_lists_identified_speakers(tmp_path):
    """El reporte incluye speakers identificados con confianza."""
    from speechlib.tools.batch_process import batch_process

    folder = _make_audio_folder(tmp_path, "session")
    voices = _make_voices_dir(tmp_path)

    with patch("speechlib.tools.batch_process.core_analysis") as mock_ca:
        mock_ca.return_value = [[0.0, 2.0, "hola Agustin", "Agustin"]]
        report = batch_process(
            folders=[folder],
            voices_folder=voices,
            language="es",
            access_token="fake_token",
        )

    assert "Agustin" in report.identified_speakers


def test_batch_process_continues_after_file_error(tmp_path):
    """Si un archivo falla, el batch continúa con los demás."""
    from speechlib.tools.batch_process import batch_process

    folder = _make_audio_folder(tmp_path, "session", n_files=3)
    voices = _make_voices_dir(tmp_path)

    call_count = 0

    def mock_ca(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Audio corrupted")
        return [[0.0, 1.0, "ok", "Agustin"]]

    with patch("speechlib.tools.batch_process.core_analysis", side_effect=mock_ca):
        report = batch_process(
            folders=[folder],
            voices_folder=voices,
            language="es",
            access_token="fake_token",
        )

    assert report.errors == 1
    assert len(report.processed_files) == 2
