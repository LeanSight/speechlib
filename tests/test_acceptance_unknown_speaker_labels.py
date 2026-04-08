"""
AT: speakers no reconocidos conservan su tag SPEAKER_XX en el output legacy.

Slice 13b: cuando _run_speaker_recognition_cached no encuentra match en la
libreria, el speaker_map.json escribe el tag pyannote como valor (NUNCA el
literal "unknown"). Esto se propaga a common_segments y al VTT writer.

Estos tests verifican el invariante a nivel del flujo completo de
core_analysis con mocks minimos y patch del helper.
"""

from contextlib import ExitStack
from unittest.mock import patch, MagicMock
from pathlib import Path


def _make_diarization_mock(speakers):
    turns = []
    for i, spk in enumerate(speakers):
        turn = MagicMock()
        turn.start = float(i * 5)
        turn.end = float(i * 5 + 4)
        turns.append((turn, None, spk))

    mock_diarization = MagicMock()
    mock_diarization.itertracks.return_value = turns
    mock_diarization.speaker_diarization = mock_diarization
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_diarization
    return mock_pipeline


def _common_patches(mock_diar_pipeline):
    """Patches comunes a los 3 tests: passthrough de pre-processing + diarization."""
    state_passthrough = lambda s: s
    return [
        patch("speechlib.core_analysis.convert_to_wav", side_effect=state_passthrough),
        patch("speechlib.core_analysis.convert_to_mono", side_effect=state_passthrough),
        patch("speechlib.core_analysis.re_encode", side_effect=state_passthrough),
        patch("speechlib.core_analysis.resample_to_16k", side_effect=state_passthrough),
        patch("speechlib.core_analysis.loudnorm", side_effect=state_passthrough),
        patch("speechlib.core_analysis.enhance_audio", side_effect=state_passthrough),
        patch(
            "speechlib.core_analysis._get_diarization_pipeline",
            return_value=mock_diar_pipeline,
        ),
        patch("torchaudio.load", return_value=(MagicMock(), 16000)),
        patch("speechlib.core_analysis.write_log_file"),
        patch("speechlib.core_analysis.merge_short_turns", side_effect=lambda s: s),
        patch("speechlib.core_analysis.absorb_micro_segments", side_effect=lambda s: s),
    ]


def _fake_transcribe(audio, common, lang, model, quant):
    return [[seg[0], seg[1], "texto", seg[2]] for seg in common]


def test_two_unknown_speakers_keep_their_speaker_xx_tags(tmp_path):
    """Slice 13b: dos speakers no reconocidos → conservan SPEAKER_00 y
    SPEAKER_01 (jamas el literal 'unknown')."""
    from speechlib import core_analysis as ca

    mock_pipeline = _make_diarization_mock(["SPEAKER_00", "SPEAKER_01"])
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)

    # Patch del helper para devolver speaker_map sin matches
    fake_speaker_map = {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_01"}

    with ExitStack() as stack:
        for p in _common_patches(mock_pipeline):
            stack.enter_context(p)
        stack.enter_context(patch(
            "speechlib.core_analysis._run_speaker_recognition_cached",
            return_value=fake_speaker_map,
        ))
        stack.enter_context(patch(
            "speechlib.core_analysis.transcribe_full_aligned",
            side_effect=_fake_transcribe,
        ))
        segments = ca.core_analysis(
            str(wav),
            voices_folder="fake_voices",
            log_folder=str(tmp_path),
            language="es",
            modelSize="large-v3",
            ACCESS_TOKEN="fake_token",
            model_type="faster-whisper",
            skip_enhance=True,
        )

    labels = {s[3] for s in segments}
    assert "unknown" not in labels
    assert "SPEAKER_00" in labels
    assert "SPEAKER_01" in labels


def test_known_speaker_and_unknown_not_merged(tmp_path):
    """SPEAKER_00 reconocido como Agustin, SPEAKER_01 no reconocido →
    el desconocido conserva SPEAKER_01, ambos sobreviven."""
    from speechlib import core_analysis as ca

    mock_pipeline = _make_diarization_mock(["SPEAKER_00", "SPEAKER_01"])
    wav = tmp_path / "b.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)

    fake_speaker_map = {"SPEAKER_00": "Agustin", "SPEAKER_01": "SPEAKER_01"}

    with ExitStack() as stack:
        for p in _common_patches(mock_pipeline):
            stack.enter_context(p)
        stack.enter_context(patch(
            "speechlib.core_analysis._run_speaker_recognition_cached",
            return_value=fake_speaker_map,
        ))
        stack.enter_context(patch(
            "speechlib.core_analysis.transcribe_full_aligned",
            side_effect=_fake_transcribe,
        ))
        segments = ca.core_analysis(
            str(wav),
            voices_folder="fake_voices",
            log_folder=str(tmp_path),
            language="es",
            modelSize="large-v3",
            ACCESS_TOKEN="fake_token",
            model_type="faster-whisper",
            skip_enhance=True,
        )

    labels = {s[3] for s in segments}
    assert "Agustin" in labels
    assert "SPEAKER_01" in labels
    assert "unknown" not in labels


def test_single_unknown_speaker_keeps_speaker_00_tag(tmp_path):
    """Un solo speaker no reconocido → conserva SPEAKER_00 (no 'unknown')."""
    from speechlib import core_analysis as ca

    mock_pipeline = _make_diarization_mock(["SPEAKER_00"])
    wav = tmp_path / "c.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)

    fake_speaker_map = {"SPEAKER_00": "SPEAKER_00"}

    with ExitStack() as stack:
        for p in _common_patches(mock_pipeline):
            stack.enter_context(p)
        stack.enter_context(patch(
            "speechlib.core_analysis._run_speaker_recognition_cached",
            return_value=fake_speaker_map,
        ))
        stack.enter_context(patch(
            "speechlib.core_analysis.transcribe_full_aligned",
            side_effect=_fake_transcribe,
        ))
        segments = ca.core_analysis(
            str(wav),
            voices_folder="fake_voices",
            log_folder=str(tmp_path),
            language="es",
            modelSize="large-v3",
            ACCESS_TOKEN="fake_token",
            model_type="faster-whisper",
            skip_enhance=True,
        )

    labels = {s[3] for s in segments}
    assert "SPEAKER_00" in labels
    assert "unknown" not in labels
