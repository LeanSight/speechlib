"""
Slice 10 tests: relabel_vtt como CLI delgado sobre el dominio.

Tests focalizados en las helpers puras + el flujo via CLI con audio sintetico
en tmp_path. Sin mocks profundos: el dominio puro ya esta probado en
test_acceptance_assign_speakers.py / test_domain_recognition.py.

Cubre el invariante anti-bug: re-evaluacion fallida JAMAS sobreescribe
con literal "[unknown]".
"""

import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torchaudio


def _make_wav(path: Path, duration_s: float = 5.0, sr: int = 16000) -> Path:
    n = int(duration_s * sr)
    torchaudio.save(str(path), 0.1 * torch.randn(1, n), sr, bits_per_sample=16)
    return path


def _write_vtt(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


# ── Pure helpers ──────────────────────────────────────────────────────────────


class TestBuildTranscriptFromVttBlocks:
    def test_unidentified_speaker_xx_label(self):
        from speechlib.tools.relabel_vtt import build_transcript_from_vtt_blocks
        from speechlib.vtt_utils import VttBlock

        blocks = [
            VttBlock(
                index="1",
                start_ms=0, end_ms=2000,
                speaker="SPEAKER_03", text="hola",
                raw_timestamp="00:00:00.000 --> 00:00:02.000",
            ),
        ]
        t = build_transcript_from_vtt_blocks(blocks, "x.wav", "es")

        assert len(t.segments) == 1
        spk = t.segments[0].speaker
        assert spk.diarization_tag == "SPEAKER_03"
        assert spk.recognized_name is None
        assert spk.label == "SPEAKER_03"

    def test_identified_name_label(self):
        from speechlib.tools.relabel_vtt import build_transcript_from_vtt_blocks
        from speechlib.vtt_utils import VttBlock

        blocks = [
            VttBlock(
                index="1",
                start_ms=0, end_ms=2000,
                speaker="Manuel Olguin", text="hola",
                raw_timestamp="00:00:00.000 --> 00:00:02.000",
            ),
        ]
        t = build_transcript_from_vtt_blocks(blocks, "x.wav", "es")

        spk = t.segments[0].speaker
        assert spk.diarization_tag == "Manuel Olguin"
        assert spk.recognized_name == "Manuel Olguin"
        assert spk.label == "Manuel Olguin"

    def test_unknown_literal_treated_as_unidentified(self):
        from speechlib.tools.relabel_vtt import build_transcript_from_vtt_blocks
        from speechlib.vtt_utils import VttBlock

        blocks = [
            VttBlock(
                index="1",
                start_ms=0, end_ms=2000,
                speaker="unknown", text="hola",
                raw_timestamp="00:00:00.000 --> 00:00:02.000",
            ),
        ]
        t = build_transcript_from_vtt_blocks(blocks, "x.wav", "es")

        spk = t.segments[0].speaker
        assert spk.recognized_name is None


class TestApplyTranscriptLabelsToBlocks:
    def test_applies_new_labels_returns_changed_count(self):
        from speechlib.domain.transcript import (
            SpeakerIdentity,
            Transcript,
            TranscriptSegment,
        )
        from speechlib.tools.relabel_vtt import apply_transcript_labels_to_blocks
        from speechlib.vtt_utils import VttBlock

        blocks = [
            VttBlock(
                index="1",
                start_ms=0, end_ms=1000,
                speaker="SPEAKER_00", text="x",
                raw_timestamp="00:00:00.000 --> 00:00:01.000",
            ),
            VttBlock(
                index="2",
                start_ms=1000, end_ms=2000,
                speaker="SPEAKER_01", text="y",
                raw_timestamp="00:00:01.000 --> 00:00:02.000",
            ),
        ]
        transcript = Transcript(
            segments=(
                TranscriptSegment(
                    start_ms=0, end_ms=1000, text="x",
                    speaker=SpeakerIdentity(
                        diarization_tag="SPEAKER_00",
                        recognized_name="Manuel",
                        similarity=0.7,
                    ),
                ),
                TranscriptSegment(
                    start_ms=1000, end_ms=2000, text="y",
                    speaker=SpeakerIdentity(diarization_tag="SPEAKER_01"),
                ),
            ),
            audio_path="x.wav",
            language="es",
        )

        changed = apply_transcript_labels_to_blocks(blocks, transcript)

        assert changed == 1
        assert blocks[0].speaker == "Manuel"
        assert blocks[1].speaker == "SPEAKER_01"  # invariante anti-bug


# ── CLI integration con audio sintetico ──────────────────────────────────────


class TestRelabelVttCli:
    def _voices(self, tmp_path: Path) -> Path:
        voices = tmp_path / "voices"
        (voices / "Manuel").mkdir(parents=True)
        _make_wav(voices / "Manuel" / "segment_01.wav", duration_s=2.0)
        return voices

    def _run(self, vtt, audio, voices, *extra):
        from speechlib.tools import relabel_vtt as m

        argv = ["relabel_vtt", str(vtt), str(audio), str(voices), *extra]
        with patch("sys.argv", argv):
            m.main()

    def test_unidentified_block_label_preserved_when_no_match(self, tmp_path):
        """Caso central del bug: SPEAKER_03 sin match → conserva SPEAKER_03,
        NO se sobreescribe con [unknown]."""
        from speechlib.tools.relabel_vtt import compute_embeddings_per_label

        vtt = _write_vtt(
            tmp_path / "t.vtt",
            """\
            WEBVTT

            1
            00:00:00.000 --> 00:00:02.000
            [SPEAKER_03] hola
            """,
        )
        audio = _make_wav(tmp_path / "a.wav", duration_s=3.0)
        voices = self._voices(tmp_path)

        # Library con embedding ortogonal al de SPEAKER_03 → no match
        with (
            patch(
                "speechlib.tools.relabel_vtt.load_avg_voice_embeddings",
                return_value={"Manuel": np.array([1.0, 0.0, 0.0])},
            ),
            patch(
                "speechlib.tools.relabel_vtt.compute_embeddings_per_label",
                return_value={"SPEAKER_03": np.array([0.0, 0.0, 1.0])},
            ),
        ):
            self._run(vtt, audio, voices)

        out = vtt.with_stem(vtt.stem + "_relabeled")
        content = out.read_text(encoding="utf-8")
        assert "[unknown]" not in content
        assert "[SPEAKER_03]" in content

    def test_identified_block_relabeled_to_match(self, tmp_path):
        """SPEAKER_03 con match al embedding de Manuel → relabel a Manuel."""
        vtt = _write_vtt(
            tmp_path / "t.vtt",
            """\
            WEBVTT

            1
            00:00:00.000 --> 00:00:02.000
            [SPEAKER_03] hola
            """,
        )
        audio = _make_wav(tmp_path / "a.wav", duration_s=3.0)
        voices = self._voices(tmp_path)

        unit = np.array([1.0, 0.0, 0.0])
        with (
            patch(
                "speechlib.tools.relabel_vtt.load_avg_voice_embeddings",
                return_value={"Manuel": unit},
            ),
            patch(
                "speechlib.tools.relabel_vtt.compute_embeddings_per_label",
                return_value={"SPEAKER_03": unit},
            ),
        ):
            self._run(vtt, audio, voices)

        out = vtt.with_stem(vtt.stem + "_relabeled")
        content = out.read_text(encoding="utf-8")
        assert "[Manuel]" in content
        assert "[SPEAKER_03]" not in content

    def test_default_mode_skips_already_identified(self, tmp_path):
        """Sin --all-speakers: no se computa embedding para bloques ya nombrados."""
        vtt = _write_vtt(
            tmp_path / "t.vtt",
            """\
            WEBVTT

            1
            00:00:00.000 --> 00:00:02.000
            [Manuel] ya identificado

            2
            00:00:02.000 --> 00:00:04.000
            [SPEAKER_03] no identificado
            """,
        )
        audio = _make_wav(tmp_path / "a.wav", duration_s=5.0)
        voices = self._voices(tmp_path)

        compute_calls = []

        def fake_compute(blocks, audio_path, target_labels=None, limit_s=60.0):
            compute_calls.append(target_labels)
            return {}

        with (
            patch(
                "speechlib.tools.relabel_vtt.load_avg_voice_embeddings",
                return_value={"Manuel": np.array([1.0, 0.0])},
            ),
            patch(
                "speechlib.tools.relabel_vtt.compute_embeddings_per_label",
                side_effect=fake_compute,
            ),
        ):
            self._run(vtt, audio, voices)

        # Solo SPEAKER_03 fue target
        assert compute_calls == [{"SPEAKER_03"}]

    def test_all_speakers_mode_targets_every_label(self, tmp_path):
        """Con --all-speakers: target_labels incluye TODOS los labels del VTT."""
        vtt = _write_vtt(
            tmp_path / "t.vtt",
            """\
            WEBVTT

            1
            00:00:00.000 --> 00:00:02.000
            [Manuel] ya identificado

            2
            00:00:02.000 --> 00:00:04.000
            [SPEAKER_03] no identificado
            """,
        )
        audio = _make_wav(tmp_path / "a.wav", duration_s=5.0)
        voices = self._voices(tmp_path)

        compute_calls = []

        def fake_compute(blocks, audio_path, target_labels=None, limit_s=60.0):
            compute_calls.append(target_labels)
            return {}

        with (
            patch(
                "speechlib.tools.relabel_vtt.load_avg_voice_embeddings",
                return_value={"Manuel": np.array([1.0, 0.0])},
            ),
            patch(
                "speechlib.tools.relabel_vtt.compute_embeddings_per_label",
                side_effect=fake_compute,
            ),
        ):
            self._run(vtt, audio, voices, "--all-speakers")

        assert compute_calls == [{"Manuel", "SPEAKER_03"}]
