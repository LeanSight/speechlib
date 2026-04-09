"""
AT: speaker recognition identifica correctamente a los speakers del audio
de ejemplo obama_zach.wav contra la library examples/voices/.

Ground truth: 2 speakers (obama, zach). Ambos estan en la library.
No debe haber falsos positivos ni falsos negativos.

Usa rutas relativas al proyecto y voces de ejemplo (no reales).

Uso:
    pytest tests/test_acceptance_speaker_recognition.py -v -s
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES = PROJECT_ROOT / "examples"
AUDIO = EXAMPLES / "obama_zach.wav"
RTTM = EXAMPLES / ".obama_zach" / "diarization.rttm"
VOICES = EXAMPLES / "voices"

skip_reason = []
if not AUDIO.exists():
    skip_reason.append(f"audio no encontrado: {AUDIO}")
if not RTTM.exists():
    skip_reason.append(f"RTTM no encontrado: {RTTM}")
if not VOICES.exists():
    skip_reason.append(f"voices no encontrada: {VOICES}")

needs_artifacts = pytest.mark.skipif(bool(skip_reason), reason=" | ".join(skip_reason) or "ok")


@pytest.fixture(scope="module")
def speaker_map():
    """Computa speaker_map completo sobre obama_zach con la library de ejemplo."""
    from pyannote.database.util import load_rttm

    from speechlib.audio_state import AudioState
    from speechlib.core_analysis import (
        _build_speaker_groups,
        _compute_averaged_embeddings_per_tag,
    )
    from speechlib.domain.recognition import _best_match
    from speechlib.speaker_recognition import (
        SPEAKER_SIMILARITY_MIN_MARGIN,
        SPEAKER_SIMILARITY_THRESHOLD,
        load_avg_voice_embeddings,
    )

    annotation = next(iter(load_rttm(str(RTTM)).values()))
    _, speakers, _, _ = _build_speaker_groups(annotation)

    state = AudioState(
        source_path=AUDIO,
        working_path=AUDIO,
        is_wav=True, is_mono=True, is_16bit=True, is_16khz=True,
    )

    embeddings_by_tag = _compute_averaged_embeddings_per_tag(state, speakers)
    library = load_avg_voice_embeddings(VOICES, enhanced=False)

    results = {}
    for tag in sorted(speakers):
        emb = embeddings_by_tag.get(tag)
        if emb is None:
            results[tag] = (None, None)
            continue
        name, sim = _best_match(
            emb, library,
            threshold=SPEAKER_SIMILARITY_THRESHOLD,
            min_margin=SPEAKER_SIMILARITY_MIN_MARGIN,
        )
        results[tag] = (name, float(sim) if sim else None)

    print("\n=== obama_zach speaker_map ===")
    for tag, (name, sim) in results.items():
        sim_str = f"{sim:.3f}" if sim is not None else "n/a"
        print(f"  {tag}: {name or '<unidentified>'} (sim={sim_str})")

    return results


@needs_artifacts
def test_both_speakers_identified(speaker_map):
    """Obama y Zach deben ser identificados."""
    identified = {name for name, _ in speaker_map.values() if name}
    assert "obama" in identified, f"obama no identificado. Map: {speaker_map}"
    assert "zach" in identified, f"zach no identificado. Map: {speaker_map}"


@needs_artifacts
def test_no_false_positives(speaker_map):
    """Solo obama y zach deben ser identificados, ningun otro."""
    identified = {name for name, _ in speaker_map.values() if name}
    expected = {"obama", "zach"}
    false_positives = identified - expected
    assert not false_positives, f"Falsos positivos: {false_positives}"


@needs_artifacts
def test_no_duplicate_identifications(speaker_map):
    """Cada speaker de la library debe matchear con a lo mas un tag."""
    names = [name for name, _ in speaker_map.values() if name]
    duplicates = [n for n in set(names) if names.count(n) > 1]
    assert not duplicates, f"Speaker identificado en multiples tags: {duplicates}"
