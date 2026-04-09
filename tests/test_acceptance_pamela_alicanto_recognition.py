"""
AT: el bug observado en Alicanto community-1 — el cluster SPEAKER_00 contiene
predominantemente a Pamela Falconi (verificado: centroide de los 5 sample
clips a distance 0.306 contra library Pamela, similarity 0.694 >> threshold
0.45). PERO el speaker_map.json post-pipeline dice
{'SPEAKER_00': 'SPEAKER_00'} (no identificado).

Esperabamos que el library matching identificara a Pamela. La metrica del
analisis de samples confirma que el cluster ES Pamela. Por lo tanto, hay
un bug en _compute_averaged_embeddings_per_tag o en assign_speakers que
hace que el matching falle pese a tener el embedding correcto cerca.

Test: usar el cache real de Alicanto (community-1) y verificar que la
identificacion de Pamela funciona. Skip si los artefactos no estan.

GOOS sin mocks: el test usa archivos reales del cache + library real.
No mockea pyannote ni el embedding model.
"""
import json
from pathlib import Path

import pytest

ART = Path(r"C:\workspace\@recordings\20260402 Alicanto\.Voz 260402_151510")
ENHANCED = ART / "enhanced.wav"
RTTM = ART / "diarization.rttm"
VOICES = Path(r"C:\workspace\#dev\speechlib\transcript_samples\voices")

artifacts_available = ENHANCED.exists() and RTTM.exists() and VOICES.exists()


@pytest.mark.skipif(not artifacts_available, reason="Alicanto cache no disponible")
def test_pamela_cluster_should_match_library():
    """Si _compute_averaged_embeddings_per_tag + _best_match operan sobre
    el cluster que es Pamela, deben identificarla."""
    import sys
    sys.path.insert(0, r"c:\workspace\dev\speechlib")

    from pyannote.database.util import load_rttm
    from speechlib.audio_state import AudioState
    from speechlib.core_analysis import _compute_averaged_embeddings_per_tag
    from speechlib.domain.recognition import _best_match
    from speechlib.speaker_recognition import (
        SPEAKER_SIMILARITY_MIN_MARGIN,
        SPEAKER_SIMILARITY_THRESHOLD,
        load_avg_voice_embeddings,
    )

    # Load community-1 RTTM (cached)
    annotation = next(iter(load_rttm(str(RTTM)).values()))
    speakers = {}
    for turn, _, tag in annotation.itertracks(yield_label=True):
        speakers.setdefault(tag, []).append([turn.start, turn.end, tag])

    # Find which SPEAKER_XX has the most segments — likely a "talking person"
    # Pamela in 3.1 was identified, so she should be a real cluster here
    # We don't know which SPEAKER_XX is Pamela in community-1, so test ALL of them
    state = AudioState(
        source_path=ENHANCED,
        working_path=ENHANCED,
        is_wav=True, is_mono=True, is_16bit=True, is_16khz=True,
        is_normalized=True, is_enhanced=True,
    )

    embeddings_by_tag = _compute_averaged_embeddings_per_tag(state, speakers)
    library = load_avg_voice_embeddings(VOICES, enhanced=True)

    # Run _best_match for each tag and find which one matches Pamela
    pamela_matches = []
    for tag, emb in embeddings_by_tag.items():
        name, sim = _best_match(
            emb, library,
            threshold=SPEAKER_SIMILARITY_THRESHOLD,
            min_margin=SPEAKER_SIMILARITY_MIN_MARGIN,
        )
        if name == "Pamela Falconi":
            pamela_matches.append((tag, sim))

    # Por user's ground truth: Pamela esta en este audio. Al menos UN cluster
    # debe matchear con Pamela.
    assert pamela_matches, (
        "Pamela Falconi NOT identified by any SPEAKER_XX. "
        "Expected at least one cluster to match her library embedding."
    )


@pytest.mark.skipif(not artifacts_available, reason="Alicanto cache no disponible")
def test_full_speaker_map_after_select_segments_fix():
    """Reporta el speaker_map completo post-fix.

    Ground truth del usuario:
    - SPEAKER_01 (~78m): Agustin Villena (en library)
    - SPEAKER_00 (~8m): Pamela Falconi (en library, era el bug)
    - Otros: Orlando/Daniel/Nicolas/Marcos (NO estan en library, deben quedar SPEAKER_XX)
    """
    import sys
    sys.path.insert(0, r"c:\workspace\dev\speechlib")

    from pyannote.database.util import load_rttm
    from speechlib.audio_state import AudioState
    from speechlib.core_analysis import _compute_averaged_embeddings_per_tag
    from speechlib.domain.recognition import _best_match
    from speechlib.speaker_recognition import (
        SPEAKER_SIMILARITY_MIN_MARGIN,
        SPEAKER_SIMILARITY_THRESHOLD,
        load_avg_voice_embeddings,
    )

    annotation = next(iter(load_rttm(str(RTTM)).values()))
    speakers = {}
    for turn, _, tag in annotation.itertracks(yield_label=True):
        speakers.setdefault(tag, []).append([turn.start, turn.end, tag])

    state = AudioState(
        source_path=ENHANCED, working_path=ENHANCED,
        is_wav=True, is_mono=True, is_16bit=True, is_16khz=True,
        is_normalized=True, is_enhanced=True,
    )

    embeddings_by_tag = _compute_averaged_embeddings_per_tag(state, speakers)
    library = load_avg_voice_embeddings(VOICES, enhanced=True)

    results = {}
    for tag in sorted(speakers):
        emb = embeddings_by_tag.get(tag)
        if emb is None:
            results[tag] = ("NO_EMBEDDING", None)
            continue
        name, sim = _best_match(
            emb, library,
            threshold=SPEAKER_SIMILARITY_THRESHOLD,
            min_margin=SPEAKER_SIMILARITY_MIN_MARGIN,
        )
        results[tag] = (name or "<unidentified>", float(sim) if sim else None)

    # Print summary
    print("\n=== Alicanto speaker_map post-fix ===")
    for tag, (name, sim) in results.items():
        sim_str = f"{sim:.3f}" if sim is not None else "n/a"
        print(f"  {tag}: {name} (sim={sim_str})")

    # Hard expectations:
    pamela_tags = [t for t, (n, _) in results.items() if n == "Pamela Falconi"]
    agustin_tags = [t for t, (n, _) in results.items() if n == "Agustin Villena"]
    assert pamela_tags, "Pamela Falconi not identified"
    assert agustin_tags, "Agustin Villena not identified"

    # Negative invariant: en Alicanto solo deben identificarse Pamela y Agustin.
    # Cualquier OTRO match con la library es un falso positivo, porque los
    # demas speakers reales (Orlando, Daniel, Nicolas, Marcos) NO estan en
    # library. Cristian Ruiz (AA) es de un contexto distinto y no asiste a
    # esta reunion.
    identified = {n for (n, _) in results.values() if n not in ("<unidentified>", "NO_EMBEDDING")}
    expected = {"Pamela Falconi", "Agustin Villena"}
    false_positives = identified - expected
    assert not false_positives, (
        f"False positive identifications detected: {false_positives}. "
        f"Only Pamela and Agustin should be identified in Alicanto."
    )
