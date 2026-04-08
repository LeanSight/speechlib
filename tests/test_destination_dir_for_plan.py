"""
Slice 16 unit tests: _destination_dir_for_plan (puro).

Verifica la decision de "donde va cada plan" sin tocar filesystem.
"""

from pathlib import Path


def _plan(label: str, identified: bool):
    from speechlib.domain.sample_extraction import SampleClip, SpeakerSamplePlan

    return SpeakerSamplePlan(
        speaker_label=label,
        is_identified=identified,
        clips=(SampleClip(0, 1000),),
    )


def test_identified_plan_goes_to_root_named_after_speaker():
    from speechlib.services.extract_samples import _destination_dir_for_plan

    plan = _plan("Manuel Olguin", identified=True)
    out = Path("/x/samples")

    assert _destination_dir_for_plan(plan, out) == Path("/x/samples/Manuel Olguin")


def test_unidentified_plan_goes_to_por_nombrar_subfolder():
    from speechlib.services.extract_samples import _destination_dir_for_plan

    plan = _plan("SPEAKER_03", identified=False)
    out = Path("/x/samples")

    assert _destination_dir_for_plan(plan, out) == Path("/x/samples/por_nombrar/SPEAKER_03")


def test_destination_is_pure_function():
    """Llamar dos veces con el mismo input devuelve el mismo output."""
    from speechlib.services.extract_samples import _destination_dir_for_plan

    plan = _plan("X", identified=True)
    out = Path("/y/samples")

    assert _destination_dir_for_plan(plan, out) == _destination_dir_for_plan(plan, out)
