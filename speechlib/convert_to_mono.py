import torchaudio
from .audio_state import AudioState
from .step_timer import timed


@timed("convert_to_mono")
def convert_to_mono(state: AudioState) -> AudioState:
    waveform, sample_rate = torchaudio.load(str(state.working_path))

    if waveform.shape[0] == 1:
        return state.model_copy(update={"is_mono": True})

    mono = waveform.mean(dim=0, keepdim=True)

    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    mono_path = state.artifacts_dir / "mono.wav"
    torchaudio.save(str(mono_path), mono, sample_rate)

    return state.model_copy(update={"working_path": mono_path, "is_mono": True})
