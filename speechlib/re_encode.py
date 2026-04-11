import wave

import torchaudio

from .audio_state import AudioState
from .step_timer import timed


@timed("re_encode")
def re_encode(state: AudioState) -> AudioState:
    try:
        with wave.open(str(state.working_path), 'rb') as f:
            if f.getparams().sampwidth == 2:
                return state.model_copy(update={"is_16bit": True})
    except wave.Error:
        pass

    waveform, sr = torchaudio.load(str(state.working_path))
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = state.artifacts_dir / "16bit.wav"
    torchaudio.save(str(out_path), waveform, sr, bits_per_sample=16)

    return state.model_copy(update={"working_path": out_path, "is_16bit": True})
