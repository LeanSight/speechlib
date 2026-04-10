"""
Compatibility shims for upstream library version mismatches.

Smell 7: aislamos los monkey-patches globales aqui para mantenerlos visibles,
documentados y centralizados en lugar de dispersos por el codigo.

Uso: import this module BEFORE the libraries que lo necesitan. Tipicamente
desde el primer import de speechlib (e.g., en core_analysis.py o en
__init__.py si se centraliza ahi).

Cada patch debe documentar:
1. POR QUE existe (incompatibilidad concreta)
2. QUE espera el caller (cual es la API que rompe)
3. CUANDO puede borrarse (issue de upstream + version objetivo)
"""

import torchaudio


def patch_torchaudio_list_audio_backends() -> None:
    """Restaura torchaudio.list_audio_backends() removido en torchaudio 2.x+.

    SpeechBrain 1.0.3 llama esta funcion al importarse via pyannote.
    Borrar cuando: speechbrain >= 1.1.0 sin bug de k2_fsa lazy import.
    """
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["sox"]


def patch_torchaudio_torchcodec() -> None:
    """Reemplaza torchaudio.load/save con implementaciones basadas en PyAV.

    Origen: torchaudio 2.10+ usa torchcodec como unico backend de audio.
    torchcodec requiere FFmpeg shared DLLs (.dll) accesibles al proceso, y en
    Windows con PyTorch CPU-only la carga de libtorchcodec_core*.dll falla con
    WinError 127 aunque FFmpeg este instalado (incompatibilidad de build).

    Este shim detecta si torchaudio.load() falla con RuntimeError de torchcodec
    y en ese caso reemplaza load/save con implementaciones PyAV + scipy/wave.

    Borrar cuando: torchcodec soporte correctamente Windows CPU-only builds
    o torchaudio restaure backends alternativos (soundfile/sox/ffmpeg).
    """
    import functools

    _original_load = torchaudio.load
    _original_save = torchaudio.save

    # Test if torchaudio.load works
    try:
        # Quick probe: try importing the torchcodec decoder directly
        from torchcodec.decoders import AudioDecoder  # noqa: F401
        return  # torchcodec works, no patch needed
    except (RuntimeError, OSError, ImportError):
        pass  # torchcodec broken, apply patch

    def _load_wav_fast(uri, normalize):
        """Fast WAV loader using stdlib wave module (no frame-by-frame decode)."""
        import wave
        import numpy as np
        import torch

        with wave.open(str(uri), 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16)
        elif sampwidth == 4:
            audio = np.frombuffer(raw, dtype=np.int32)
        elif sampwidth == 1:
            audio = np.frombuffer(raw, dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        # Reshape to (channels, samples)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).T
        else:
            audio = audio.reshape(1, -1)

        waveform = torch.from_numpy(audio.copy())

        if normalize:
            if sampwidth == 2:
                waveform = waveform.to(torch.float32) / 32768.0
            elif sampwidth == 4:
                waveform = waveform.to(torch.float32) / 2147483648.0
            elif sampwidth == 1:
                waveform = (waveform.to(torch.float32) - 128.0) / 128.0

        return waveform, sample_rate

    def _load_with_pyav(uri, frame_offset=0, num_frames=-1, normalize=True,
                        channels_first=True, format=None, buffer_size=4096,
                        backend=None):
        """torchaudio.load replacement using wave (WAV) or PyAV (other formats).

        Matches the original torchaudio.load contract:
        - normalize=True (default): int PCM -> float32 in [-1.0, 1.0]
        - normalize=False: returns raw dtype (e.g. int16)
        """
        import numpy as np
        import torch

        uri_str = str(uri)

        # Fast path for WAV files using stdlib (avoids slow frame-by-frame PyAV decode)
        if uri_str.lower().endswith('.wav') and frame_offset == 0 and num_frames == -1:
            try:
                waveform, sample_rate = _load_wav_fast(uri_str, normalize)
                if not channels_first:
                    waveform = waveform.T
                return waveform, sample_rate
            except Exception:
                pass  # Fall through to PyAV

        # General path using PyAV for non-WAV formats
        import av

        container = av.open(uri_str)
        stream = container.streams.audio[0]
        sample_rate = stream.rate

        arrays = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()  # shape: (channels, samples)
            arrays.append(arr)
        container.close()

        if not arrays:
            waveform = torch.zeros(1, 0)
        else:
            audio = np.concatenate(arrays, axis=1)
            waveform = torch.from_numpy(audio.copy())

        # Normalize int formats to float32 [-1.0, 1.0] (torchaudio default)
        if normalize and waveform.dtype != torch.float32:
            if waveform.dtype == torch.int16:
                waveform = waveform.to(torch.float32) / 32768.0
            elif waveform.dtype == torch.int32:
                waveform = waveform.to(torch.float32) / 2147483648.0
            elif waveform.dtype == torch.uint8:
                waveform = (waveform.to(torch.float32) - 128.0) / 128.0
            else:
                waveform = waveform.to(torch.float32)

        # Handle frame_offset and num_frames
        if frame_offset > 0:
            waveform = waveform[:, frame_offset:]
        if num_frames > 0:
            waveform = waveform[:, :num_frames]

        if not channels_first:
            waveform = waveform.T

        return waveform, sample_rate

    def _save_with_wave(uri, src, sample_rate, channels_first=True,
                        format=None, encoding=None, bits_per_sample=16,
                        buffer_size=4096, compression=None, backend=None):
        """torchaudio.save replacement using stdlib wave module."""
        import wave
        import numpy as np
        import torch

        if not channels_first:
            src = src.T

        # Convert to numpy int16
        if src.dtype == torch.float32 or src.dtype == torch.float64:
            max_val = 2 ** (bits_per_sample - 1) - 1
            audio_np = (src.numpy() * max_val).clip(-max_val - 1, max_val).astype(np.int16)
        else:
            audio_np = src.numpy().astype(np.int16)

        n_channels = audio_np.shape[0]
        # Interleave channels: (channels, samples) -> (samples, channels) -> flat
        audio_interleaved = audio_np.T.flatten()

        with wave.open(str(uri), 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(bits_per_sample // 8)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_interleaved.tobytes())

    torchaudio.load = _load_with_pyav
    torchaudio.save = _save_with_wave


# Aplicamos los patches al importar este modulo. Cualquier consumer de speechlib
# que importe `from speechlib import compat` (o cualquier modulo que lo
# transitivamente importe) tendra los patches activos.
patch_torchaudio_list_audio_backends()
patch_torchaudio_torchcodec()
