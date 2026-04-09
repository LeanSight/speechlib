import warnings

from .core_analysis import core_analysis
from .re_encode import re_encode
from .convert_to_mono import convert_to_mono
from .convert_to_wav import convert_to_wav
from .resample_to_16k import resample_to_16k
from .loudnorm import loudnorm
from .enhance_audio import enhance_audio
from .audio_state import AudioState


class Transcriptor:
    def __init__(
        self,
        file,
        log_folder,
        language,
        modelSize,
        ACCESS_TOKEN,
        voices_folder=None,
        quantization=False,
    ):
        """Transcribe a wav file with speaker diarization.

        Args:
            file: name of wav file with extension (e.g. ``file.wav``)
            log_folder: name of folder where transcript will be stored
            language: ISO language code (``es``, ``en``, ``pt``, ...). The
                full list of supported languages is documented externally
                at https://github.com/Navodplayer1/speechlib (Smell 4: 220
                lineas de enumeracion movidas a docs externos).
            modelSize: ``tiny``, ``small``, ``medium``, ``large``,
                ``large-v1``, ``large-v2``, ``large-v3``. Modelos mas grandes
                son mas precisos pero mas lentos.
            ACCESS_TOKEN: huggingface access token
            voices_folder: folder con subfolders por speaker con voice samples
                para speaker recognition. Default ``None`` (sin recognition).
            quantization: int8 quantization (default ``False``)

        Methods:
            ``whisper()``           openai-whisper
            ``faster_whisper()``    faster-whisper (recommended)
            ``custom_whisper(p)``   custom whisper model
            ``huggingface_model(id)``  HuggingFace transcription model
            ``assembly_ai_model(k)`` AssemblyAI cloud transcription
        """
        # Smell 4: docstring enumerando 99 idiomas movido a docs externos.
        self.file = file
        self.voices_folder = voices_folder
        self.language = language
        self.log_folder = log_folder
        self.modelSize = modelSize
        self.quantization = quantization
        self.ACCESS_TOKEN = ACCESS_TOKEN

    def _run(
        self,
        model_type: str,
        custom_model_path=None,
        hf_model_id=None,
        aai_api_key=None,
    ):
        """Helper privado: dispatcha a core_analysis con el model_type correcto.
        Smell 4: colapsa los 5 metodos publicos en uno parametrizado."""
        return core_analysis(
            self.file,
            self.voices_folder,
            self.log_folder,
            self.language,
            self.modelSize,
            self.ACCESS_TOKEN,
            model_type,
            self.quantization,
            custom_model_path,
            hf_model_id,
            aai_api_key,
        )

    def whisper(self):
        return self._run("whisper")

    def faster_whisper(self):
        return self._run("faster-whisper")

    def custom_whisper(self, custom_model_path):
        return self._run("custom", custom_model_path=custom_model_path)

    def huggingface_model(self, hf_model_id):
        return self._run("huggingface", hf_model_id=hf_model_id)

    def assembly_ai_model(self, aai_api_key):
        """Transcribe usando AssemblyAI cloud API."""
        return self._run("assemblyAI", aai_api_key=aai_api_key)

    def assemby_ai_model(self, aai_api_key):
        """DEPRECATED: typo legacy. Use ``assembly_ai_model`` instead."""
        warnings.warn(
            "Transcriptor.assemby_ai_model is deprecated due to a typo; "
            "use Transcriptor.assembly_ai_model instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.assembly_ai_model(aai_api_key)


class PreProcessor:
    """
    class for preprocessing audio files.

    methods:

    convert_to_wav(file)   -> convert any format to WAV
    convert_to_mono(file)  -> convert stereo to mono
    re_encode(file)        -> re-encode to 16-bit PCM
    resample_to_16k(file)  -> resample to 16 kHz
    loudnorm(file)         -> normalize to -14 LUFS EBU R128
    enhance_audio(file)    -> speech enhancement (ClearVoice MossFormer2_SE_48K)

    """

    @staticmethod
    def _apply(step, file):
        """Helper privado: aplica un audio step (re_encode, loudnorm, etc.)
        a un file path. Encapsula la conversion path -> AudioState -> path
        que vivia repetida en cada metodo publico (Smell 5)."""
        from pathlib import Path

        state = AudioState(source_path=Path(file), working_path=Path(file))
        return str(step(state).working_path)

    def re_encode(self, file):
        return self._apply(re_encode, file)

    def convert_to_mono(self, file):
        return self._apply(convert_to_mono, file)

    def convert_to_wav(self, file):
        return self._apply(convert_to_wav, file)

    def resample_to_16k(self, file):
        return self._apply(resample_to_16k, file)

    def loudnorm(self, file):
        return self._apply(loudnorm, file)

    def enhance_audio(self, file):
        return self._apply(enhance_audio, file)
