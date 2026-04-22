import torch
from .whisper_sinhala import (whisper_sinhala)
from faster_whisper import WhisperModel, BatchedInferencePipeline
import whisper
import os
from transformers import pipeline
import assemblyai as aai
from functools import lru_cache

TRANSCRIPTION_BATCH_SIZE = 4


@lru_cache(maxsize=4)
def _get_faster_whisper_model(model_size: str, device: str, compute_type: str) -> "WhisperModel":
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _load_faster_whisper_model(model_size: str, quantization: bool) -> "WhisperModel":
    """Selecciona device + compute_type según CUDA disponibilidad y retorna el modelo."""
    if torch.cuda.is_available():
        compute_type = "int8_float16" if quantization else "float16"
        return _get_faster_whisper_model(model_size, "cuda", compute_type)
    compute_type = "int8" if quantization else "float32"
    return _get_faster_whisper_model(model_size, "cpu", compute_type)


def transcribe_full_aligned(
    file_name, segments, language, model_size, quantization,
    *, initial_prompt: str | None = None,
):
    """Transcribe el audio completo de una vez y mapea texto por overlap de timestamp.

    En lugar de llamar a transcribe() N veces (una por segmento), esta funcion:
    1. Llama a batched.transcribe() UNA sola vez con el audio completo
    2. Mapea el texto resultante a cada segmento de diarizacion por overlap temporal

    Args:
        file_name: ruta al archivo de audio
        segments: lista de [start, end, speaker] de diarizacion
        language: codigo de idioma
        model_size: tamano del modelo whisper
        quantization: si usar cuantizacion
        initial_prompt: texto opcional que sesga la decodificación de Whisper
            hacia términos de dominio (nombres, jerga, siglas). None = sin sesgo.

    Returns:
        lista de [start, end, text, speaker]
    """
    model = _load_faster_whisper_model(model_size, quantization)
    batched = BatchedInferencePipeline(model=model)
    whisper_segments, _ = batched.transcribe(
        file_name, language=language, beam_size=1, batch_size=TRANSCRIPTION_BATCH_SIZE,
        word_timestamps=True, initial_prompt=initial_prompt,
    )
    return _assign_text_to_segments(list(whisper_segments), segments)


def _assign_text_to_segments(whisper_segs, segments):
    """Asigna el texto de cada whisper_seg al segmento de diarización correspondiente.

    Word-level: cada palabra va al segmento cuya ventana contiene su midpoint
    (fallback: segmento con mayor overlap). Segment-level (sin word timestamps):
    cada whisper_seg al segmento con mayor overlap.
    """
    seg_texts = [[] for _ in segments]

    for ws in whisper_segs:
        words = getattr(ws, "words", None) or []
        if words:
            for word in words:
                mid = (word.start + word.end) / 2
                best_idx = -1
                best_overlap = 0.0
                for i, seg in enumerate(segments):
                    if seg[0] <= mid <= seg[1]:
                        best_idx = i
                        break
                    overlap = min(word.end, seg[1]) - max(word.start, seg[0])
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = i
                if best_idx >= 0:
                    seg_texts[best_idx].append(word.word.strip())
        else:
            best_idx = -1
            best_overlap = 0.0
            for i, seg in enumerate(segments):
                overlap = min(ws.end, seg[1]) - max(ws.start, seg[0])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            if best_idx >= 0:
                seg_texts[best_idx].append(ws.text)

    result = []
    for i, seg in enumerate(segments):
        start_s, end_s, speaker = seg[0], seg[1], seg[2]
        text = " ".join(seg_texts[i]).strip()
        result.append([start_s, end_s, text, speaker])
    return result


def transcribe(file, language, model_size, model_type, quantization, custom_model_path, hf_model_path, aai_api_key):
    res = ""
    if language in ["si", "Si"]:
        res = whisper_sinhala(file)
        return res
    elif model_size in ["base", "tiny", "small", "medium", "large", "large-v1", "large-v2", "large-v3"]:
        if model_type == "faster-whisper":
            model = _load_faster_whisper_model(model_size, quantization)

            if language in model.supported_languages:
                batched = BatchedInferencePipeline(model=model)
                segments, info = batched.transcribe(
                    file,
                    language=language,
                    beam_size=5,
                    batch_size=16,
                )

                for segment in segments:
                    res += segment.text + " "

                return res
            else:
                Exception("Language code not supported.\nThese are the supported languages:\n", model.supported_languages)
        elif model_type == "whisper":
            try:
                if torch.cuda.is_available():
                    model = whisper.load_model(model_size, device="cuda")
                    result = model.transcribe(file, language=language, fp16=True)
                    res = result["text"]
                else:
                    model = whisper.load_model(model_size, device="cpu")
                    result = model.transcribe(file, language=language, fp16=False)
                    res = result["text"]

                return res
            except Exception as err:
                print("an error occured while transcribing: ", err)
        elif model_type == "custom":
            model_folder = os.path.dirname(custom_model_path)
            model_folder = model_folder + "/"
            print("model file: ", custom_model_path)
            print("model fodler: ", model_folder)
            try:
                if torch.cuda.is_available():
                    model = whisper.load_model(custom_model_path, download_root=model_folder, device="cuda")
                    result = model.transcribe(file, language=language, fp16=True)
                    res = result["text"]
                else:
                    model = whisper.load_model(custom_model_path, download_root=model_folder, device="cpu")
                    result = model.transcribe(file, language=language, fp16=False)
                    res = result["text"]

                return res
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        elif model_type == "huggingface":
            try:
                if torch.cuda.is_available():
                    pipe = pipeline("automatic-speech-recognition", model=hf_model_path, device="cuda")
                    result = pipe(file)
                    res = result['text']
                else:
                    pipe = pipeline("automatic-speech-recognition", model=hf_model_path, device="cpu")
                    result = pipe(file)
                    res = result['text']
                return res
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        elif model_type == "assemblyAI":
            try:
                # Replace with your API key
                aai.settings.api_key = aai_api_key

                # You can set additional parameters for the transcription
                config = aai.TranscriptionConfig(
                    speech_model=aai.SpeechModel.nano,
                    language_code=language
                )

                transcriber = aai.Transcriber(config=config)
                transcript = transcriber.transcribe(file)

                if transcript.status == aai.TranscriptStatus.error:
                    print(transcript.error)
                    raise Exception(f"an error occured while transcribing: {transcript.error}")
                else:
                    res = transcript.text
                    return res
            except Exception as err:
                raise Exception(f"an error occured while transcribing: {err}")
        else:
            raise Exception(f"model_type {model_type} is not supported")
    else:
        raise Exception("only 'base', 'tiny', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3' models are available.")

