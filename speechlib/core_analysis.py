import os
import json
import threading
import time
from .wav_segmenter import wav_file_segmentation
from .transcribe import transcribe_full_aligned
from .step_timer import measure, print_report
from .kernel_profiler import measure as kmeasure, print_report as kprint_report

import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["sox"]

from .diarization import get_diarization_pipeline as _get_diarization_pipeline
from .speaker_recognition import speaker_recognition

try:
    from pyannote.database.util import load_rttm as _load_rttm
except ImportError:
    _load_rttm = None
from .write_log_file import write_log_file
from .segment_merger import (
    merge_short_turns,
    merge_transcript_turns,
    group_by_sentences,
    group_by_speaker,
    absorb_micro_segments,
)

from pathlib import Path
from .audio_state import AudioState
from .re_encode import re_encode
from .convert_to_mono import convert_to_mono
from .convert_to_wav import convert_to_wav
from .resample_to_16k import resample_to_16k
from .loudnorm import loudnorm
from .enhance_audio import enhance_audio
from .compress_audio import compress_audio


def _run_speaker_recognition_cached(
    state: AudioState,
    voices_folder: str,
    speakers: dict,
    speaker_tags: list,
) -> dict:
    """Identifica cada SPEAKER_XX contra la libreria de voces, con cache.

    Si artifacts_dir/speaker_map.json existe, lo carga. Si no, corre
    speaker_recognition() por cada tag y guarda el cache.

    Aplica el hack historico (raiz del bug original): cuando el reconocimiento
    devuelve "unknown", lo reescribe al SPEAKER_XX para que el VTT writer no
    muestre el literal. Slice 13b lo eliminara cuando core_analysis use
    assign_speakers directo.
    """
    speaker_map_path = state.artifacts_dir / "speaker_map.json"

    if speaker_map_path.exists():
        speaker_map = json.loads(speaker_map_path.read_text(encoding="utf-8"))
        print("speaker_map loaded from cache.")
        return speaker_map

    speaker_map: dict = {}
    start_time = int(time.time())
    print("running speaker recognition...")
    for spk_tag, spk_segments in speakers.items():
        spk_name = speaker_recognition(
            str(state.working_path), voices_folder, spk_segments,
            enhanced=state.is_enhanced,
        )
        speaker_map[spk_tag] = spk_name
    elapsed = int(time.time() - start_time)
    print(f"speaker recognition done. Time taken: {elapsed} seconds.")

    # Hack historico: "unknown" -> tag pyannote para que el VTT no muestre literal
    for spk_tag in speaker_tags:
        if speaker_map.get(spk_tag) == "unknown":
            speaker_map[spk_tag] = spk_tag

    speaker_map_path.write_text(
        json.dumps(speaker_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return speaker_map


def _build_speaker_groups(annotation):
    """Itera la annotation pyannote y construye las tres estructuras paralelas
    que el resto del flujo legacy consume.

    Returns:
      common: list[[start, end, SPEAKER_XX]] — segmentos planos en orden.
      speakers: dict[SPEAKER_XX, list[[start, end, SPEAKER_XX]]] — agrupado por tag.
      speaker_tags: list[SPEAKER_XX] — orden de aparicion (sin duplicados).
      speaker_map: dict[SPEAKER_XX, SPEAKER_XX] — placeholder identidad=tag.
    """
    common: list = []
    speakers: dict = {}
    speaker_tags: list = []
    speaker_map: dict = {}

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start = round(turn.start, 1)
        end = round(turn.end, 1)
        common.append([start, end, speaker])

        if speaker not in speaker_tags:
            speaker_tags.append(speaker)
            speaker_map[speaker] = speaker
            speakers[speaker] = []

        speakers[speaker].append([start, end, speaker])

    return common, speakers, speaker_tags, speaker_map


def _run_diarization_cached(state: AudioState, access_token: str | None):
    """Devuelve la annotation pyannote, reutilizando diarization.rttm si existe.

    Recompute si el cache existe pero esta corrupto. Escribe el RTTM cuando
    recomputa.
    """
    rttm_path = state.artifacts_dir / "diarization.rttm"

    if rttm_path.exists():
        try:
            return next(iter(_load_rttm(str(rttm_path)).values())), True
        except Exception as e:
            print(f"WARNING: could not load diarization.rttm ({e}), recomputing.")
            rttm_path.unlink(missing_ok=True)

    pipeline = _get_diarization_pipeline(access_token)
    waveform, sample_rate = torchaudio.load(str(state.working_path))
    print("running diarization...")
    with measure("diarization", gpu=True), kmeasure("diarization"):
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    print("diarization done.")
    annotation = (
        diarization.speaker_diarization
        if hasattr(diarization, "speaker_diarization")
        else diarization
    )
    with open(rttm_path, "w") as f:
        annotation.write_rttm(f)
    return annotation, False


def _preprocess_audio(file_name: str, *, skip_enhance: bool) -> AudioState:
    """Pipeline de pre-processing del audio fuente.

    Pasos: convert_to_wav -> mono -> re_encode -> 16k -> loudnorm -> [enhance].
    Reusa cache 16k.wav cuando existe en artifacts_dir.
    """
    state = AudioState(source_path=Path(file_name), working_path=Path(file_name))
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)

    cached_16k = state.artifacts_dir / "16k.wav"
    if cached_16k.exists():
        state = state.model_copy(
            update={
                "working_path": cached_16k,
                "is_wav": True,
                "is_mono": True,
                "is_16bit": True,
                "is_16khz": True,
            }
        )
    else:
        state = convert_to_wav(state)
        state = convert_to_mono(state)
        state = re_encode(state)
        state = resample_to_16k(state)

    state = loudnorm(state)
    if not skip_enhance:
        state = enhance_audio(state)
    return state


# by default use google speech-to-text API
# if False, then use whisper finetuned version for sinhala
def core_analysis(
    file_name,
    voices_folder,
    log_folder,
    language,
    modelSize="large-v3-turbo",
    ACCESS_TOKEN=None,
    model_type="faster-whisper",
    quantization=False,
    custom_model_path=None,
    hf_model_id=None,
    aai_api_key=None,
    output_format: str = "vtt",
    skip_enhance: bool = False,
    compress: bool = False,
    grouping_mode: str = "sentences",
):
    if log_folder is None:
        log_folder = os.path.join(os.path.dirname(os.path.abspath(file_name)), "output")

    # <-------------------PreProcessing file-------------------------->

    state = _preprocess_audio(file_name, skip_enhance=skip_enhance)

    # <--------------------running analysis--------------------------->

    # Launch compression in background thread (CPU) while diarization runs (GPU)
    compress_thread = None
    if compress:
        compress_thread = threading.Thread(
            target=compress_audio,
            args=(state.working_path, state.source_path.with_suffix(".m4a")),
            daemon=True,
        )
        compress_thread.start()

    annotation, from_cache = _run_diarization_cached(state, ACCESS_TOKEN)
    if from_cache:
        print("diarization loaded from cache.")

    common, speakers, speaker_tags, speaker_map = _build_speaker_groups(annotation)

    has_voices_folder = voices_folder is not None and voices_folder != ""
    if has_voices_folder:
        speaker_map = _run_speaker_recognition_cached(
            state, voices_folder, speakers, speaker_tags
        )

    keys_to_remove = []
    merged = []

    # merging same speakers
    for spk_tag1, spk_segments1 in speakers.items():
        for spk_tag2, spk_segments2 in speakers.items():
            if (
                spk_tag1 not in merged
                and spk_tag2 not in merged
                and spk_tag1 != spk_tag2
                and speaker_map[spk_tag1] == speaker_map[spk_tag2]
            ):
                for segment in spk_segments2:
                    speakers[spk_tag1].append(segment)

                merged.append(spk_tag1)
                merged.append(spk_tag2)
                keys_to_remove.append(spk_tag2)

    # fixing the speaker names in common
    for segment in common:
        speaker = segment[2]
        segment[2] = speaker_map[speaker]

    for key in keys_to_remove:
        del speakers[key]
        del speaker_map[key]

    # absorb micro-segments into longer neighbors, then merge same-speaker turns
    common = absorb_micro_segments(common)
    common = merge_short_turns(common)
    speakers = {}
    for segment in common:
        spk = segment[2]
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append(segment)

    # transcribing the texts differently according to speaker
    print("running transcription...")
    with measure("transcription", gpu=True):
        if model_type == "faster-whisper":
            common_segments = transcribe_full_aligned(
                str(state.working_path), common, language, modelSize, quantization
            )
        else:
            for spk_tag, spk_segments in speakers.items():
                spk = speaker_map.get(spk_tag, spk_tag)
                segment_out = wav_file_segmentation(
                    str(state.working_path),
                    spk_segments,
                    language,
                    modelSize,
                    model_type,
                    quantization,
                    custom_model_path,
                    hf_model_id,
                    aai_api_key,
                )
                speakers[spk_tag] = segment_out
    print("transcription done.")

    if model_type != "faster-whisper":
        common_segments = []
        for item in common:
            speaker = item[2]
            start = item[0]
            end = item[1]

            for spk_tag, spk_segments in speakers.items():
                if speaker == speaker_map.get(spk_tag, spk_tag):
                    for segment in spk_segments:
                        if start == segment[0] and end == segment[1]:
                            common_segments.append([start, end, segment[2], speaker])

    # group post-transcription segments according to grouping_mode
    if model_type == "faster-whisper":
        if grouping_mode == "sentences":
            common_segments = group_by_sentences(common_segments)
        else:
            common_segments = group_by_speaker(common_segments)

    # writing log file
    with measure("write_log_file"):
        write_log_file(
            common_segments,
            log_folder,
            str(state.working_path),
            language,
            output_format,
        )

    # ── Slice 5: publicar el aggregate del nuevo dominio en paralelo ────────
    # transcript.json es el formato canonico futuro; el VTT queda como render.
    # Cero impacto sobre el output legacy: si esto falla, no rompe la corrida.
    try:
        from .domain.sample_extraction import plan_speaker_samples
        from .services.extract_samples import extract_speaker_samples
        from .services.transcript_builder import (
            build_transcript_from_legacy_segments,
        )

        annotation_turns = [
            (turn.start, turn.end, tag)
            for turn, _, tag in annotation.itertracks(yield_label=True)
        ]
        transcript = build_transcript_from_legacy_segments(
            legacy_segments=common_segments,
            annotation_turns=annotation_turns,
            speaker_map=speaker_map,
            audio_path=str(state.working_path),
            language=language,
        )
        transcript.save(state.artifacts_dir / "transcript.json")

        plans = plan_speaker_samples(
            transcript,
            max_clips_per_speaker=5,
            min_clip_duration_ms=2000,
        )
        if plans:
            extract_speaker_samples(
                plans=plans,
                audio_path=state.working_path,
                output_dir=state.artifacts_dir / "samples",
            )
    except Exception as exc:
        print(f"WARNING: domain transcript publish failed ({exc}). Legacy output OK.")

    # Wait for background compression to finish
    if compress_thread is not None:
        compress_thread.join()

    print_report()
    kprint_report()
    return common_segments
