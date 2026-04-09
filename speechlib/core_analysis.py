import logging
import os
import json
import threading
import time

logger = logging.getLogger(__name__)

# Smell 7: monkey-patch de torchaudio aislado en speechlib/compat.py.
# Importar compat ANTES de cualquier modulo que use torchaudio (pyannote, etc.)
from . import compat  # noqa: F401  side-effect: patches torchaudio

from .wav_segmenter import wav_file_segmentation
from .transcribe import transcribe_full_aligned
from .step_timer import measure, print_report
from .kernel_profiler import measure as kmeasure, print_report as kprint_report

import torchaudio

from .diarization import get_diarization_pipeline as _get_diarization_pipeline
from .speaker_recognition import (
    MIN_SEGMENT_DURATION_S,
    SPEAKER_SIMILARITY_MIN_MARGIN,
    SPEAKER_SIMILARITY_THRESHOLD,
    _get_inference,
    load_avg_voice_embeddings,
)
from .audio_utils import slice_and_save
from .domain.recognition import (
    assign_speakers,
    average_embeddings,
    select_segments_for_embedding,
)
from .services.transcript_builder import apply_speaker_map_to_segments
from .domain.transcript import (
    SpeakerIdentity,
    Transcript,
    TranscriptSegment,
)

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


def _group_post_transcription(common_segments: list, *, model_type: str, grouping_mode: str) -> list:
    """Aplica grouping post-transcripcion solo cuando model_type=faster-whisper."""
    if model_type != "faster-whisper":
        return common_segments
    if grouping_mode == "sentences":
        return group_by_sentences(common_segments)
    return group_by_speaker(common_segments)


def _publish_domain_artifacts(
    common_segments: list,
    annotation,
    speaker_map: dict,
    state: AudioState,
    language: str,
) -> None:
    """Publica el aggregate del nuevo dominio en paralelo al output legacy.

    transcript.json es el formato canonico futuro; el VTT queda como render.
    Cero impacto sobre el output legacy: si esto falla, no rompe la corrida.
    """
    try:
        from .domain.sample_extraction import plan_speaker_samples
        from .services.extract_samples import extract_speaker_samples
        from .services.transcript_builder import (
            build_transcript_from_annotation_turns,
            build_transcript_from_legacy_segments,
        )

        annotation_turns = [
            (turn.start, turn.end, tag)
            for turn, _, tag in annotation.itertracks(yield_label=True)
        ]

        # Transcript readable: usa los segmentos post-merge/post-grouping
        # del legacy. Buenos para texto, NO para sample extraction porque
        # pueden ser de 50+ segundos con crosstalk.
        transcript = build_transcript_from_legacy_segments(
            legacy_segments=common_segments,
            annotation_turns=annotation_turns,
            speaker_map=speaker_map,
            audio_path=str(state.working_path),
            language=language,
        )
        transcript.save(state.artifacts_dir / "transcript.json")

        # Slice 18: Transcript de muestreo construido desde los turnos RAW
        # del RTTM (cada turno = un segmento). Single-speaker garantizado
        # por construccion de pyannote diarization. Threshold mas bajo
        # porque los turnos raw son tipicamente cortos (~0.5-3s).
        sample_transcript = build_transcript_from_annotation_turns(
            annotation_turns=annotation_turns,
            speaker_map=speaker_map,
            audio_path=str(state.working_path),
            language=language,
        )
        plans = plan_speaker_samples(
            sample_transcript,
            max_clips_per_speaker=5,
            min_clip_duration_ms=1500,
            min_unidentified_clip_duration_ms=500,
        )
        if plans:
            extract_speaker_samples(
                plans=plans,
                audio_path=state.working_path,
                output_dir=state.artifacts_dir / "samples",
            )
    except Exception:
        # Smell 3: log full traceback (no swallow silencioso). El legacy output
        # sigue funcionando, pero ahora podemos diagnosticar regresiones de la
        # nueva feature de Slice 5 con stack frames + line numbers.
        logger.exception(
            "domain transcript publish failed; legacy output continues unaffected"
        )


def _transcribe_segments(
    state: AudioState,
    common: list,
    speaker_map: dict,
    *,
    language: str,
    model_size: str,
    model_type: str,
    quantization: bool,
    custom_model_path,
    hf_model_id,
    aai_api_key,
) -> list:
    """Transcribe segmentos del audio. Soporta faster-whisper (alineado) y
    el path generico (segmentacion por speaker).

    Smell 2: para el path faster-whisper (default) NO necesitamos el dict
    `speakers` — solo `common`. Para el path generico, lo regeneramos
    in-line (era _regroup_speakers_from_common, ahora inlined porque solo
    se usa aqui) y eliminamos el helper top-level.
    """
    print("running transcription...")
    with measure("transcription", gpu=True):
        if model_type == "faster-whisper":
            return transcribe_full_aligned(
                str(state.working_path), common, language, model_size, quantization
            )

        # Path generico: regenerar speakers desde common (post absorb/merge)
        speakers: dict = {}
        for segment in common:
            spk = segment[2]
            speakers.setdefault(spk, []).append(segment)

        for spk_tag, spk_segments in speakers.items():
            speakers[spk_tag] = wav_file_segmentation(
                str(state.working_path),
                spk_segments,
                language,
                model_size,
                model_type,
                quantization,
                custom_model_path,
                hf_model_id,
                aai_api_key,
            )

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
    return common_segments


def _compute_averaged_embeddings_per_tag(
    state: AudioState,
    speakers: dict,
    *,
    limit_s: float = 60.0,
):
    """Para cada SPEAKER_XX, computa el embedding promedio a partir de chunks.

    Application service: orquesta el I/O (slice del audio + inference).
    La logica de aggregation (promedio + filtro NaN) vive en la funcion
    pura speechlib.domain.recognition.average_embeddings (Smell 6 fix).
    """
    import numpy as np

    inference = _get_inference()
    folder_name = str(Path(state.working_path).parent / "tmp")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    embeddings_by_tag: dict = {}
    for spk_tag, spk_segments in speakers.items():
        # Bug fix Pamela: seleccion por duracion descendente, no por orden
        # de documento. Funcion pura del dominio (select_segments_for_embedding).
        selected = select_segments_for_embedding(
            spk_segments,
            limit_s=limit_s,
            min_segment_s=MIN_SEGMENT_DURATION_S,
        )
        per_chunk_embeddings: list = []
        for i, segment in enumerate(selected):
            start_ms = segment[0] * 1000
            end_ms = segment[1] * 1000
            chunk = (
                folder_name + "/"
                + os.path.splitext(os.path.basename(str(state.working_path)))[0]
                + f"_{spk_tag}_chunk_{i}.wav"
            )
            try:
                slice_and_save(str(state.working_path), start_ms, end_ms, chunk)
                per_chunk_embeddings.append(np.asarray(inference(chunk)))
            except Exception as exc:
                print(f"Error extracting embedding from segment: {exc}")
            finally:
                try:
                    os.remove(chunk)
                except OSError:
                    pass
        averaged = average_embeddings(per_chunk_embeddings)
        if averaged is not None:
            embeddings_by_tag[spk_tag] = averaged
    return embeddings_by_tag


def _run_speaker_recognition_cached(
    state: AudioState,
    voices_folder: str,
    speakers: dict,
    speaker_tags: list,
) -> dict:
    """Identifica cada SPEAKER_XX contra la libreria de voces, con cache.

    Si artifacts_dir/speaker_map.json existe, lo carga. Si no, computa
    embeddings por tag y delega en assign_speakers (dominio puro). El
    resultado se serializa al formato legacy {tag: name_or_tag}.

    Slice 13b: elimino el hack historico de "unknown" -> tag. Ahora la
    transformacion la hace SpeakerIdentity.label por construccion del
    dominio: si recognized_name es None, label cae al diarization_tag.
    Es estructuralmente imposible que el speaker_map.json contenga el
    literal "unknown".

    Mantiene comportamiento observable identico para los consumidores
    legacy de speaker_map.json.
    """
    speaker_map_path = state.artifacts_dir / "speaker_map.json"

    if speaker_map_path.exists():
        speaker_map = json.loads(speaker_map_path.read_text(encoding="utf-8"))
        print("speaker_map loaded from cache.")
        return speaker_map

    start_time = int(time.time())
    print("running speaker recognition...")

    voice_library = load_avg_voice_embeddings(
        Path(voices_folder), enhanced=state.is_enhanced
    )
    embeddings_by_tag = _compute_averaged_embeddings_per_tag(state, speakers)

    transcript = Transcript(
        segments=tuple(
            TranscriptSegment(
                start_ms=int(speakers[tag][0][0] * 1000),
                end_ms=int(speakers[tag][0][1] * 1000),
                text="",
                speaker=SpeakerIdentity(diarization_tag=tag),
            )
            for tag in speaker_tags
            if tag in embeddings_by_tag
        ),
        audio_path=str(state.working_path),
        language="",
    )

    relabeled = assign_speakers(
        transcript,
        embeddings_by_tag,
        voice_library,
        threshold=SPEAKER_SIMILARITY_THRESHOLD,
        min_margin=SPEAKER_SIMILARITY_MIN_MARGIN,
    )

    elapsed = int(time.time() - start_time)
    print(f"speaker recognition done. Time taken: {elapsed} seconds.")

    # Construir speaker_map legacy desde el aggregate del dominio.
    # SpeakerIdentity.label devuelve recognized_name OR diarization_tag,
    # NUNCA el literal "unknown" — el bug es estructuralmente imposible.
    speaker_map: dict = {
        seg.speaker.diarization_tag: seg.speaker.label
        for seg in relabeled.segments
    }
    # Tags sin embedding (turnos demasiado cortos): preservar como SPEAKER_XX
    for spk_tag in speaker_tags:
        if spk_tag not in speaker_map:
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

    # Smell 1: reemplaza _merge_same_speakers (legacy mutating helper) con
    # apply_speaker_map_to_segments (funcion pura). Las mutaciones de
    # `speakers` y `speaker_map` que el legacy hacia eran DEAD CODE — speakers
    # se regenera 2 lineas abajo via _regroup_speakers_from_common, y
    # speaker_map se usa solo via .get(name, name) que nunca toca las claves
    # borradas.
    common = apply_speaker_map_to_segments(common, speaker_map)

    # absorb micro-segments into longer neighbors, then merge same-speaker turns
    common = absorb_micro_segments(common)
    common = merge_short_turns(common)

    # Smell 2: dejamos de regenerar `speakers` aqui — _transcribe_segments lo
    # regenera in-line solo cuando el path generico (non-faster-whisper) lo
    # necesita. faster-whisper (default) opera solo sobre `common`.
    common_segments = _transcribe_segments(
        state, common, speaker_map,
        language=language,
        model_size=modelSize,
        model_type=model_type,
        quantization=quantization,
        custom_model_path=custom_model_path,
        hf_model_id=hf_model_id,
        aai_api_key=aai_api_key,
    )
    print("transcription done.")

    common_segments = _group_post_transcription(
        common_segments, model_type=model_type, grouping_mode=grouping_mode
    )

    # writing log file
    with measure("write_log_file"):
        write_log_file(
            common_segments,
            log_folder,
            str(state.working_path),
            language,
            output_format,
        )

    _publish_domain_artifacts(common_segments, annotation, speaker_map, state, language)

    # Wait for background compression to finish
    if compress_thread is not None:
        compress_thread.join()

    print_report()
    kprint_report()
    return common_segments
