import logging
import os
import json
import shutil
import threading
import time

logger = logging.getLogger(__name__)

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
    apply_speaker_map_to_transcript,
    average_embeddings,
    build_score_matrix,
    build_suggestions,
    filter_voice_library,
    select_segments_for_embedding,
)
from .domain.transcript import Transcript
from .services.transcript_builder import apply_speaker_map_to_segments

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
    """Publica transcript.json y samples en artifacts_dir. No afecta el VTT."""
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

        transcript = build_transcript_from_legacy_segments(
            legacy_segments=common_segments,
            annotation_turns=annotation_turns,
            speaker_map=speaker_map,
            audio_path=str(state.working_path),
            language=language,
        )
        transcript.save(state.artifacts_dir / "transcript.json")

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
    initial_prompt: str | None = None,
) -> list:
    """Transcribe segmentos del audio. Soporta faster-whisper (alineado) y
    el path generico (segmentacion por speaker).

    """
    with measure("transcription", gpu=True):
        if model_type == "faster-whisper":
            return transcribe_full_aligned(
                str(state.working_path), common, language, model_size, quantization,
                initial_prompt=initial_prompt,
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
    """Para cada SPEAKER_XX, computa el embedding promedio a partir de chunks."""
    import numpy as np

    inference = _get_inference()
    working_path = Path(state.working_path)
    audio_stem = working_path.stem
    tmp_dir = working_path.parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    embeddings_by_tag: dict = {}
    with measure("speaker_embeddings", gpu=True):
        for spk_tag, spk_segments in speakers.items():
            selected = select_segments_for_embedding(
                spk_segments,
                limit_s=limit_s,
                min_segment_s=MIN_SEGMENT_DURATION_S,
            )
            per_chunk_embeddings: list = []
            for i, segment in enumerate(selected):
                start_ms = segment[0] * 1000
                end_ms = segment[1] * 1000
                chunk = str(tmp_dir / f"{audio_stem}_{spk_tag}_chunk_{i}.wav")
                try:
                    slice_and_save(str(state.working_path), start_ms, end_ms, chunk)
                    per_chunk_embeddings.append(np.asarray(inference(chunk)))
                except Exception as exc:
                    logger.debug("Error extracting embedding from segment: %s", exc)
                finally:
                    Path(chunk).unlink(missing_ok=True)
            averaged = average_embeddings(per_chunk_embeddings)
            if averaged is not None:
                embeddings_by_tag[spk_tag] = averaged
    return embeddings_by_tag


def _resolve_voice_library(
    voices_folder: str,
    enhanced: bool,
    allowed_speakers: list[str] | None,
) -> tuple[dict, list[str]]:
    """Carga voice library y separa speakers con/sin sample. I/O boundary."""
    full_library = load_avg_voice_embeddings(Path(voices_folder), enhanced=enhanced)
    if allowed_speakers is not None:
        with_sample = set(allowed_speakers) & set(full_library.keys())
        without_sample = [s for s in allowed_speakers if s not in full_library]
        return filter_voice_library(full_library, allowed_names=with_sample), without_sample
    return full_library, []


def _run_speaker_recognition_cached(
    state: AudioState,
    voices_folder: str,
    speakers: dict,
    speaker_tags: list,
    *,
    allowed_speakers: list[str] | None = None,
) -> dict:
    """Computa suggestions + diagnostics con cache. NO decide speaker_map.

    Retorna el dict de suggestions (formato build_suggestions). La aplicación
    de nombres reales queda al subcomando `confirm`, que lee un speaker_map.json
    escrito por el usuario.
    """
    suggestions_path = state.artifacts_dir / "speaker_map_suggestions.json"
    params_path = state.artifacts_dir / "speaker_map_params.json"

    current_params = {
        "allowed_speakers": sorted(allowed_speakers) if allowed_speakers else None,
        "threshold": SPEAKER_SIMILARITY_THRESHOLD,
        "min_margin": SPEAKER_SIMILARITY_MIN_MARGIN,
    }

    # Cache hit: suggestions + params coinciden
    if suggestions_path.exists() and params_path.exists():
        saved_params = json.loads(params_path.read_text(encoding="utf-8"))
        if saved_params == current_params:
            return json.loads(suggestions_path.read_text(encoding="utf-8"))
        suggestions_path.unlink()

    voice_library, _without_sample = _resolve_voice_library(
        voices_folder, state.is_enhanced, allowed_speakers
    )
    embeddings_by_tag = _compute_averaged_embeddings_per_tag(state, speakers)

    suggestions = build_suggestions(
        embeddings_by_tag, voice_library,
        threshold=SPEAKER_SIMILARITY_THRESHOLD,
        min_margin=SPEAKER_SIMILARITY_MIN_MARGIN,
    )
    suggestions_path.write_text(
        json.dumps(suggestions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    params_path.write_text(
        json.dumps(current_params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    diagnostics = build_score_matrix(
        embeddings_by_tag, voice_library,
        threshold=SPEAKER_SIMILARITY_THRESHOLD,
        min_margin=SPEAKER_SIMILARITY_MIN_MARGIN,
    )
    diag_path = state.artifacts_dir / "recognition_diagnostics.json"
    diag_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return suggestions


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


def _run_diarization_cached(state: AudioState, access_token: str | None, *, num_speakers: int | None = None):
    """Devuelve la annotation pyannote, reutilizando diarization.rttm si existe.

    Recompute si el cache existe pero esta corrupto. Escribe el RTTM cuando
    recomputa.
    """
    rttm_path = state.artifacts_dir / "diarization.rttm"

    if rttm_path.exists():
        try:
            return next(iter(_load_rttm(str(rttm_path)).values())), True
        except Exception as e:
            logger.warning("could not load diarization.rttm (%s), recomputing.", e)
            rttm_path.unlink(missing_ok=True)

    pipeline = _get_diarization_pipeline(access_token)
    waveform, sample_rate = torchaudio.load(str(state.working_path))
    with measure("diarization", gpu=True), kmeasure("diarization"):
        pipeline_kwargs = {}
        if num_speakers is not None:
            pipeline_kwargs["num_speakers"] = num_speakers
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **pipeline_kwargs)
    annotation = (
        diarization.speaker_diarization
        if hasattr(diarization, "speaker_diarization")
        else diarization
    )
    with open(rttm_path, "w") as f:
        annotation.write_rttm(f)
    return annotation, False


def _resolve_working_path_from_cache(state: AudioState) -> AudioState:
    """Apunta working_path al audio procesado del cache (enhanced/16k)."""
    for candidate in ("enhanced.wav", "16k.wav"):
        p = state.artifacts_dir / candidate
        if p.exists():
            return state.model_copy(update={"working_path": p, "is_wav": True})
    return state


def run_recognition(
    file_name: str,
    voices_folder: str,
    *,
    allowed_speakers: list[str] | None = None,
    force: bool = False,
) -> dict:
    """Re-ejecuta solo speaker recognition sobre RTTM existente."""
    state = AudioState(source_path=Path(file_name), working_path=Path(file_name))
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Cargar RTTM
    rttm_path = state.artifacts_dir / "diarization.rttm"
    if not rttm_path.exists():
        raise FileNotFoundError(f"No diarization.rttm in {state.artifacts_dir}. Run full pipeline first.")
    annotation = next(iter(_load_rttm(str(rttm_path)).values()))

    state = _resolve_working_path_from_cache(state)

    if force:
        (state.artifacts_dir / "speaker_map_suggestions.json").unlink(missing_ok=True)
        (state.artifacts_dir / "speaker_map_params.json").unlink(missing_ok=True)

    _, speakers, speaker_tags, _ = _build_speaker_groups(annotation)

    return _run_speaker_recognition_cached(
        state, voices_folder, speakers, speaker_tags,
        allowed_speakers=allowed_speakers,
    )


def run_confirm(file_name: str) -> dict:
    """Aplica speaker_map.json (editado por usuario) al VTT publicado.

    Lee del cache:
    - speaker_map.json: dict {SPEAKER_XX: nombre} escrito por el usuario
    - transcript.json: aggregate Transcript con segments del run previo

    Regenera el VTT en el cache y publica una copia con sufijo _limpio
    al lado del audio original. Tags ausentes del map (o mapeados a sí
    mismos) quedan como [SPEAKER_XX] literal en el VTT.

    Retorna el speaker_map aplicado para que el CLI lo imprima.
    """
    state = AudioState(source_path=Path(file_name), working_path=Path(file_name))
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)
    state = _resolve_working_path_from_cache(state)

    map_path = state.artifacts_dir / "speaker_map.json"
    if not map_path.exists():
        raise FileNotFoundError(
            f"No speaker_map.json en {state.artifacts_dir}. "
            "Ejecutá `speechlib run` primero, revisá speaker_map_suggestions.json, "
            "y escribí tu propio speaker_map.json con {SPEAKER_XX: 'nombre real'}."
        )

    transcript_path = state.artifacts_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"No transcript.json en {state.artifacts_dir}. "
            "Ejecutá `speechlib run` primero."
        )

    speaker_map = json.loads(map_path.read_text(encoding="utf-8"))
    transcript = Transcript.load(transcript_path)
    relabeled = apply_speaker_map_to_transcript(transcript, speaker_map)

    common_segments = [
        [seg.start_ms / 1000, seg.end_ms / 1000, seg.text, seg.speaker.label]
        for seg in relabeled.segments
    ]
    write_log_file(
        common_segments,
        None,
        str(state.working_path),
        relabeled.language,
        "vtt",
    )
    relabeled.save(transcript_path)
    _publish_to_source_folder(state, relabeled.language, "vtt")

    return speaker_map


def run_diagnose(
    file_name: str,
    voices_folder: str,
    *,
    allowed_speakers: list[str] | None = None,
) -> dict:
    """Genera matriz de scores de recognition sin modificar artifacts."""
    state = AudioState(source_path=Path(file_name), working_path=Path(file_name))
    state.artifacts_dir.mkdir(parents=True, exist_ok=True)

    rttm_path = state.artifacts_dir / "diarization.rttm"
    if not rttm_path.exists():
        raise FileNotFoundError(f"No diarization.rttm in {state.artifacts_dir}. Run full pipeline first.")
    annotation = next(iter(_load_rttm(str(rttm_path)).values()))

    state = _resolve_working_path_from_cache(state)

    _, speakers, speaker_tags, _ = _build_speaker_groups(annotation)

    voice_library, without_sample = _resolve_voice_library(
        voices_folder, state.is_enhanced, allowed_speakers
    )
    embeddings_by_tag = _compute_averaged_embeddings_per_tag(state, speakers)

    return build_score_matrix(
        embeddings_by_tag, voice_library,
        threshold=SPEAKER_SIMILARITY_THRESHOLD,
        min_margin=SPEAKER_SIMILARITY_MIN_MARGIN,
    )


def _publish_to_source_folder(state: AudioState, language: str, output_format: str) -> None:
    """Copia outputs finales al source folder con naming _limpio."""
    source_dir = state.source_path.parent
    stem = state.source_path.stem.strip()
    ext = "vtt" if output_format == "vtt" else "txt"

    transcript_src = state.artifacts_dir / f"transcript_{language}.{ext}"
    if transcript_src.exists():
        shutil.copy2(transcript_src, source_dir / f"{stem}_limpio.{ext}")


def _preprocess_audio(file_name: str) -> AudioState:
    """Preprocessing hasta loudnorm (sin enhance). Reusa cache 16k.wav."""
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
    return state


def _start_compress_thread(state: AudioState) -> threading.Thread:
    """Lanza compresion AAC en background thread."""
    limpio_path = state.source_path.parent / f"{state.source_path.stem.strip()}_limpio.m4a"
    t = threading.Thread(target=compress_audio, args=(state.working_path, limpio_path), daemon=True)
    t.start()
    return t


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
    allowed_speakers: list[str] | None = None,
    initial_prompt: str | None = None,
):
    import torch
    from rich.console import Console
    console = Console()

    if not torch.cuda.is_available():
        console.print("[yellow]WARN: CUDA no disponible — CPU mode (lento). Activaste mamba?[/]")

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    console.print(f"[dim]speechlib | {device} | {Path(file_name).name}[/]")

    if log_folder is None:
        log_folder = os.path.join(os.path.dirname(os.path.abspath(file_name)), "output")

    with console.status("Preprocessing..."):
        state_loudnorm = _preprocess_audio(file_name)

    # Compress en paralelo con diarize cuando no hay enhance (GPU vs CPU,
    # sin contención). Enhance necesita GPU -> se serializa más abajo.
    compress_thread = None
    if compress and skip_enhance:
        compress_thread = _start_compress_thread(state_loudnorm)

    num_speakers = len(allowed_speakers) if allowed_speakers else None

    with console.status("Diarization..."):
        annotation, from_cache = _run_diarization_cached(
            state_loudnorm, ACCESS_TOKEN, num_speakers=num_speakers
        )
    console.print(f"[green]OK[/] Diarization {'(cache)' if from_cache else 'done'}")

    common, speakers, speaker_tags, speaker_map = _build_speaker_groups(annotation)

    has_voices_folder = voices_folder is not None and voices_folder != ""
    if has_voices_folder:
        with console.status("Speaker suggestions..."):
            suggestions = _run_speaker_recognition_cached(
                state_loudnorm, voices_folder, speakers, speaker_tags,
                allowed_speakers=allowed_speakers,
            )
        n_recommended = sum(
            1 for s in suggestions.get("tags", {}).values()
            if s.get("recommended") is not None
        )
        total = len(suggestions.get("tags", {}))
        console.print(
            f"[green]OK[/] Speaker suggestions — {n_recommended}/{total} "
            f"recommended (user must confirm via `speechlib confirm`)"
        )

    # speaker_map sigue siendo el identity map del diarization (tag->tag).
    # Aplicarlo es no-op sobre los segments, pero preserva el shape del flujo
    # legacy. La asignación de nombres reales ocurre en el subcomando confirm.
    common = apply_speaker_map_to_segments(common, speaker_map)
    common = absorb_micro_segments(common)
    common = merge_short_turns(common)

    with console.status("Transcription..."):
        common_segments = _transcribe_segments(
            state_loudnorm, common, speaker_map,
            language=language,
            model_size=modelSize,
            model_type=model_type,
            quantization=quantization,
            custom_model_path=custom_model_path,
            hf_model_id=hf_model_id,
            aai_api_key=aai_api_key,
            initial_prompt=initial_prompt,
        )
    console.print("[green]OK[/] Transcription done")

    common_segments = _group_post_transcription(
        common_segments, model_type=model_type, grouping_mode=grouping_mode
    )

    with console.status("Writing output..."):
        with measure("write_log_file"):
            write_log_file(
                common_segments,
                log_folder,
                str(state_loudnorm.working_path),
                language,
                output_format,
            )
        with measure("publish_artifacts"):
            _publish_domain_artifacts(common_segments, annotation, speaker_map, state_loudnorm, language)

        # Enhance + compress serial (ambos GPU, no paraleliza con diarize).
        # El path skip_enhance ya arrancó como thread después del preprocess.
        if compress and not skip_enhance:
            with console.status("Enhance + compress..."):
                state_enhanced = enhance_audio(state_loudnorm)
                compress_audio(state_enhanced.working_path,
                    state_loudnorm.source_path.parent / f"{state_loudnorm.source_path.stem.strip()}_limpio.m4a")
        elif compress_thread is not None:
            compress_thread.join()

        _publish_to_source_folder(state_loudnorm, language, output_format)

    console.print("[green]OK[/] Output written")

    print_report()
    kprint_report()
    return common_segments
