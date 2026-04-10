# Pipeline de Proceso y Modelo de Profiling

**Fecha:** 2026-04-10
**Hardware:** NVIDIA RTX 2070 Super (8 GB VRAM)
**Benchmark:** obama_zach.wav (~6 min), large-v3-turbo, community-1

---

## Diagrama de proceso

```
                    SPEECHLIB PIPELINE
                    ==================

 STAGE 1: PREPROCESSING (secuencial, cache en 16k.wav)
 ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
 │ convert_wav │→ │  mono    │→ │re_encode │→ │resample  │→ │ loudnorm │
 │   (I/O)     │  │  (CPU)   │  │  (CPU)   │  │  16kHz   │  │ EBU R128 │
 └─────────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
                                                  @timed       @timed
                  ── si 16k.wav existe, salta a loudnorm ──
                                                              │
                              ┌────────────────────────────────┤
                              ▼                                ▼
 STAGE 2: COMPRESS       STAGE 3: ENHANCE               STAGE 4: DIARIZE
 (background thread)     (condicional)                   (usa audio pre-enhance)
 ┌──────────────┐        ┌──────────────┐               ┌──────────────┐
 │compress_audio│        │ MossFormer2  │               │   pyannote   │
 │  → _limpio   │        │   SE_48K     │               │ community-1  │
 │    .m4a      │        │   (GPU)      │               │   (GPU)      │
 └──────────────┘        └──────────────┘               └──────────────┘
    @timed                @timed + @ktimed               @measure + @kmeasure
    daemon=True           cache: enhanced.wav            cache: diarization.rttm
                              │                              │
                              ▼                              ▼
                         state (enhanced           annotation (pyannote)
                          o loudnorm)                    │
                              │                          ▼
                              │              ┌──────────────────┐
                              │              │ _build_speaker   │
                              │              │   _groups()      │
                              │              │   (CPU puro)     │
                              │              └──────────────────┘
                              │                    │
                              ▼                    ▼
                   STAGE 5: SPEAKER RECOGNITION (condicional, cache en speaker_map.json)
                   ┌─────────────────────────────────────────────────────┐
                   │ _resolve_voice_library()     carga + filtra library │
                   │ _compute_avg_embeddings()    GPU inference × N tags │
                   │ _build_speaker_map()         matching puro (CPU)    │
                   │ assign_extra_speakers()      nombres sin sample     │
                   └─────────────────────────────────────────────────────┘
                     SIN PROFILING ← brecha de observabilidad
                                          │
                                          ▼
                   STAGE 6: SEGMENT MERGING (CPU puro, <0.1s)
                   ┌──────────────────────────────────────────┐
                   │ apply_speaker_map → absorb_micro → merge │
                   └──────────────────────────────────────────┘
                                          │
                                          ▼
                   STAGE 7: TRANSCRIPTION (GPU)
                   ┌──────────────────────────────────────────┐
                   │ faster-whisper BatchedInferencePipeline   │
                   │ batch_size=4, word_timestamps=True        │
                   │ mapeo word→segmento por overlap temporal  │
                   └──────────────────────────────────────────┘
                     @measure("transcription", gpu=True)
                                          │
                                          ▼
                   STAGE 8: POST-GROUPING (CPU, <0.1s)
                   ┌──────────────────────────────────────────┐
                   │ group_by_sentences() o group_by_speaker()│
                   └──────────────────────────────────────────┘
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                   STAGE 9: DOMAIN          STAGE 10: VTT/TXT
                   ┌──────────────┐         ┌──────────────┐
                   │transcript.json│         │transcript_   │
                   │  + samples/   │         │  {lang}.vtt  │
                   └──────────────┘         └──────────────┘
                     SIN PROFILING            @measure
                              │                       │
                              └───────────┬───────────┘
                                          ▼
                   STAGE 11: PUBLISH
                   ┌──────────────────────────────────────────┐
                   │ copy → {stem}_limpio.vtt + join compress │
                   └──────────────────────────────────────────┘
```

---

## Benchmark medido (2026-04-10, con enhance + speaker recognition)

| Step | Tiempo | % Total | Recurso | VRAM delta |
|------|--------|---------|---------|------------|
| convert_to_wav | 0.000s | 0% | skip | - |
| convert_to_mono | 0.000s | 0% | skip | - |
| re_encode | 0.000s | 0% | skip | - |
| resample_to_16k | 0.290s | 0.3% | CPU | - |
| loudnorm | 0.674s | 0.8% | CPU | - |
| **enhance_audio** | **51.6s** | **61.7%** | GPU | +221 MB |
| **diarization** | **13.7s** | **16.4%** | GPU | +0 MB |
| **speaker_embeddings** | **2.5s** | **3.0%** | GPU | +0 MB |
| **transcription** | **13.9s** | **16.7%** | GPU | +0 MB |
| write_log_file | 0.001s | 0% | I/O | - |
| publish_artifacts | 0.962s | 1.2% | I/O | - |
| **TOTAL** | **~83.6s** | | | 270 MB peak |

Sin enhance: **~27s** total. Todos los pasos medidos — zero brechas.

---

## Brechas de observabilidad

Todas las brechas criticas y menores fueron cerradas (2026-04-10):
- `convert_to_wav`, `convert_to_mono`, `re_encode`: @timed agregado
- `speaker_embeddings`: measure() con gpu=True (2.5s medidos)
- `publish_artifacts`: measure() agregado (0.96s medidos)

### Pendiente (no critico)
| Paso | Observacion |
|------|------------|
| `transcribe_full_aligned` internals | No distingue carga de modelo vs inference vs word mapping. Sub-timers agregarían overhead. |

---

## Modelo de profiling propuesto

### Nivel 1: Step Timer (actual, SPEECHLIB_PROFILE=1)

Lo que ya mide:
```
resample_to_16k, loudnorm, enhance_audio, diarization, transcription, 
write_log_file, compress_audio
```

Agregar:
```
convert_to_wav, convert_to_mono, re_encode, speaker_embeddings, 
speaker_matching, publish_artifacts
```

### Nivel 2: Kernel Profiler (actual, SPEECHLIB_PROFILE_KERNELS=1)

Lo que ya mide: `enhance_audio, diarization`

Agregar: `speaker_embeddings, transcription`

### Nivel 3: Pipeline Summary (nuevo)

Al final del pipeline, imprimir resumen ejecutivo:

```
=== Pipeline Summary ===
Audio: Voice_260127.m4a (3h 30m)
Device: RTX 2070 Super
Steps:  preprocessing 0.8s | enhance 44.7s | diarize 12.4s (cache) |
        recognition 3.1s | transcribe 11.5s | output 0.1s
Total:  72.8s (RTF: 0.006x)
Speakers: 6 detected, 4 identified, 2 extra-assigned
Cache:  16k.wav HIT | enhanced.wav HIT | rttm HIT | speaker_map MISS
```

### Nivel 4: Trace Export (actual, SPEECHLIB_PROFILE_KERNELS=1)

Chrome trace JSON en `./profiling_traces/` — visualizable en perfetto.dev.

---

## Oportunidades de optimización identificadas

| Oportunidad | Ahorro estimado | Validado | Estado |
|---|---|---|---|
| batch_size 4→16 | -3s en audio corto, +? en largo | SI, **regresión** en audio corto | Descartado |
| Overlap enhance+diarize | -10s teórico | SI, **solo 3.6%** por contención GPU | Descartado |
| Skip enhance (--skip-enhance) | -44.7s (61% del pipeline) | SI | Disponible |
| Cache 16k.wav | -0.8s en re-runs | SI | Implementado |
| Cache diarization.rttm | -12.4s en re-runs | SI | Implementado |
| Cache speaker_map.json | -3s en re-runs | SI | Implementado |
| Cache enhanced.wav | -44.7s en re-runs | SI | Implementado |
| Batch embeddings per tag | -1s (reduce GPU calls) | NO | Pendiente |
| Profiling completo (nivel 1) | 0 (observabilidad) | NO | **Recomendado** |

---

## Comandos de profiling

```bash
# Nivel 1: tiempos por paso + VRAM
SPEECHLIB_PROFILE=1 python -m speechlib audio.m4a -v

# Nivel 2: breakdown CPU vs CUDA kernels (10x overhead)
SPEECHLIB_PROFILE=1 SPEECHLIB_PROFILE_KERNELS=1 python -m speechlib audio.m4a -v

# Benchmark reproducible
python benchmark_pipeline.py --audio examples/obama_zach.wav
python benchmark_pipeline.py --audio examples/obama_zach.wav --skip-enhance
```
