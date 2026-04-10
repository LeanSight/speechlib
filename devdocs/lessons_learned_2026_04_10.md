# Lecciones Aprendidas — Sesión 2026-04-10

## Qué es speechlib

speechlib es una librería y CLI Python para transcripción de audio con diarización (quién habla cuándo) y reconocimiento de speakers (asignar nombres). Usa pyannote (diarización), faster-whisper (transcripción), y pyannote/embedding (speaker recognition por cosine similarity). Repo: `c:\workspace\dev\speechlib`, branch `refactor/speaker-domain`.

## Glosario

- **WER** (Word Error Rate): porcentaje de palabras incorrectas vs referencia humana.
- **Diarización**: segmentar audio por "quién habla cuándo" (produce SPEAKER_00, SPEAKER_01, etc.)
- **Speaker recognition**: matchear SPEAKER_XX contra voice samples conocidos por cosine similarity de embeddings.
- **MossFormer2_SE_48K**: modelo de speech enhancement de ClearVoice (GitHub: ClearerVoice-Studio). Reduce ruido pero distorsiona embeddings.
- **AT** (Acceptance Test): test que verifica comportamiento observable desde afuera del sistema (como lo ve el usuario). Se escribe en RED antes de implementar.
- **GPU boundaries (en tests)**: el único código mockeado en tests son las funciones que cargan modelos GPU (pyannote Pipeline, WhisperModel, embedding inference). Todo lo demás se testea real. Regla: máximo 2 patches por archivo de test.
- **min_margin**: diferencia mínima entre el score del top-1 y top-2 match para aceptar una identificación. Si top1=0.60 y top2=0.55, margin=0.05 < min_margin(0.10) → match rechazado como ambiguo.
- **--compress**: genera un archivo `{stem}_limpio.m4a` (AAC mono 96kbps 16kHz) en la carpeta del audio original. Si enhance está activo, el audio comprimido se enhancee primero (para escucha humana).
- **Shims (compat.py)**: monkey-patches para incompatibilidades de librerías. Actualmente: (1) `torchaudio.list_audio_backends` removido en torchaudio 2.x pero requerido por SpeechBrain 1.0.3, (2) `torchaudio.load/save` reemplazado por PyAV cuando torchcodec falla en Windows.

## Intención de la sesión

Refactorear speechlib para resolver el problema de **false positives en speaker recognition** y optimizar el pipeline de transcripción. El usuario provee grabaciones de reuniones con N asistentes (algunos con voice samples, otros sin) y necesita que speechlib identifique correctamente quién habla.

## Qué funcionó

### 1. Constrained speaker recognition (closed-set verification)
- **Problema**: speechlib matcheaba contra toda la voice library (19 speakers). Speakers ausentes (Orlando, Marcos, Pamela) matcheaban con scores altos contra voces presentes sin sample.
- **Solución**: `--speakers "A,B,C,D,E,F"` filtra la library a solo los asistentes esperados con sample, y asigna nombres sin sample a tags sobrantes por cantidad de segmentos.
- **Resultado**: eliminó false positives de speakers ausentes.
- **Asignación por descarte**: los tags SPEAKER_XX no matcheados se ordenan por cantidad de segmentos (el que más habla primero) y reciben los nombres sin sample en el orden en que aparecen en --speakers. Si hay más tags no matcheados que nombres sin sample, los sobrantes quedan como SPEAKER_XX. Si hay más nombres que tags, los nombres extra se ignoran.

### 2. A/B test de enhancement
- **Hallazgo**: speech enhancement (MossFormer2) perjudica ASR. WER sube +1.6pp a +6.2pp, speaker accuracy baja de 59% a 36%.
- **Validación**: A/B test con 2 audios de BCI Seguros (5 min y 15 min) contra SRT de referencia validados por el usuario. Archivos en `C:\workspace\@recordings\20260211 Bci Seguros - Data\*.srt`. WER baseline sin enhance: 26.0% (5min), 16.8% (15min). Con enhance: 27.6% y 23.0% respectivamente.
- **Causa**: enhance distorsiona embeddings y confunde Whisper.
- **Acción**: mover enhance a post-processing (solo para output _limpio.m4a de escucha humana). Diarization, recognition y transcription siempre usan audio sin enhance.

### 3. A/B test de loudnorm
- **Hallazgo**: loudnorm es neutral para WER (<0.5pp diferencia) y marginalmente positivo para speaker recognition (+7.7pp en un caso).
- **Acción**: mantener habilitado. Costo trivial (~0.5s).

### 4. Profiling completo del pipeline
- Agregamos @timed a todos los pasos. Zero blind spots.
- Benchmark reproducible: `python benchmark_pipeline.py`
- Tiempos medidos: enhance 52s (61%), diarize 13s (16%), transcribe 13s (17%), speaker_embeddings 2.5s (3%), publish_artifacts 1s (1%).

### 5. Migración a Typer con subcomandos
- CLI migrado de argparse a Typer con validación de paths built-in.
- Subcomandos: `run`, `recognize`, `diagnose`.
- `recognize --force` permite re-ejecutar solo recognition sin re-diarizar.
- `diagnose` muestra score matrix JSON sin modificar artifacts.

### 6. Cache invalidation inteligente
- `speaker_map_params.json` sidecar registra params usados.
- Si --speakers o threshold cambian, cache se invalida automáticamente.

### 7. Diagnóstico JSON para iteración por agente
- `recognition_diagnostics.json` se guarda en artifacts después de recognition.
- Contiene: threshold, min_margin, scores per tag × voice, decision.
- Un agente puede leerlo, evaluar, y relanzar `recognize --force` con params ajustados.

## Qué no funcionó

### 1. batch_size 4→16
- **Hipótesis**: benchmark previo mostraba 5.12x speedup con batch_size=16.
- **Realidad**: en audio corto (6 min) con large-v3-turbo, batch_size=16 es MÁS LENTO (14.7s vs 11.6s). El overhead de batching no se amortiza.
- **Lección**: siempre medir antes/después en el contexto real.

### 2. Overlap enhance + diarization (CUDA streams paralelos)
- **Hipótesis**: correr enhance y diarization en paralelo ahorraría ~20% (de 57s a 45s).
- **Realidad**: solo 3.6% mejora. GPU SM contention causa que enhance suba de 44.7s a 54s.
- **Lección**: en GPU con pocos SMs (RTX 2070), dos modelos compitiendo se degradan mutuamente.

### 3. SpeechBrain 1.1.0
- **Hipótesis**: upgrade eliminaría shim list_audio_backends.
- **Realidad**: SpeechBrain 1.1.0 tiene bug de lazy import de k2_fsa que rompe diarization pipeline.
- **Acción**: revertir a 1.0.3 y mantener el shim.

### 4. JP Traverso y Carlos Soublette no matchean
- **Problema**: sus voice samples no son representativos del audio de la reunión (micrófono diferente, distancia, etc.). Score máximo 0.501 y 0.334 contra threshold 0.55.
- **Solución parcial**: bajar threshold a 0.48 captura a JP (0.501) pero Carlos sigue sin matchear (0.334 máximo). Esto fue un experimento manual, no se cambió el threshold en producción.
- **Threshold actual en producción**: 0.55 (constante `SPEAKER_SIMILARITY_THRESHOLD` en `speaker_recognition.py`). El experimento con 0.48 se hizo ad-hoc y no se mergeó.
- **Lección**: la calidad de los samples de enrollment es crítica. Samples de un contexto diferente pueden ser inútiles.

## Hallazgos técnicos clave

1. **Enhancement perjudica ASR** — validado con A/B test en 2 audios con SRT de referencia.
2. **Loudnorm es neutral/positivo** — mantener, costo trivial.
3. **GPU contention real** — en RTX 2070 (2560 SMs), dos modelos en paralelo no ganan vs secuencial.
4. **torchaudio 2.10 en Windows**: torchcodec no funciona, requiere shim PyAV en compat.py.
5. **speechbrain 1.0.3 requiere list_audio_backends shim** — no upgradeable a 1.1.0 por bug k2_fsa.
6. **Cache por etapas funciona**: 16k.wav, enhanced.wav, diarization.rttm, speaker_map.json cada uno con su ciclo de vida.
7. **Typer > argparse** para CLIs Python 2026: validación de paths built-in, rich output, subcomandos.

## Arquitectura actual del pipeline

```
python -m speechlib run "audio.m4a" --speakers "A,B,C,D,E,F" --compress -v

preprocessing → loudnorm → diarize(num_speakers=6) → recognize(filtered library) → transcribe → output
                                                                                         ↓
                                                                              enhance → compress → _limpio.m4a
```

Preprocessing: convert_to_wav → convert_to_mono → re_encode (16-bit) → resample_to_16k. Cache en `16k.wav`.

- ASR siempre usa audio post-loudnorm (sin enhance)
- num_speakers derivado de --speakers
- Library filtrada a solo asistentes con sample
- Nombres sin sample asignados por descarte (más segmentos → primer nombre)
- Enhance solo para output de escucha humana

## Plan futuro

### Pendiente de la sesión
1. **Re-diarizar CCS Gerentes con num_speakers=6**: el RTTM actual fue generado sin hint. Borrar rttm y relanzar con --speakers para obtener diarización óptima.
2. **Enrollar Carlos Soublette**: sus samples actuales no son representativos. Extraer clips de una reunión donde habla claro y enrollar.
3. **Tercer audio BCI Seguros** (48 min): no se alcanzó a procesar ni testear.

### Benchmark
- Script: `benchmark_pipeline.py` en la raíz del repo.
- Uso: `python benchmark_pipeline.py [--audio PATH] [--skip-enhance] [--voices PATH]`
- Default: usa `examples/obama_zach.wav` (~6 min, 2 speakers).
- Requiere `SPEECHLIB_PROFILE=1` (se activa internamente).
- Alternativas para enhance: FastEnhancer (github.com/aask1357/fastenhancer, RTF 0.012), DeepFilterNet3 (`pip install deepfilternet`, real-time on CPU).

### Features pendientes
1. **--threshold y --min-margin como flags CLI**: hoy son constantes globales (0.55/0.10). Permitir override por corrida.
2. **Invalidación de RTTM por cambio de num_speakers**: hoy el RTTM no tiene sidecar de params. Si cambias --speakers (y por ende num_speakers), el RTTM viejo se reutiliza incorrectamente.
3. **Agente automático de iteración**: script que lee recognition_diagnostics.json, evalúa contra ground truth, y ajusta params automáticamente.
4. **Alternativas a MossFormer2 para enhance de output**: FastEnhancer (25-50x más rápido) o DeepFilterNet3 (pip installable).

### Deuda técnica
1. **uv.lock**: el lockfile cambió pero speechlib ya no usa uv (managed=false). Considerar eliminar uv.lock del repo.
2. **tests/test_acceptance_speaker_recognition.py**: usa examples/obama_zach.wav como e2e fixture. Puede ser flaky si el modelo pyannote/embedding cambia.
3. **compat.py**: dos shims activos (list_audio_backends para speechbrain 1.0.3, torchcodec para Windows). Revisitar cuando upgrade sea posible.

## Métricas de la sesión

- Tests: 356 → 373 (17 nuevos AT)
- Commits: 12 en refactor/speaker-domain
- Patches (mocks): todos ≤ 2 por test file (solo GPU boundaries)
- Pipeline sin enhance: 27s para 6 min de audio
- Pipeline con enhance: 83s para 6 min de audio
- False positives eliminados: 3 (de 3 posibles) con --speakers
