# Speaker Recognition — Aprendizajes para agentes

Documento orientado a un agente de código que trabaje en `speechlib`.
Contiene decisiones validadas empíricamente, trampas conocidas, y
reglas que NO son obvias desde el código.

---

## Regla 1: seleccionar segmentos por duración, no por orden de documento

**Contexto**: `_compute_averaged_embeddings_per_tag` computa el embedding
promedio de un SPEAKER_XX para matchear contra la library de voces.

**Trampa**: iterar turnos en orden cronológico y parar al sumar N segundos
produce un embedding sesgado. En meetings largos los primeros turnos de un
speaker suelen ser cortos (0.5-2s) y contaminados por crosstalk o ruido
inicial. El embedding promedio NO representa al speaker.

**Evidencia**: Alicanto SPEAKER_00 (Pamela Falconi):
- Primeros 25 turnos (62s, orden doc): similarity 0.39 → no identificada
- Top-5 turnos más largos (46s): similarity 0.71 → identificada correctamente

**Regla**: usar `select_segments_for_embedding` que ordena por duración
descendente. Los turnos largos (>5s) son casi siempre monólogos limpios de
un solo speaker. Ubicación: `speechlib/domain/recognition.py`.

**Test que lo protege**: `tests/test_acceptance_pamela_alicanto_recognition.py`

---

## Regla 2: el rounding de _build_speaker_groups afecta embeddings

**Contexto**: `_build_speaker_groups` redondea start/end a 0.1s
(`round(turn.start, 1)`). Los tests de aceptación que construyen speakers
directamente desde la annotation de pyannote NO aplican este rounding.

**Trampa**: un test que computa embeddings sin rounding puede medir
similarity=0.498, pero el pipeline real con rounding mide similarity=0.509.
Si el threshold está en 0.50, el test pasa pero el pipeline falla.

**Regla**: en tests de aceptación que verifican speaker recognition, usar
`_build_speaker_groups(annotation)` para obtener los speakers con el mismo
rounding que el pipeline real. NUNCA construir speakers manualmente
iterando `annotation.itertracks()` sin redondear.

---

## Regla 3: threshold actual es 0.55 y el margen es 0.10

**Valores calibrados empíricamente sobre Alicanto (community-1)**:

| Speaker | Similarity | Resultado con 0.55 + margin 0.10 |
|---|---|---|
| Pamela Falconi | 0.720 | ✅ identificada (margen 0.288) |
| Agustin Villena | 0.719 | ✅ identificado (margen 0.231) |
| SPEAKER_02 (unknown) | 0.509 | ✅ rechazado (< threshold) |
| SPEAKER_05 (unknown) | 0.510 | ✅ rechazado (< threshold) |
| SPEAKER_04 (unknown) | 0.572 | ✅ rechazado (margen 0.064 < 0.10) |

**Trampa**: bajar threshold para "mejorar recall" genera falsos positivos
en voces masculinas similares. A threshold 0.45, speakers desconocidos
matchean falsamente con voces de la library de otros contextos.

**Regla**: si necesitas cambiar threshold, correr AT de Alicanto
(`test_full_speaker_map_after_select_segments_fix`) que valida tanto
identificaciones correctas como ausencia de falsos positivos.

**Tests que lo protegen**: `tests/test_speaker_threshold_constant.py`,
`tests/test_acceptance_recognition_quality.py`

---

## Regla 4: pyannote/embedding tiene baja discriminación intra-speaker

**Evidencia empírica** (25 clips con ground truth de Alicanto):

| Caso | Distancia coseno intra-speaker |
|---|---|
| Pamela (4 clips puros) | mediana 0.437, max 0.587 |
| Orlando (5 clips, 2 tags) | mediana 0.564, max 0.733 |
| Nicolas Loira (4 clips) | mediana 0.536, max 0.686 |

**Trampa**: asumir que el embedding model produce distancias intra-speaker
< 0.3 (como sería en un modelo ECAPA-TDNN). Con pyannote/embedding, la
varianza intra-speaker es ~0.40-0.55, comparable a la distancia
inter-speaker en casos difíciles (Daniel vs Nicolas: 0.495).

**Implicación**: NO intentar subclustering o cross-tag merging con
thresholds agresivos (< 0.30) usando pyannote/embedding. Los clusters
se confunden. Si se necesita subclustering, considerar primero un
upgrade a ECAPA-TDNN (datos empíricos en `devdocs/alicanto_fixture.md`).

---

## Regla 5: el cache de speaker_map.json requiere invalidación manual

**Contexto**: `_run_speaker_recognition_cached` carga speaker_map.json si
existe, sin verificar si los parámetros (threshold, modelo, RTTM) cambiaron.

**Trampa**: cambiar threshold o modelo de embedding y esperar que el pipeline
refleje el cambio. El cache persiste el resultado anterior.

**Regla**: después de cambiar SPEAKER_SIMILARITY_THRESHOLD, modelo de
embedding, o diarization.rttm, borrar `speaker_map.json` del artifacts_dir
antes de re-correr. La ruta es `{source_parent}/.{source_stem}/speaker_map.json`.

---

## Regla 6: los 3 variants de audio (raw, normalized, enhanced) producen resultados similares en audio limpio

**Evidencia**: Bci Seguros (reunión de oficina, grabación limpia):
- raw vs normalized: correlación 0.996 (solo gain)
- raw vs enhanced: correlación 0.750 (denoising real pero speakers iguales)
- Los 3 producen los mismos 4 speakers identificados

**Implicación**: en audio limpio, el enhancement no mejora el speaker
recognition. El beneficio del enhancement es en audio con ruido/reverb
(Alicanto, salas grandes).

**Regla**: no asumir que "enhanced = mejor". Comparar resultados antes de
invertir tiempo de procesamiento en enhancement.

---

## Regla 7: el trailing space en filenames rompe artifacts_dir en Windows

**Contexto**: `AudioState.artifacts_dir` computa `.{stem}` del source path.
Si el filename tiene trailing space (ej. `"Voz .m4a"` → stem `"Voz "`),
Windows no puede crear directorios con trailing space.

**Fix**: `self.source_path.stem.strip()` en la property. Ya aplicado.

**Test que lo protege**: `tests/test_audio_state.py::test_artifacts_dir_strips_trailing_spaces`

---

## Regla 8: community-1 produce mejores clusters que 3.1 en meetings largos

**Evidencia**: Alicanto 3.25h:
- 3.1: ~10 clusters con alta fragmentación (mismo speaker en 2-3 clusters)
- community-1: 6 clusters más compactos (mejor match con 6 speakers reales)

**Pero**: en audio corto/limpio (Bci Seguros 48min), la diferencia es mínima.

**Regla**: el modelo actual es `pyannote/speaker-diarization-community-1`
en `speechlib/diarization.py`. NO revertir a 3.1. Si hay regresiones,
verificar primero que el RTTM cacheado corresponda al modelo correcto.

---

## Anti-patrón: investigar bugs con scripts ad-hoc

**Qué NO funcionó**: crear scripts en `temp/` para investigar bugs
(ej. `temp/debug_pamela_match.py`). El usuario rechaza este approach.

**Qué SÍ funcionó**: escribir un AT que capture el bug en RED,
investigar la causa raíz leyendo los resultados del AT, luego escribir
unit tests focales y el fix siguiendo TDD. El AT sirve tanto de
diagnóstico como de validación.

**Regla**: todo proceso de nueva feature o corrección debe seguir
ATDD y GOOS-sin-mocks. Ver `C:\workspace\#dev\cen-valtx4\docs\desarrollo\goos-sin-mocks-python.md`.

---

## Anti-patrón: abstraer funciones de 3 líneas al dominio

**Qué NO funcionó**: intentar extraer `loudnorm.py` al dominio.
Son 5 líneas de aritmética (gain dB → linear → clamp). Extraerlas
sería abstracción prematura.

**Regla**: extraer al dominio solo cuando la lógica es:
1. No trivial (>10 líneas de lógica de negocio)
2. Testeable independientemente (el test puro aporta valor)
3. Reutilizable o propensa a bugs (como la selección de segmentos)

Tres líneas de math inline son mejores que un helper en otro archivo.
