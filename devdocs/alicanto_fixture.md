# Alicanto Fixture — Audio de referencia para validar speaker recognition

## Audio fuente

**Path**: `C:\workspace\@recordings\20260402 Alicanto\Voz 260402_151510.m4a`
**Duración**: 3.25 h (11705 s, 195 min)
**Sample rate**: 48 kHz (re-sampled a 16 kHz para el pipeline)
**Tipo**: meeting de equipo, ~6 speakers reales, audio recording de sala con reverb y crosstalk

**Artefactos cacheados**: `C:\workspace\@recordings\20260402 Alicanto\.Voz 260402_151510\`
- `16k.wav` (374 MB) — preprocesado mono 16kHz
- `enhanced.wav` (374 MB) — MossFormer2 enhanced
- `diarization.rttm` — pyannote community-1 (backup 3.1 en `diarization.3.1.rttm.bak`)
- `speaker_map.json` — recognition contra library, threshold 0.55 (backup 3.1 en `speaker_map.3.1.json.bak`)
- `transcript_es.vtt` — whisper large-v3-turbo
- `transcript.json` — aggregate del nuevo dominio (Slice 5+)
- `samples/` — clips por speaker (Slice 16+18)

## Por qué es un buen fixture

Alicanto reúne en un solo recording **todos los modos de falla** del speaker recognition tradicional:

1. **Recording largo (3.25h)** → embedding drift de pyannote
2. **Múltiples voces masculinas con timbres similares** (Daniel, Nicolas Loira, Marcos, Orlando)
3. **Audio de un video reproducido en la sala** capturado por el mismo mic — voces femeninas que NO son participantes
4. **Ruido ambiental** (toses, sillas, papel) que pyannote a veces clusteriza como "speaker"
5. **2 speakers en la library** (Pamela, Agustin) → permite verificar identification path
6. **4 speakers NO en la library** → permite verificar el path de no identificados (`por_nombrar/`)
7. **Crosstalk frecuente** → pone a prueba la robustez del embedding promediado
8. **Distribución de tiempo desbalanceada** (Pamela 95 segs vs Agustin 658 segs) — algunos hablan mucho, otros poco

Cualquier estrategia nueva de speaker recognition debe pasar por este fixture antes de promoverse.

## Speakers reales (ground truth confirmado por el usuario)

| Persona | En library? | Caracteristicas |
|---|---|---|
| **Pamela Falconi** | ✓ | Mujer, voz clara |
| **Agustin Villena** | ✓ | Hombre, dominante en participacion |
| **Orlando** | ✗ | Hombre |
| **Daniel** | ✗ | Hombre, timbre similar a Nicolas |
| **Nicolas Loira** | ✗ | Hombre, timbre similar a Daniel |
| **Marcos** | ✗ | Hombre, intervenciones cortas, generalmente mixed |

Total: 6 personas reales + audio de un video (mujeres, ignorar) + ruido ambiental.

## Ground truth community-1 (modelo actual, 2026-04-09)

Mapping confirmado por el usuario escuchando clips de `samples/por_nombrar/`:

| Tag community-1 | Persona | Segmentos | Tiempo total |
|---|---|---|---|
| SPEAKER_00 | **Pamela Falconi** ✓ library | 140 | ~8 min |
| SPEAKER_01 | **Agustin Villena** ✓ library | 745 | ~78 min |
| SPEAKER_02 | Marcos | 246 | ~22 min |
| SPEAKER_03 | Orlando | 361 | ~34 min |
| SPEAKER_04 | Nicolas Loira | 498 | ~25 min |
| SPEAKER_05 | Daniel | 122 | ~8 min |

**6 clusters = 6 personas reales**. Community-1 clavó el conteo exacto de speakers. No hay clusters de ruido, video ni fragmentación (mejora significativa vs 3.1 que producía ~8-10 clusters con mezcla).

**Speaker recognition (threshold 0.55, min_margin 0.10)**:
- Pamela: sim=0.720, margen=0.288 → identificada ✅
- Agustin: sim=0.719, margen=0.231 → identificado ✅
- Marcos: sim=0.509 < 0.55 → no identificado (correcto, no está en library) ✅
- Orlando: sim=0.319 → no identificado ✅
- Nicolas: sim=0.572, margen=0.064 < 0.10 → rechazado por margen ✅
- Daniel: sim=0.510 < 0.55 → no identificado ✅

---

## Ground truth pyannote 3.1 (histórico, pre-upgrade)

Estos clips son el output del pipeline (Slice 18 RAW turns) sobre el cache pyannote 3.1. El usuario los caracterizó manualmente. **Los SPEAKER_XX de esta sección NO corresponden a los de community-1 (numeración diferente).**

### Speakers identificados via library

#### `samples/Agustin Villena/`
| Clip | Contenido real |
|---|---|
| clip_01.wav | Agustin (puro) |
| clip_02.wav | **Agustin + Nicolas** (mixed) |
| clip_03.wav | **Agustin + Orlando** (mixed) |
| clip_04.wav | **Marcos + Agustin** (sequential) |
| clip_05.wav | Agustin (puro) |

#### `samples/Pamela Falconi/`
| Clip | Contenido real |
|---|---|
| clip_01.wav | Pamela (puro) |
| clip_02.wav | Pamela (puro) |
| clip_03.wav | Pamela (puro) |
| clip_04.wav | Pamela (puro) |
| clip_05.wav | **Pamela + Orlando** (mixed) |

### Speakers no identificados (en `samples/por_nombrar/`)

#### `SPEAKER_00/` — 3 fuentes mezcladas
| Clip | Contenido real |
|---|---|
| clip_01.wav | Orlando |
| clip_02.wav | mujeres del video |
| clip_03.wav | mujeres del video |
| clip_04.wav | Orlando |
| clip_05.wav | Orlando + mix |

#### `SPEAKER_03/` — 2 personas mezcladas
| Clip | Contenido real |
|---|---|
| clip_01.wav | Daniel |
| clip_02.wav | Daniel |
| clip_03.wav | Nicolas Loira |
| clip_04.wav | Nicolas Loira |
| clip_05.wav | Nicolas Loira |

#### `SPEAKER_05/` — 4 personas mezcladas (cluster temporalmente dominante, 454 segs)
| Clip | Contenido real |
|---|---|
| clip_01.wav | Orlando |
| clip_02.wav | Orlando |
| clip_03.wav | Nicolas Loira |
| clip_04.wav | Orlando |
| clip_05.wav | **Marcos + Daniel** (mixed) |

#### `SPEAKER_06/` — audio del video (ignorar)
Todos los clips son audio del video. No es un participante. Debe quedarse como unknown / no enrolar.

#### `SPEAKER_07/` — Nicolas + ruido + Marcos mixed
| Clip | Contenido real |
|---|---|
| clip_01.wav | Nicolas Loira |
| clip_02.wav | ruido ambiental |
| clip_03.wav | ruido ambiental |
| clip_04.wav | ruido ambiental |
| clip_05.wav | **Marcos + Orlando** (mixed) |

### Marcos puro (extracción manual de 5 segundos)

**Range**: `2:40:18` a `2:40:23` (9618s a 9623s en `enhanced.wav`)

Es la **única ocurrencia confirmada de Marcos sin contaminación** que tenemos. Útil como anchor embedding para validar estrategias nuevas. En el RTTM de pyannote 3.1, ese rango está dentro de turnos etiquetados como `SPEAKER_05` (overlap 3.35s + 1.40s).

## Hallazgos empíricos del análisis de fingerprints

Embeddings computados sobre los 25 clips de SPEAKER_XX no identificados + 10 clips de identificados + Marcos GT. Cacheados en:
- `C:\Users\agust\AppData\Local\Temp\alicanto_embeddings.json` — pyannote/embedding (modelo actual)
- `C:\Users\agust\AppData\Local\Temp\alicanto_ecapa.json` — speechbrain/spkrec-ecapa-voxceleb (ECAPA-TDNN)

### Distancias INTRA-grupo (clips de la misma persona — IDEAL: pequeñas)

| Speaker | n clips | pyannote (avg) | ECAPA (avg) | mejora ECAPA |
|---|---|---|---|---|
| Pamela puros | 4 | 0.468 | **0.290** | −38% |
| Orlando (en SPEAKER_00 y SPEAKER_05) | 5 | 0.524 | **0.407** | −22% |
| Daniel | 2 | 0.417 | **0.272** | −35% |
| Nicolas Loira | 4 | 0.503 | **0.364** | −28% |

### Distancias INTER-grupo (personas distintas — IDEAL: grandes)

| Test | pyannote | ECAPA |
|---|---|---|
| Daniel vs Nicolas Loira (DIFÍCIL — voces similares) | 0.495 | 0.343 |
| Orlando vs Daniel | 0.771 | 0.741 |
| Orlando vs Nicolas Loira | 0.832 | 0.792 |
| Orlando vs Pamela (FÁCIL — sexo distinto) | 0.855 | 0.879 |

### Discriminative ratio (inter / intra, MAYOR = mejor)

| Test | pyannote | ECAPA | ganador |
|---|---|---|---|
| Daniel intra / Daniel-Nicolas inter | 1.19 | **1.26** | ECAPA +6% |
| Nicolas intra / Daniel-Nicolas inter | 0.98 | 0.94 | pyann (marginal) |
| Orlando intra / Orlando-Daniel inter | 1.47 | **1.82** | ECAPA +24% |
| Orlando intra / Orlando-Nicolas inter | 1.59 | **1.95** | ECAPA +22% |
| Pamela intra / Pamela-Orlando inter | 1.83 | **3.03** | ECAPA +66% |
| Orlando intra / Orlando-Pamela inter | 1.63 | **2.16** | ECAPA +33% |

**ECAPA-TDNN gana en 5 de 6 casos**, con mejoras significativas (+22-66%) en casos donde los speakers son razonablemente distintos. El caso "voces masculinas similares" (Daniel/Nicolas) mejora marginalmente — ningún embedding model puro lo resuelve.

### Library matching (con pyannote/embedding)

Distancias del **Marcos puro** (5s) vs la voice library:

```
0.742  CCS - Juan Pablo Traverso  ← top match (FALSO — Marcos no esta en library)
0.759  AA - Cristian Ruiz
0.778  BciS - Jolyon Abello
0.803  Agustin Villena
```

Con threshold actual 0.45 (similarity ≥0.55, distancia ≤0.55) Marcos NO se identifica como nadie (correcto). Pero si bajara a threshold 0.30 sería falsamente identificado como Juan Pablo Traverso. **Min margin de Slice 9 (0.10) lo salva por 0.017** — está al límite.

Para `Daniel puro` vs library (Daniel NO está en library):

```
0.389  AA - Cristian Ruiz   ← peligrosamente cerca, pasaria threshold 0.45
0.515  BciS - Jolyon Abello
```

Con threshold actual, Daniel sería falsamente identificado como Cristian Ruiz. Solo lo salva el min_margin (gap 0.126).

### Crosstalk impact

`AG_02` (Agustin + Nicolas mixed) vs centroide Agustin puro: **distancia 0.317** — entra en el rango intra-Agustin (0.337). La contaminación por crosstalk **no desplaza significativamente** el embedding cuando un speaker domina. Esto explica por qué library matching funciona razonablemente para Pamela y Agustin a pesar de tener clips contaminados — Slice 5 cabling hace `assign_speakers` sobre el promedio del SPEAKER_XX entero.

## Estrategias evaluadas

| Estrategia | Aprobada empíricamente? | Costo de implementación |
|---|---|---|
| **A** — pyannote 3.1 baseline | Estado actual (problemático) | 0 |
| **B** — pyannote community-1 (drop-in) | Pendiente — corrida en progreso es lenta (>1h) | ~30 min de re-diarización |
| **C** — pyannote 3.1 + num_speakers hint | No probada (requiere community-1) | ~1 hora |
| **D** — ECAPA-TDNN como embedding | Validada (+22-66% discriminative power) | ~3 horas (1 slice) |
| **E** — Subclustering intra-SPEAKER_XX | Validada con simulación (separa 11 de 13 clips correctamente) | ~6 horas (1 slice) |
| **F** — Cross-tag merging post-subclustering | Validada con simulación pero descubrió bug del puente transitivo | ~4 horas (1 slice) |

### Resultados de la simulación de Estrategia E (subclustering)

K-means con auto k sobre los embeddings ECAPA, agrupado por SPEAKER_XX:

```
SPEAKER_00 → 3 sub-grupos
  sub_a: Orlando × 3 (clips 1, 4, 5) ✓ PURO
  sub_b: Video × 1 (clip 3) ✓
  sub_c: Video × 1 (clip 2) ✓

SPEAKER_03 → 3 sub-grupos
  sub_a: Daniel × 1 (clip 1) ✓
  sub_b: Nicolas × 2 (clips 3, 4) ✓ PURO
  sub_c: Daniel + Nicolas × 2 (clips 2, 5) ⚠️ MIXED — caso difícil

SPEAKER_05 → 3 sub-grupos
  sub_a: Orlando × 2 (clips 1, 2) ✓ PURO
  sub_b: Marcos × 1 (clip 5) ✓
  sub_c: Nicolas + Orlando × 2 (clips 3, 4) ⚠️ MIXED

SPEAKER_07 → 3 sub-grupos
  sub_a: Nicolas × 1 (clip 1) ✓
  sub_b: Ruido × 1 (clip 2) ✓
  sub_c: Marcos × 1 (clip 5) ✓
```

**11 de 13 sub-grupos correctos**. Los 2 mixed son del caso "voces masculinas similares" — limitación inherente que ningún algoritmo de clustering resolverá sin información semántica adicional.

### Resultados de la simulación de Estrategia F (cross-tag merging)

Pares con distancia < 0.30 detectados entre los sub-grupos del paso anterior:

```
SP00_a_Orlando  <-> SP05_a_Orlando   d=0.166  ✓ CORRECTO
SP03_a_Daniel   <-> SP03_c_Mixed     d=0.226  ⚠️ Daniel + clip mixto
SP03_b_NicolasL <-> SP03_c_Mixed     d=0.241  ⚠️ Nicolas + mismo clip mixto
SP05_a_Orlando  <-> SP05_c_Mixed     d=0.299  ⚠️ Orlando + clip mixto
```

**Bug del puente transitivo**: el sub-grupo "mixed" actúa como puente entre Daniel y Nicolas. Si el algoritmo permite merging transitivo, fusionaría los dos en uno solo. Fix: descartar sub-grupos con varianza interna alta antes del merging cross-tag, O no permitir merging transitivo (procesar pares en orden sin propagación).

## Estado actual del experimento (Apr 2026)

### community-1 validado sobre Alicanto

Tras descubrir que el run anterior de community-1 (>1h 54min) era por NO tener torch+CUDA activo (entorno equivocado), re-corrida en entorno conda `speechlib` con GPU activa:

- **14.8 minutos** total para 3.25h de audio (vs >2h hung)
- 6 clusters detectados (vs 8 con 3.1)
- **Cluster purity 64% → 83.3% (+19.3pp)** medido contra los labels del usuario

Ver `<audio>/experiments/01_community1_results.md` para análisis completo.

**Hallazgos clave**:
- ✅ community-1 es upgrade neto: +19.3pp en cluster purity
- ✅ Filtra audio del video y ruido ambiental (no aparecen como clusters propios)
- ✅ Agustin se identifica perfecto vs library
- ⚠️ Pamela perdió library matching (cluster es 80% puro pero el match falla)
- ⚠️ Orlando sigue fragmentado en 2 clusters (problema del embedding model, no del diarization)
- ⚠️ Daniel/Nicolas siguen mixed (caso "voces masculinas similares" sin solución por diarización)

**Decisión**: `pyannote/speaker-diarization-community-1` queda como modelo default en `speechlib/diarization.py`. Es upgrade neto en audios hard sin regresión en audios fáciles (verificado contra Bci Seguros — ver `bci_seguros_fixture.md`).

## Cómo usar este fixture

Cualquier estrategia nueva (modelo, threshold, algoritmo de clustering) debe ejecutarse sobre los embeddings cacheados de Alicanto y compararse contra esta caracterización. Métricas mínimas:

1. **Cluster purity**: cada cluster final del algoritmo debe contener mayoría (>80%) de clips de UNA persona real
2. **Speaker coverage**: cada persona real con suficiente audio debe aparecer en exactamente UN cluster (no fragmentación)
3. **Noise isolation**: audio del video y ruido ambiental deben quedar en clusters propios, separados de los speakers humanos
4. **Marcos puro discrimination**: el clip Marcos puro (2:40:18-23) debe ser closest a algún cluster que contenga Marcos, no a Orlando ni a la library

Reproducir la validación:

```bash
python -c "
import json, numpy as np
from pathlib import Path
ec = json.loads(Path(r'C:\Users\agust\AppData\Local\Temp\alicanto_ecapa.json').read_text())
# ... aplicar tu algoritmo a ec ...
# ... comparar resultados con la tabla de ground truth de este doc ...
"
```

## Lessons learned para otros recordings

1. **El voice library matching tiene threshold ajustable**, pero el modo de falla de pyannote (over-clustering + under-clustering simultáneo) es estructural — no se resuelve con threshold tuning.

2. **Embedding model alternativo (ECAPA-TDNN) ayuda dramáticamente** para discriminative power general (+22-66%) pero NO resuelve el caso "voces masculinas similares" (Daniel/Nicolas).

3. **Crosstalk no rompe library matching** mientras un speaker domine el clip — el embedding promedio se mantiene cerca del speaker mayoritario.

4. **Re-clustering post-hoc puede separar 11 de 13 sub-clusters** sintéticos. Los 2 fallos son los casos verdaderamente ambiguos.

5. **Cross-tag merging tiene bug del puente transitivo** que requiere fix de diseño (no permitir cadenas A↔M y B↔M donde M es heterogéneo).

6. **Pyannote 4.0 community-1 está disponible** y promete mejoras significativas en speaker confusion (DER AliMeeting 24.5% → 20.3%) pero la corrida actual es 7x más lenta de lo esperado — pendiente validar si vale la pena el trade-off.

## Referencias

- Plan completo: `~/.claude/plans/foamy-jumping-music.md`
- Cache pyannote 3.1: `.Voz 260402_151510/diarization.3.1.rttm.bak`
- Cache speaker_map 3.1: `.Voz 260402_151510/speaker_map.3.1.json.bak`
- Embeddings pyannote: `%TEMP%\alicanto_embeddings.json`
- Embeddings ECAPA: `%TEMP%\alicanto_ecapa.json`
- Marcos GT slice: `%TEMP%\marcos_gt.wav` (5s, 2:40:18-23 del enhanced.wav)
