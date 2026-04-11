# HOWTO: speechlib — transcripción con speaker recognition

**Fecha**: 2026-04-10 · **Branch**: `refactor/speaker-domain`

Guía práctica para usar speechlib desde la CLI: transcribir reuniones,
identificar speakers contra una voice library, y ajustar recognition
iterativamente con un agente guiado por el usuario.

> **Scope**: este HOWTO cubre uso del CLI moderno (Typer, subcomandos
> `run`/`recognize`/`diagnose`). **No cubre**: la API legacy de la clase
> `Transcriptor` del README, ni el entrenamiento de modelos custom, ni
> deployment en producción.

---

## 1. Qué es speechlib

Librería + CLI Python. Cuatro pasos: diarización (`pyannote`) → speaker
recognition (cosine similarity de embeddings contra una voice library
opcional, **produce sugerencias**, no decide) → transcripción
(`faster-whisper`) → confirmación humana opcional para asignar nombres
reales. Output `.vtt`/`.txt`: por default tiene `[SPEAKER_00]`,
`[SPEAKER_01]`, etc. Tras correr `confirm` con un `speaker_map.json`
escrito por vos, los tags mapeados se reemplazan por nombres reales.

---

## 2. Requisitos previos

### Bloqueantes

1. **`HF_TOKEN` de HuggingFace**. Creá cuenta, generá token en
   Settings → Access Tokens, y exportalo (`export HF_TOKEN=hf_xxx` en
   Linux/Mac/Git Bash, `setx HF_TOKEN hf_xxx` en Windows). O pasalo
   con `--token hf_xxx` (CLI gana sobre env var).

2. **Aceptar licencias de 2 modelos gated** en HuggingFace (click
   "Agree" logueado): `pyannote/speaker-diarization-community-1` y
   `pyannote/embedding`.

   Sin esto, el pipeline falla con `401 Unauthorized`. Para verificar
   setup antes de invertir tiempo en un audio largo: simplemente
   **corré `run` sobre un audio corto** (10-30s). El comando carga los
   modelos antes de procesar, así que el error aparece en los primeros
   segundos. No hay "dry run" separado.

3. **GPU CUDA + cuDNN**: recomendado (CPU ~10x más lento). VRAM mínima:
   **6 GB** para `large-v3-turbo` (default), **4 GB** con `--model medium`,
   **2 GB** con `--model small`.

### Opcional

- **Voice library** (`--voices-folder`): para que el pipeline produzca
  sugerencias de identidad de cada cluster (top-3 candidatos + score)
  que vos confirmás vía `confirm` para obtener nombres reales en el VTT
  final. Ver secciones 3, 5 y 7.

---

## 3. Quick start — transcribir una reunión

El comando mínimo para transcribir un audio en español (default):

```bash
python -m speechlib run "mi_reunion.m4a"
```

Podés correrlo desde cualquier directorio con path relativo o absoluto.

**Qué hace**: preprocesa a 16 kHz + loudnorm, diariza con pyannote,
transcribe con faster-whisper, y guarda el VTT final en
`<audio_dir>/<stem>_limpio.vtt` (ej. `audios/reunion.m4a` →
`audios/reunion_limpio.vtt`). El sufijo `_limpio` es **automático
aunque NO pases `--compress`** — es el naming canónico del output
publicado, no implica enhancement/compresión. Cachea artefactos
intermedios en `<audio_dir>/.<stem>/` (oculto). Imprime una línea por
etapa indicando cache reuse vs recompute.

Sin voice library el VTT trae `[SPEAKER_00]`, `[SPEAKER_01]`, etc. Para
nombres reales necesitás voice library (sección 5).

### Flujo suggest + confirm con voice library

> ⚠️ **Cambio de comportamiento (2026-04-11)**: `run` ya **no aplica
> nombres automáticamente** al VTT incluso con voice library. Produce
> sugerencias y un VTT con `[SPEAKER_XX]` crudo. La asignación de
> nombres reales pasa por el subcomando nuevo `confirm`, después de que
> vos editás un `speaker_map.json` basado en las sugerencias.

Caso típico: tenés samples de **algunos** asistentes pero no todos.

1. **Creá la carpeta `voices/`** con un subdirectorio por speaker:

   ```
   voices/
   ├── Ana/
   │   └── enrolment.wav
   └── Bruno/
       └── clip1.wav
   ```

2. **Corré `run` con `--speakers`** listando los asistentes:

   ```bash
   python -m speechlib run reunion.m4a \
     --voices-folder ./voices \
     --speakers "Ana,Bruno,Carla Lopez,Diego" \
     -v
   ```

   El pipeline computa scores contra la voice library, escribe
   `<cache>/speaker_map_suggestions.json` con top-3 candidatos y
   `recommended` por cluster, y publica el VTT con `[SPEAKER_XX]` crudo.
   El mensaje del CLI dice `Speaker suggestions — N/M recommended`.

3. **Revisá el JSON de sugerencias**:

   ```bash
   cat audios/.reunion/speaker_map_suggestions.json
   ```

   Te da, por cada cluster pyannote: top-3 candidatos con scores y un
   `recommended` (que es `null` si el match es ambiguo o bajo threshold).

4. **Escribí tu propio `speaker_map.json`** en el cache, con tus
   decisiones finales:

   ```bash
   cat > audios/.reunion/speaker_map.json <<'EOF'
   {
     "SPEAKER_00": "Ana",
     "SPEAKER_01": "Bruno",
     "SPEAKER_02": "Carla Lopez",
     "SPEAKER_03": "Diego"
   }
   EOF
   ```

   Para clusters que no querés mapear todavía (o que son ruido),
   omitilos del JSON o mapealos a sí mismos (`"SPEAKER_04": "SPEAKER_04"`)
   — quedan literales en el VTT.

5. **Confirmá**:

   ```bash
   python -m speechlib confirm reunion.m4a
   ```

   Regenera `<audio_dir>/<stem>_limpio.vtt` con los nombres reales
   aplicados a los clusters mapeados, y deja `[SPEAKER_XX]` literal en
   los no mapeados.

Si después del paso 5 querés cambiar algo (renombrar, marcar uno como
no identificado), editá el `speaker_map.json` y re-corré `confirm`. Es
barato — segundos.

---

## 4. Los 4 subcomandos

### `run` — pipeline completo (produce sugerencias, no aplica)

Corre preprocess + diarize + (recognition score matrix si hay
voices_folder) + transcribe + publica VTT crudo + suggestions JSON.
**No** aplica nombres reales al VTT — eso lo hace `confirm`.

```bash
python -m speechlib run <audio_file> [opciones]
```

**Flags principales** (corré `--help` para la lista completa):

| Flag | Default | Uso |
|---|---|---|
| `--voices-folder PATH` | — | Voice library. Activa speaker recognition. |
| `--speakers "A,B,C"` | — | Asistentes esperados (nombres **case-sensitive** de subcarpetas en voices/). Filtra library y fuerza `num_speakers`. Ver sección 6. |
| `--language` | `es` | ISO 639-1. |
| `--model` | `large-v3-turbo` | Alternativas: `medium`, `small`, `tiny` (menos VRAM). |
| `--output-format` | `vtt` | O `txt`. |
| `--compress` | off | Genera además `{stem}_limpio.m4a`. El VTT ya usa `_limpio.vtt` sin este flag. |
| `--token hf_xxx` | `$HF_TOKEN` | Solo si no tenés env var. |
| `-v` | off | Logs por etapa. Recomendado primera vez. |

### `recognize` — re-ejecuta solo speaker recognition (suggestions)

Re-computa el `speaker_map_suggestions.json` sin re-diarizar ni
re-transcribir. Imprime el suggestions dict como JSON.

```bash
python -m speechlib recognize <audio_file> --voices-folder voices/ [--force]
```

Usá el **mismo path** que en `run` (lo necesita para el cache `.<stem>/`).

**Cuándo**: cambiaste la voice library, `--speakers`, o threshold/min_margin
y querés ver las nuevas sugerencias sin re-correr todo el pipeline.
`--force` borra el cache de suggestions para forzar recomputo aunque los
params no hayan cambiado (seguro al iterar). **No regenera el VTT** —
para eso editás `speaker_map.json` y corrés `confirm`.

### `confirm` — aplica el speaker_map.json del usuario al VTT

```bash
python -m speechlib confirm <audio_file>
```

Lee `<cache>/speaker_map.json` (escrito por vos basado en
`speaker_map_suggestions.json`), aplica el mapeo a los segments del
transcript cacheado, regenera `transcript_<lang>.vtt` en el cache, y
re-publica `<stem>_limpio.vtt` al lado del audio. Tags ausentes del map
quedan como `[SPEAKER_XX]` literal. El JSON del usuario no se modifica.

**Errores claros**: si falta `speaker_map.json` o `transcript.json` en
el cache, el comando avisa con un mensaje accionable. La precondición
es haber corrido `run` antes.

### `diagnose` — matriz de scores (read-only)

```bash
python -m speechlib diagnose <audio_file> --voices-folder voices/
```

No modifica artefactos. Imprime resumen en terminal **y** guarda el JSON
completo en `<audio_dir>/.<stem>/recognition_diagnostics.json`. Estructura:

```json
{
  "threshold": 0.55,
  "min_margin": 0.10,
  "tags": {
    "SPEAKER_00": {
      "scores": {"Ana": 0.6234, "Bruno": 0.4892, "Carla": 0.3105},
      "decision": "Ana"
    },
    "SPEAKER_01": {
      "scores": {"Ana": 0.5012, "Bruno": 0.5018, "Carla": 0.4200},
      "decision": "SPEAKER_01"
    }
  }
}
```

`SPEAKER_01` no se identifica porque top1 (Bruno 0.5018) y top2 (Ana
0.5012) están a margin 0.0006 < min_margin 0.10, y ambos bajo threshold
0.55. El JSON te da la data exacta para decidir qué ajustar.

---

## 5. Voice library — estructura y enrolamiento

Estructura: **un subdirectorio por speaker** con **1+ samples `.wav`**.
El nombre del subdirectorio es el que aparece en el output.

```
voices/
├── Ana/
│   ├── clip1.wav
│   └── clip2.wav
├── Bruno/
│   └── sample.wav
└── Carla Lopez/
    └── introduccion.wav
```

**Reglas**:

- **Formato**: WAV (cualquier SR / bit depth).
- **Nombres case-sensitive**, espacios y acentos preservados.
- **Duración por sample**: **5-30 segundos recomendado**. Umbral duro
  **<0.5s** (se ignoran con warning, pyannote no puede generar embedding).
  Entre 0.5s y 5s los samples se usan pero pueden dar embeddings
  inestables — evitalos. Samples de minutos son desperdicio (el embedding
  no mejora con más duración).
- **Representatividad crítica** (error silencioso más común): los
  samples deben venir de un contexto acústico **similar al audio a
  transcribir**. Regla práctica: si podés escuchar la diferencia de
  ambiente en 1 segundo, el embedding también la nota. Cruzar muy
  distintos (teléfono 8 kHz vs lavalier 48 kHz) falla.
- **Múltiples samples**: se promedian los embeddings. Más samples
  diversos = embedding más robusto.
- **Reenrolamiento**: no hay comando de "enrol" separado. Agregá un
  `.wav` nuevo a `voices/Ana/`, corré `recognize --force`, y el
  pipeline automáticamente lee los `.wav` de `voices/Ana/`, computa
  embeddings de cada uno, y calcula el promedio fresco. No hay cache
  persistente de voice library — cada corrida recarga desde cero.
- **Prefijo `_` ignora el subdir**. Excepción: `_enhanced/` — ver Avanzado.

---

## 6. `--speakers`: constrained diarization

```bash
python -m speechlib run reunion.m4a \
  --voices-folder voices/ \
  --speakers "Ana,Bruno,Carla Lopez,Diego"
```

**Dos efectos**:

1. **Filtra la voice library** a esos speakers (elimina falsos positivos
   contra ausentes).
2. **Fuerza `num_speakers=N` en pyannote**: produce exactamente N tags.
   Si hay físicamente más personas, pyannote fusiona; si hay menos,
   parte el que más habla. **El conteo importa** — si no estás seguro
   cuántos hablan, omitilo y dejá que pyannote infiera.

**Asignación por descarte**: si un tag no matchea contra ninguno de los
speakers con samples, se asigna por **cantidad de segmentos** — el tag
que más habla recibe el primer nombre sin matchear de `--speakers`, etc.
Determinístico. Esto permite pasar nombres aunque NO tengas samples de
todos.

**Sin warning en conflictos**: si `num_speakers` forzado no coincide con
lo que pyannote habría inferido, el pipeline no avisa. El síntoma en el
VTT: un tag agrupando dos voces (fusión) o dos tags para una voz (partición).

---

## 7. Workflow iterativo con agente — ajustar speaker_map.json

Patrón cuando las sugerencias automáticas no aciertan en todos los
speakers y querés iterar sin repetir el pipeline entero.

**Diferencia clave con el flujo viejo**: ya no se itera bajando
thresholds. El loop ahora es **edición del `speaker_map.json` + `confirm`**.
La decisión humana es explícita y reversible: cada commit del map
regenera el VTT en segundos.

**Roles**: **vos** editás `speaker_map.json` y corrés `confirm`. El
**agente** (Claude Code u otro LLM) lee `speaker_map_suggestions.json` /
`recognition_diagnostics.json` / los clips de `samples/` y te propone
matches con razones, pero **no** decide solo.

**Cuándo empieza**: después de una corrida de `run` donde ves
sugerencias dudosas o clusters sin `recommended` que vos sí podés
identificar escuchando los clips.
**Cuándo termina**: cuando tu `speaker_map.json` cubre todos los
clusters que querés nombrar.

> ⚠️ **Trampa antes de re-correr `run`**: si cambiás el **conteo** de
> `--speakers` (ej. 4 → 5), `diarization.rttm` NO se invalida
> automáticamente. Borralo a mano. (`confirm` solo trabaja sobre
> el cache existente, no toca diarization.)

### Ciclo de iteración

**Precondición**: ya corriste `run` al menos una vez con
`--voices-folder`, así existen `speaker_map_suggestions.json` y
`transcript.json` en el cache.

1. **Inspeccionar las sugerencias**:
   ```bash
   cat audios/.reunion/speaker_map_suggestions.json
   ```
   Ves top-3 candidatos + recommended por cluster. Para cada uno donde
   `recommended` sea `null` o sospechoso, escuchá los clips en
   `audios/.reunion/samples/SPEAKER_XX/clip_*.wav`.

2. **Pedile contexto al agente**:
   > "SPEAKER_02 tiene recommended=null. Top1 Ana 0.51, top2 Bruno 0.50.
   > ¿Es Ana, Bruno, o ninguno? Mirá el JSON y los clips."

3. **Agente lee el JSON + escucha clips (via path)** y reporta sus
   hipótesis con razones, sin proponer cambio automático.

4. **Vos decidís** y editás `speaker_map.json` con tu decisión:
   ```json
   {
     "SPEAKER_00": "Ana",
     "SPEAKER_01": "Bruno",
     "SPEAKER_02": "Ana"
   }
   ```

5. **Re-confirmá**:
   ```bash
   python -m speechlib confirm reunion.m4a
   ```
   El VTT se regenera en segundos. Releé el output.

6. **Iterar** hasta que el VTT te conforme.

**Si el problema son los samples** (todos los scores <0.40 contra el
speaker esperado), reenrolá: agregá un `.wav` nuevo a `voices/Ana/` y
corré `recognize --force` para regenerar las suggestions con embeddings
frescos. Después seguís editando el `speaker_map.json` desde el paso 4.

**Reglas**: el agente nunca edita `speaker_map.json` por vos sin
mostrarte la decisión y la razón primero; vos sos quien commitea el map;
`confirm` es barato (segundos) y reversible — usalo libremente.

### Threshold vs min_margin

Reglas que el código aplica para aceptar un match:

1. `top1_score >= threshold` (default `0.55`)
2. `top1_score - top2_score >= min_margin` (default `0.10`)

Ambas deben cumplirse.

**Regla conservadora para bajar threshold**: elegí `T = max(top1 - 0.02, 0.40)`.
Deja 2pp de buffer sobre el top1 medido y nunca baja de 0.40 (por debajo
los embeddings son demasiado ruidosos). Antes de committear, **chequeá
el resto de la matriz** con `diagnose` para confirmar que ningún otro
`SPEAKER_XX` cruce contra un nombre equivocado al nuevo `T`.

**`min_margin`**: bajalo solo en casos raros donde dos voces son
genuinamente casi idénticas. Un margin chico normalmente significa
embeddings mal definidos.

### Cuándo el problema NO es el threshold (fix upstream)

- **Todos los scores <0.40**: samples desalineados del contexto acústico. Reenrolar.
- **Top1 inestable entre corridas**: samples muy cortos/ruidosos. Reenrolar >5s.
- **Score alto contra speaker equivocado**: samples contaminados con otra voz. Limpiar.
- **Speaker esperado no aparece en ningún turno**: pyannote no creó cluster.
  Probar sin `--speakers` para ver clusters detectados libremente.

---

## 8. Cache y artefactos

Cada corrida crea `<audio_dir>/.<stem>/` con artefactos reutilizables:

- `16k.wav` — audio post-loudnorm (input al embedding model + transcribe).
- `diarization.rttm` — output crudo de pyannote.
- `speaker_map_suggestions.json` — sugerencias top-3 + recommended por cluster (escrito por `run` y `recognize` cuando hay voice library).
- `speaker_map_params.json` — sidecar con los params usados (allowed_speakers, threshold, min_margin) para invalidar el cache de suggestions.
- `recognition_diagnostics.json` — matriz completa de scores (formato más rico).
- `transcript.json` — aggregate Transcript (segments con SpeakerIdentity).
- `transcript_<lang>.vtt` — VTT en el cache, fuente de la copia publicada.
- `samples/SPEAKER_XX/clip_*.wav` — clips por cluster para inspección humana.
- `speaker_map.json` — **lo escribe el usuario** (no el pipeline). Lo lee `confirm`.

**Cache invalidation automática**: cambiar `--speakers`, threshold o
min_margin invalida `speaker_map_suggestions.json` automáticamente vía
el sidecar.

⚠️ **Excepción crítica — `diarization.rttm` no se invalida automáticamente**
si cambiás el **conteo** de `--speakers` (ej. 4 → 5 nombres). El RTTM
viejo se reutiliza silenciosamente (en `run` y `recognize`). **Borralo
manualmente**:

```bash
rm audios/.reunion/diarization.rttm    # Linux/Mac/Git Bash
del audios\.reunion\diarization.rttm   # Windows cmd
```

Si solo tocás threshold/min_margin/samples, no hace falta borrarlo.

---

## 9. Gotchas comunes

- **El VTT inicial sale con `[SPEAKER_XX]` crudo aunque uses
  `--voices-folder`** — eso es esperado. Para nombres reales, escribí
  `speaker_map.json` y corré `confirm`. Ver sección 3 paso 4-5.
- **El VTT sale como `<stem>_limpio.vtt` aunque NO pases `--compress`**.
  Buscalo al lado del audio, no en `output/`.
- **`--speakers` con nombres inexistentes es silencioso**: si pasás un
  nombre que no tiene subcarpeta en `voices/`, el CLI no falla — el
  speaker queda como candidato sin embedding en el `speaker_map_suggestions.json`.
  Verificá nombres con `ls voices/` antes de correr.
- **`recognize` sin haber corrido `run` antes** falla — necesita
  `diarization.rttm` cacheado.
- **`confirm` sin `speaker_map.json` previo** falla con mensaje claro —
  tenés que escribir el map vos basado en `speaker_map_suggestions.json`.
- **Samples en contexto acústico muy distinto al audio**: error
  silencioso más común. Ver sección 5.

---

## 10. Troubleshooting

| Síntoma | Fix |
|---|---|
| `HF_TOKEN not set` | `export HF_TOKEN=hf_...` o `--token hf_...` |
| `401 Unauthorized` bajando pyannote | Aceptar licencias (sección 2) |
| `WinError 1314` (Windows) | Correr como Administrador la primera vez (Avanzado) |
| Output con `SPEAKER_XX` esperando nombres | Es esperado tras `run`. Escribí `speaker_map.json` y corré `confirm` (sección 3) |
| Transcripción con alucinaciones | Verificar `--language`; si es muy ruidoso, `--model medium` |
| Speaker nunca matchea (score <0.40) | Reenrolar con samples del mismo contexto acústico |
| Diarización parte un turno en dos | Pasar `--speakers` con conteo exacto |

Para empezar de cero borrá la cache: `rm -rf audios/.reunion/` (o
`rmdir /s /q audios\.reunion` en Windows cmd).

---

## 11. Defaults relevantes

Hardcoded en `speechlib/speaker_recognition.py`:

- `SPEAKER_SIMILARITY_THRESHOLD = 0.55` — score mínimo para que
  `recommended` no sea `null` en el suggestions JSON. **Ya no afecta el
  VTT** (el VTT crudo sale igual con o sin match recommended).
- `SPEAKER_SIMILARITY_MIN_MARGIN = 0.10` — diferencia top1-top2 mínima
  para tomar `recommended`. Mismo scope: solo afecta el campo
  `recommended` de las sugerencias, no el VTT.
- `MIN_SEGMENT_DURATION_S = 0.5` — turnos pyannote más cortos se ignoran
  para computar embeddings (el segmento sigue en el VTT con tag crudo).

Los dos primeros no tienen flag CLI — editar el archivo es la única
forma. Como ya no se aplican al VTT automáticamente, su impacto es
limitado a la "recomendación" que ves en el suggestions JSON. Si vos
sabés mejor, ignorá el `recommended` y escribí tu propio
`speaker_map.json`.

**Dónde encontrar el archivo**:
- **Dev install** (`pip install -e .` desde un clone): en el repo
  clonado, ej. `~/speechlib/speechlib/speaker_recognition.py`.
- **Install normal** (`pip install speechlib`): en
  `site-packages/speechlib/speaker_recognition.py`. Para encontrarlo:
  `python -c "import speechlib; print(speechlib.__file__)"` y mirá
  el directorio al lado.

**Reversible** con `git checkout` (dev install) o
`pip install --force-reinstall speechlib` (install normal). Cambio
global al environment Python — afecta todas las corridas.

---

## Avanzado

### Subdirectorio `_enhanced/` en la voice library

Dentro de cada carpeta de speaker, un subdirectorio opcional `_enhanced/`.
Si existe y tiene samples, se usa **en lugar de** los samples de la raíz
(no se mezclan). Si está vacío, fallback a la raíz. Los creás vos
(speechlib no los genera) — sirve para tener una versión denoised
conviviendo con la original.

### Gotchas de Windows

- **Administrador solo para la primera descarga**: HuggingFace crea
  symlinks en `%USERPROFILE%\.cache\huggingface\`. Si aparece
  `WinError 1314`, abrí terminal como Administrador y corré. Después de
  la primera corrida exitosa, las siguientes NO requieren Administrador.
- **`torchcodec` y `torchaudio 2.10+`**: ya mitigados vía `compat.py`,
  no hay que hacer nada.
- **Paths con espacios**: usá comillas dobles siempre
  (`"C:/audios/Reunión.m4a"`).

### Formatos de audio

El CLI usa ffmpeg internamente — cualquier formato que ffmpeg entiende
funciona (`.m4a`, `.mp3`, `.wav`, `.ogg`, `.opus`, etc.). **ffmpeg no
viene con speechlib**, tiene que estar en el PATH (`apt install ffmpeg`,
`brew install ffmpeg`, `winget install ffmpeg`).
