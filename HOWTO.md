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

Librería + CLI Python. Tres pasos: diarización (`pyannote`) → speaker
recognition (cosine similarity de embeddings contra una voice library
opcional) → transcripción (`faster-whisper`). Output: `.vtt` o `.txt`
con `[Nombre real] texto...` (o `[SPEAKER_00]` sin voice library).

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

- **Voice library** (`--voices-folder`): para nombres reales en el
  output en vez de `SPEAKER_00/01/...`. Ver sección 5.

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

### Flujo recomendado la primera vez (con voice library parcial)

Caso típico: tenés samples de **algunos** asistentes pero no todos.

1. **Creá la carpeta `voices/`** en cualquier lado (raíz del proyecto,
   home, donde te convenga). Path relativo o absoluto — da igual.
   Adentro, un subdirectorio por speaker con los samples que tengas:

   ```
   voices/
   ├── Ana/
   │   └── enrolment.wav
   └── Bruno/
       └── clip1.wav
   ```

2. **Corré `run` con `--speakers` listando TODOS los asistentes** (con
   samples y sin samples), en el orden del que más habla primero si lo
   sabés:

   ```bash
   python -m speechlib run reunion.m4a \
     --voices-folder ./voices \
     --speakers "Ana,Bruno,Carla Lopez,Diego" \
     -v
   ```

   Ana y Bruno matchean por embedding; Carla y Diego (sin samples) se
   asignan por descarte a los `SPEAKER_XX` restantes por cantidad de
   segmentos (ver sección 6).

3. **Revisá el VTT** en `<audio_dir>/<stem>_limpio.vtt` (ej.
   `audios/reunion_limpio.vtt` si el audio era `audios/reunion.m4a`).
   Si los 4 nombres aparecen correctamente, listo. Si un speaker
   aparece como `SPEAKER_02` o con nombre incorrecto, pasá al workflow
   iterativo (sección 7).

---

## 4. Los 3 subcomandos

### `run` — pipeline completo

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

### `recognize` — re-ejecuta solo speaker recognition

Re-computa el mapeo `SPEAKER_XX → nombre` sin re-diarizar ni
re-transcribir.

```bash
python -m speechlib recognize <audio_file> --voices-folder voices/ [--force]
```

Usá el **mismo path** que en `run` (lo necesita para el cache `.<stem>/`).

**Cuándo**: cambiaste la voice library, `--speakers`, o threshold/min_margin.
`--force` es redundante en teoría (el cache se invalida automáticamente vía
`speaker_map_params.json`) pero lo usamos como seguro al iterar.

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

## 7. Workflow iterativo con agente — ajustar speaker recognition

Este es el patrón cuando recognition falla en algún speaker y querés
arreglarlo sin repetir el pipeline entero.

**Roles en el loop**: **vos** corrés los comandos del CLI (`diagnose`,
`recognize --force`). El **"agente"** (Claude Code u otro LLM agente)
interpreta los scores, propone cambios de threshold, y edita
`speaker_recognition.py` por vos.

**Punto de entrada concreto**: no hay un comando `speechlib agent` —
abrís Claude Code en la terminal del proyecto, corrés `diagnose` en
otra ventana, y le pegás el output al agente (o le pedís "leé
`<audio_dir>/.<stem>/recognition_diagnostics.json`"). Vos y el agente
coordinan turnándose, no hay automación del lado del CLI.

**Cuándo empieza**: después de una corrida de `run` donde ves un
speaker con nombre incorrecto o `SPEAKER_XX` esperando nombre.
**Cuándo termina**: cuando tu VTT tiene todos los nombres correctos
o decidís que el fix requiere reenrolar (no params).

> ⚠️ **Trampa antes de iterar**: si cambiás el **conteo** de
> `--speakers` (ej. 4 → 5), `diarization.rttm` NO se invalida
> automáticamente. Borralo a mano. Si solo tocás threshold/min_margin/samples,
> no hace falta.

### Ciclo de iteración

**Precondición**: ya corriste `run` al menos una vez (para que
`diarization.rttm` esté en cache; `diagnose` lo reusa).

1. **Inspeccionar**: `python -m speechlib diagnose reunion.m4a --voices-folder voices/`.
2. **Usuario guía al agente**: "Ana Pérez aparece como SPEAKER_02, mirá
   el JSON y decime por qué no matchea."
3. **Agente lee el JSON, detecta el patrón, reporta y pregunta antes de
   actuar**: "Ana top1=0.5012 bajo threshold; margin vs Bruno=0.019.
   ¿Bajar threshold a 0.48 o reenrolar Ana primero?"
4. **Usuario decide**. Si baja threshold: agente edita
   `SPEAKER_SIMILARITY_THRESHOLD` en `speechlib/speaker_recognition.py`
   (no hay flag CLI — ver sección 11). El cambio se activa en la
   próxima corrida (cada invocación es proceso nuevo que re-importa).
5. **Re-correr** `recognize --force` y `diagnose` para verificar.
6. **Iterar** hasta que todos los speakers esperados aparezcan.

**Reglas**: agente muestra data antes de proponer; usuario decide el
trade-off; usar `recognize --force` (no `run`) para iterar; si el
problema son los samples (no threshold), reenrolar antes de tocar params.

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
`diarization.rttm` (output crudo de pyannote), `speaker_map.json` +
`speaker_map_params.json` (mapeo + sidecar de params), y
`recognition_diagnostics.json` (matriz de scores). Cambiar `--speakers`,
threshold o min_margin invalida `speaker_map.json` automáticamente vía
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

- **El VTT sale como `<stem>_limpio.vtt` aunque NO pases `--compress`**.
  Buscalo al lado del audio, no en `output/`.
- **`--speakers` con nombres inexistentes es silencioso**: si pasás un
  nombre que no tiene subcarpeta en `voices/`, el CLI no falla — lo
  pone en cola de asignación por descarte. Verificá nombres con
  `ls voices/` antes de correr, o mirá `recognition_diagnostics.json`
  después (los speakers con embeddings aparecen como keys).
- **`recognize` sin haber corrido `run` antes** falla — necesita
  `diarization.rttm` cacheado. Corré `run` al menos una vez primero.
- **Samples en contexto acústico muy distinto al audio**: error
  silencioso más común. Ver sección 5.

---

## 10. Troubleshooting

| Síntoma | Fix |
|---|---|
| `HF_TOKEN not set` | `export HF_TOKEN=hf_...` o `--token hf_...` |
| `401 Unauthorized` bajando pyannote | Aceptar licencias (sección 2) |
| `WinError 1314` (Windows) | Correr como Administrador la primera vez (Avanzado) |
| Output con `SPEAKER_XX` esperando nombres | `diagnose` → workflow sección 7 |
| Transcripción con alucinaciones | Verificar `--language`; si es muy ruidoso, `--model medium` |
| Speaker nunca matchea (score <0.40) | Reenrolar con samples del mismo contexto acústico |
| Diarización parte un turno en dos | Pasar `--speakers` con conteo exacto |

Para empezar de cero borrá la cache: `rm -rf audios/.reunion/` (o
`rmdir /s /q audios\.reunion` en Windows cmd).

---

## 11. Defaults relevantes

Hardcoded en `speechlib/speaker_recognition.py`:

- `SPEAKER_SIMILARITY_THRESHOLD = 0.55` — score mínimo para match.
- `SPEAKER_SIMILARITY_MIN_MARGIN = 0.10` — diferencia top1-top2 mínima.
- `MIN_SEGMENT_DURATION_S = 0.5` — turnos más cortos se ignoran **solo
  para recognition** (el segmento sigue en el VTT con tag crudo).

Los dos primeros no tienen flag CLI — editar el archivo es la única
forma. Deuda técnica conocida.

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
