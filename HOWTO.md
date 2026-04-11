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

Librería + CLI Python. Tres pasos integrados:

1. **Diarización** (`pyannote`): "quién habla cuándo" → tags `SPEAKER_00`,
   `SPEAKER_01`, ...
2. **Speaker recognition** (cosine similarity de embeddings
   `pyannote/embedding`, scores 0-1, más alto = más parecido): matchea
   cada tag contra una **voice library** opcional para asignar nombres.
3. **Transcripción** (`faster-whisper`): texto por segmento.

Output: `.vtt` o `.txt` con `[Nombre real] texto...` (o `[SPEAKER_00]`
sin voice library).

---

## 2. Requisitos previos

### Bloqueantes

1. **`HF_TOKEN` de HuggingFace**. Creá cuenta en huggingface.co,
   generá un token en Settings → Access Tokens, y exportalo:
   ```bash
   export HF_TOKEN=hf_xxx    # Linux/Mac/Git Bash
   setx HF_TOKEN hf_xxx      # Windows (reabrir terminal)
   ```
   O pasalo con `--token hf_xxx` en cada llamada.

2. **Aceptar licencias de 2 modelos gated** en HuggingFace (click
   "Agree" logueado):
   - `pyannote/speaker-diarization-community-1`
   - `pyannote/embedding`

   Sin esto, el pipeline falla con `401 Unauthorized`. Para verificar
   antes de invertir tiempo en un audio largo, corré `diagnose` sobre
   un audio corto — si carga los modelos, las licencias están OK.

3. **GPU CUDA + cuDNN**: recomendado (CPU funciona ~10x más lento).
   Testeado con CUDA 11.x + cuDNN 8; CUDA 12.x funciona si tu PyTorch
   es compatible. Verificá con
   `python -c "import torch; print(torch.cuda.is_available())"`.
   VRAM mínima: **6 GB** para `large-v3-turbo` (default), **4 GB** con
   `--model medium`, **2 GB** con `--model small`.

### Opcional

- **Voice library** (`--voices-folder`): solo si querés nombres reales
  en el output en vez de `SPEAKER_00/01/...`. Ver sección 5.

---

## 3. Quick start — transcribir una reunión

El comando mínimo para transcribir un audio en español (default):

```bash
python -m speechlib run "mi_reunion.m4a"
```

Podés correrlo desde cualquier directorio con path relativo o absoluto.

**Qué hace**:
- Preprocesa a 16 kHz + loudnorm.
- Diariza con `pyannote/speaker-diarization-community-1`.
- Transcribe con `faster-whisper` / `large-v3-turbo`.
- Guarda el VTT final en `<audio_dir>/<stem>_limpio.vtt` (ej.
  `Voice 260127.m4a` → `Voice 260127_limpio.vtt`). El sufijo
  `_limpio` es **automático aunque NO pases `--compress`** — es el
  naming canónico del output publicado, no implica que haya habido
  enhancement/compresión (ver §4 y §9).
- Cachea artefactos intermedios en `<audio_dir>/.<stem>/` (oculto),
  incluyendo una copia interna `transcript_<lang>.vtt` que es la
  fuente de la publicación al source folder. Usá `-v` para ver si
  una etapa reusó cache o recomputó.

**Output sin voice library**:
```vtt
WEBVTT

00:00:01.200 --> 00:00:04.500
[SPEAKER_00] Hola a todos, empezamos la reunión.

00:00:04.500 --> 00:00:08.100
[SPEAKER_01] Gracias. ¿Podemos revisar el primer punto?
```

Para que `SPEAKER_00` se reemplace por nombres reales, necesitás una
voice library (sección 5).

**Tiempos aproximados** (audio de 30-60 min):
- Primera corrida (con descarga de modelos): ~15-30 min en GPU, ~45-90 min en CPU.
- Corridas siguientes (modelos cacheados): ~5-15 min en GPU, ~30-60 min en CPU.
- Con `--compress`: sumá ~10-40 min por el enhancement MossFormer2.

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

3. **Revisá el VTT** en `output/transcript_es.vtt`. Si los 4 nombres
   aparecen correctamente, listo. Si un speaker aparece como
   `SPEAKER_02` o con nombre incorrecto, pasá al workflow iterativo
   (sección 7).

---

## 4. Los 3 subcomandos

### `run` — pipeline completo

Corre todo de punta a punta.

```bash
python -m speechlib run <audio_file> [opciones]
```

**Flags principales** (todos opcionales salvo `<audio_file>`):

| Flag | Default | Cuándo usarlo |
|---|---|---|
| `--voices-folder PATH` | — | Voice library. Activa speaker recognition. |
| `--speakers "A,B,C"` | — | Asistentes esperados (nombres exactos de subcarpetas en voices/, **case-sensitive**). Filtra la library y fuerza `num_speakers` en pyannote. Ver sección 6. |
| `--language` | `es` | Código ISO 639-1: `es`, `en`, `fr`, `pt`, etc. |
| `--model` | `large-v3-turbo` | Default está bien. Alternativas: `medium`, `small`, `tiny` (más rápido, menos VRAM). |
| `--output-format` | `vtt` | Alternativa: `txt` (sin timestamps). |
| `--grouping` | `sentences` | `sentences`: agrupa por oración completa (más legible). `timestamps`: un segmento por timestamp crudo de Whisper (más granular). |
| `--compress` | off | Genera **además** `{stem}_limpio.m4a` (AAC 96kbps mono 16kHz) al lado del audio original, para archivo. No afecta la transcripción. **Nota**: el VTT ya usa el sufijo `_limpio.vtt` sin este flag — `--compress` solo agrega el `.m4a`. |
| `--skip-enhance` | off | Salta el enhancement (MossFormer2) antes de comprimir. **Sin efecto si no pasás `--compress`** — el enhance nunca toca la transcripción. |
| `--token hf_xxx` | `$HF_TOKEN` | Solo si no tenés la env var. |
| `-v` / `--verbose` | off | Logs por etapa. Recomendado la primera vez. |

Flags menos frecuentes: `--log-folder`, `--quantization` (int8 para faster-whisper). Corré `python -m speechlib run --help` para la lista completa.

### `recognize` — re-ejecuta solo speaker recognition

Re-computa el mapeo `SPEAKER_XX → nombre` sin re-diarizar ni
re-transcribir.

```bash
python -m speechlib recognize <audio_file> --voices-folder voices/ [--force] [-v]
```

El `<audio_file>` debe ser el **mismo path** que usaste en `run` —
speechlib lo usa para encontrar el cache `.<stem>/`.

**Cuándo usarlo**: cambiaste la voice library, `--speakers`, o editaste
threshold/min_margin.

**`--force`**: el cache se invalida automáticamente cuando cambian los
params (via el sidecar `speaker_map_params.json`). En teoría `--force`
es redundante, pero lo usamos como seguro al iterar — elimina duda
sobre si la invalidación agarró el cambio.

### `diagnose` — matriz de scores (read-only)

No modifica ningún artefacto. Solo imprime (y guarda) la matriz de
cosine similarity de cada `SPEAKER_XX` contra cada voice en la library.

```bash
python -m speechlib diagnose <audio_file> --voices-folder voices/
```

**Cuándo usarlo**: entender por qué un speaker no matchea. `diagnose`
imprime un resumen en terminal **y** guarda el JSON completo en
`<audio_dir>/.<stem>/recognition_diagnostics.json`. Estructura:

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

- **Formato**: WAV (cualquier SR / bit depth). Archivos corruptos se
  saltan con warning, no rompen el pipeline.
- **Nombres case-sensitive**, espacios y acentos preservados.
  `Carla Lopez/` aparece exactamente así en el output.
- **Duración por sample**: **5-30 segundos**, audio limpio. Samples <0.5s
  se ignoran (pyannote no puede generar embedding). Samples de minutos
  son desperdicio.
- **Representatividad crítica** (error silencioso más común): los
  samples deben venir de un contexto acústico **similar al audio a
  transcribir**. Regla práctica: **si podés escuchar la diferencia de
  ambiente en 1 segundo, el embedding también la nota**. Mismo
  dispositivo/plataforma (Zoom↔Zoom) funciona; cruzar muy distintos
  (teléfono 8 kHz vs lavalier 48 kHz) falla.
- **Prefijo `_` ignora el subdir**. Excepción: `_enhanced/` — ver Avanzado.
- **Múltiples samples**: se promedian los embeddings. Más samples
  diversos = embedding más robusto.

---

## 6. `--speakers`: constrained diarization

```bash
python -m speechlib run reunion.m4a \
  --voices-folder voices/ \
  --speakers "Ana,Bruno,Carla Lopez,Diego"
```

**Dos efectos**:

1. **Filtra la voice library**: solo estos speakers se consideran en
   recognition. Elimina falsos positivos contra ausentes.
2. **Fuerza `num_speakers=N` en pyannote**: la diarización produce
   **exactamente** N tags. Si hay físicamente más personas hablando,
   pyannote fusiona al más parecido a otro cluster. Si hay menos,
   parte el que más habla en dos. **El conteo importa**: si no estás
   seguro cuántos hablan, mejor omitir `--speakers` y dejar que
   pyannote infiera.

**Asignación por descarte**: si un tag no matchea contra ninguno de los
speakers con samples, se asigna por orden de **cantidad de segmentos**
— el tag que más habla recibe el primer nombre sin matchear de
`--speakers`, etc. Es determinístico. Esto permite pasar nombres aunque
NO tengas samples de todos: Ana y Bruno matchean vía embeddings, Carla
y Diego (sin samples) se asignan a los SPEAKER_XX restantes por cantidad
de segmentos.

---

## 7. Workflow iterativo con agente — ajustar speaker recognition

Este es el patrón que usa el usuario cuando recognition falla en algún
speaker y quiere guiar a un agente (Claude Code) para arreglarlo sin
repetir el pipeline entero.

### Contexto del problema

Corriste `run` con voice library y `--speakers`. El VTT tiene algunos
`SPEAKER_02` en vez de nombres reales, o asignó un nombre incorrecto.
Ahora querés diagnosticar y ajustar.

> ⚠️ **Trampa antes de iterar**: si cambiás el **conteo** de
> `--speakers` (ej. 4 → 5), `diarization.rttm` NO se invalida
> automáticamente. Borralo a mano antes de re-correr. Si solo tocás
> threshold/min_margin/samples (no el conteo), no hace falta.

### Ciclo de iteración (guided loop)

**Precondición**: ya corriste `run` al menos una vez sobre el audio
para tener `diarization.rttm` en cache. `diagnose` lo reusa, no
diariza de cero.

1. **Inspeccionar**: `python -m speechlib diagnose reunion.m4a
   --voices-folder voices/` — imprime scores en terminal y guarda JSON
   en `<audio_dir>/.<stem>/recognition_diagnostics.json` (formato
   descrito en sección 4).

2. **Usuario guía al agente con input concreto**:

   > "En esta reunión habla Ana Pérez pero el VTT dice SPEAKER_02. Mirá
   > `recognition_diagnostics.json` y decime por qué no matchea."

3. **Agente lee el JSON y detecta el patrón**. Ejemplo: Ana top1 con
   0.5012 (bajo threshold 0.55), margin vs Bruno 0.019 (muy chico).
   **Dos problemas combinados**.

4. **Agente reporta y pregunta antes de actuar**:

   > "Ana top1 con 0.5012 pero bajo threshold; margin vs Bruno 0.019.
   > Threshold=0.48 puede capturarla pero el margin bajo sugiere samples
   > poco distintivos. ¿Probar 0.48 o preferís reenrolar Ana primero?"

5. **Usuario decide**. Si dice "probá 0.48":
   - Agente edita `SPEAKER_SIMILARITY_THRESHOLD = 0.48` en
     `speechlib/speaker_recognition.py` (hoy no hay flag CLI, ver
     sección 11).
   - El cambio se activa automáticamente en la próxima corrida del CLI
     — cada invocación arranca un proceso Python nuevo que re-importa
     el módulo. Excepción: dentro de notebook/REPL persistente hay que
     recargar con `importlib.reload(speechlib.speaker_recognition)`.
   - Corre `recognize --force` para recomputar:
     ```bash
     python -m speechlib recognize reunion.m4a --voices-folder voices/ --force
     ```
     (`--force` es opcional — el cache se invalida automáticamente vía
     el sidecar `speaker_map_params.json` — pero lo usamos como seguro
     al iterar.)
   - Re-corre `diagnose` y verificá que Ana ahora matchee.

6. **Iterar hasta que todos los speakers esperados aparecen**. Si el
   threshold bajo causa falsos positivos en otros, el usuario le dice
   al agente: "ahora Carla aparece en turnos de Diego, revertí
   threshold y reenrolá Carla con samples mejores".

### Reglas del loop

- **Agente muestra la data antes de proponer cambios**. Nunca ajusta
  a ciegas.
- **Usuario decide el trade-off** (bajar threshold captura más pero
  introduce falsos positivos).
- **Agente usa `recognize --force`, no `run`**, para ahorrar minutos.
- **`diagnose` es read-only**: inspeccionar libremente.
- **Si el problema son los samples, no el threshold**, el agente
  debe avisar antes de ajustar params (parchar con threshold cuando
  el sample está mal es frágil).

### Threshold vs min_margin — cuándo tocar cada uno

- **`threshold` (0.55)**: score mínimo absoluto. **Bajalo** si el top1
  del speaker está claramente por encima del resto pero abajo de 0.55
  (ej. 0.48 vs 0.30 y 0.25). **No** si hay competencia cercana.
- **`min_margin` (0.10)**: diferencia top1-top2 mínima. **Bajalo solo**
  en casos raros donde dos voces son genuinamente casi idénticas. Un
  margin chico normalmente significa embeddings mal definidos — aceptar
  esos matches mete ruido.

### Cuándo el problema NO es el threshold (no lo toques, fix upstream)

- **Todos los scores del speaker faltante <0.40**: samples desalineados
  con el contexto acústico del audio. Reenrolar.
- **Top1 inestable entre corridas**: embeddings no determinísticos por
  samples muy cortos o ruidosos. Reenrolar con samples limpios >5s.
- **Score alto contra speaker equivocado** (Ana 0.70 contra Bruno, 0.30
  contra sí misma): los samples de Ana probablemente tienen la voz de
  Bruno contaminada. Revisá y limpiá los samples.
- **Un speaker esperado no aparece en ningún turno del VTT**: pyannote
  no creó cluster para él — problema anterior a recognition. Probar
  sin `--speakers` para ver cuántos clusters detecta libremente.

---

## 8. Cache y artefactos

Cada corrida crea una carpeta `<audio_dir>/.<stem>/` con artefactos
reutilizables. Los que importan:

- **`diarization.rttm`** — output crudo de pyannote (quién habla cuándo).
- **`speaker_map.json`** + **`speaker_map_params.json`** — mapeo
  `SPEAKER_XX → nombre` y sidecar con los params usados.
- **`recognition_diagnostics.json`** — matriz de scores (si hay
  `--voices-folder`).

**Cache invalidation automática**: cambiar `--speakers`, threshold o
min_margin invalida `speaker_map.json` automáticamente vía el sidecar.
Todo el resto se reusa.

⚠️ **Excepción crítica — `diarization.rttm` no se invalida automáticamente**:
si cambiás el **conteo** de `--speakers` (ej. de 4 a 5 nombres),
pyannote necesita re-diarizar con `num_speakers` nuevo, pero el RTTM
viejo se reutiliza silenciosamente. **Borralo manualmente** antes de
re-correr:

```bash
rm audios/.reunion/diarization.rttm    # Linux/Mac/Git Bash
del audios\.reunion\diarization.rttm   # Windows cmd
```

Si cambiás threshold/min_margin/samples pero **no el conteo** de
speakers, no hace falta borrarlo.

**Espacio en disco**: ~100-300 MB por hora de audio (dominado por el
wav 16 kHz resampled).

---

## 9. Gotchas comunes la primera corrida

- **El VTT sale como `<stem>_limpio.vtt` aunque NO pases `--compress`**:
  el sufijo `_limpio` es el naming canónico del output publicado, no
  implica que el audio fue enhanced/comprimido. `--compress` solo
  controla si **además** se genera el `.m4a` comprimido. Si esperabas
  `output/transcript_es.vtt`, ese path no existe — buscá el `.vtt` al
  lado del audio.
- **`--compress` no es gratis**: agrega tiempo sustancial (el enhance
  MossFormer2 corre antes de comprimir). Solo usalo si querés el
  `_limpio.m4a` de archivo.
- **`--model` default está bien**: `large-v3-turbo` es el sweet spot.
  Cambialo solo si te quedás sin VRAM.
- **`--speakers` con nombres inexistentes es silencioso**: si pasás un
  nombre que no tiene subcarpeta en `voices/`, el CLI **no falla** —
  asume speaker sin samples y lo pone en la cola de asignación por
  descarte. Verificá los nombres antes de correr.
- **Samples en contexto acústico muy distinto al audio**: error silencioso
  más común. Ver sección 5.

---

## 10. Troubleshooting

### Errores de setup

| Síntoma | Fix |
|---|---|
| Error con "token" / `HF_TOKEN not set` | `export HF_TOKEN=hf_...` o `--token hf_...` |
| `401 Unauthorized` bajando pyannote | Aceptar licencias en huggingface.co (ver sección 2) |
| `WinError 1314` (Windows) | Correr como Administrador la primera vez (ver Avanzado) |

### Problemas de calidad

| Síntoma | Fix |
|---|---|
| Output con `SPEAKER_XX` esperando nombres | Corré `diagnose` → workflow sección 7 |
| Transcripción con alucinaciones | Verificar `--language`; si es muy ruidoso, `--model medium` |
| Primera corrida tarda mucho | Descarga de modelos; las siguientes son rápidas |
| Speaker nunca matchea (score <0.40) | Reenrolar con samples del mismo contexto acústico |
| Diarización parte un turno en dos | Pasar `--speakers` con conteo exacto |

### Re-correr parcialmente

```bash
# Solo recognition (sin re-diarizar):
python -m speechlib recognize reunion.m4a --voices-folder voices/ --force

# Empezar de cero (borra cache):
rm -rf audios/.reunion/                       # Linux/Mac/Git Bash
rmdir /s /q audios\.reunion                   # Windows cmd
python -m speechlib run reunion.m4a --voices-folder voices/
```

---

## 11. Defaults relevantes

Hardcoded en `speechlib/speaker_recognition.py`:

- `SPEAKER_SIMILARITY_THRESHOLD = 0.55` — score mínimo para aceptar match.
- `SPEAKER_SIMILARITY_MIN_MARGIN = 0.10` — diferencia top1-top2 mínima.
- `MIN_SEGMENT_DURATION_S = 0.5` — turnos más cortos se ignoran **solo
  para recognition** (el segmento sigue en el VTT con su tag
  `SPEAKER_XX` crudo).

Los dos primeros no tienen flag CLI — editar el archivo es la única
forma de cambiarlos. Cambio global (afecta todas las corridas). Deuda
técnica conocida.

---

## 12. Para profundizar

- `README.md`: API legacy de la clase `Transcriptor` (no cubierta acá).
- `speechlib/speaker_recognition.py`: defaults editables
  (`SPEAKER_SIMILARITY_THRESHOLD`, `SPEAKER_SIMILARITY_MIN_MARGIN`).
- `speechlib/__main__.py`: definición del CLI con Typer.
- `devdocs/lessons_learned_2026_04_10.md`: hallazgos técnicos validados,
  experimentos A/B, deuda técnica pendiente.

---

## Avanzado

Secciones para casos menos frecuentes. La mayoría de usuarios no las
necesita al arrancar.

### Subdirectorio `_enhanced/` en la voice library

Dentro de cada carpeta de speaker podés tener un subdirectorio opcional
`_enhanced/`. Si existe y tiene samples, se usa **en lugar de** los
samples de la raíz del speaker (no se mezclan). Si existe pero está
vacío, fallback a la raíz. **Esos samples los creás vos** — speechlib
no los genera. Sirve para tener una versión "limpia" del enrolment
(post-procesada con tu herramienta favorita de denoise) conviviendo
con la original, sin tener que elegir entre las dos.

### Gotchas de Windows

- **Administrador solo para la primera descarga**: cuando HuggingFace
  baja los modelos por primera vez, crea symlinks en
  `%USERPROFILE%\.cache\huggingface\`. Si aparece `WinError 1314`
  (privilegio denegado), cerrá la terminal, abrí una nueva como
  Administrador, y corré. Después de esa primera corrida exitosa, los
  modelos quedan en disco y las siguientes corridas NO requieren
  Administrador (ni siquiera después de reiniciar la máquina).
- **`torchcodec` falla en Windows CPU-only** (`libtorchcodec_core*.dll
  WinError 127`). Ya mitigado automáticamente vía `compat.py` con shim
  PyAV + `wave` stdlib — no hay que hacer nada.
- **`torchaudio 2.10+`**: `list_audio_backends` removido pero
  SpeechBrain 1.0.3 lo usa. Ya mitigado con shim en `compat.py`.
- **Paths con espacios**: comillas dobles en cmd/PowerShell:
  `python -m speechlib run "C:/audios/Reunión de equipo.m4a"`. Comillas
  simples funcionan en Git Bash pero no en cmd — usá siempre dobles.

### Formatos de audio

El CLI usa ffmpeg internamente, así que cualquier formato que ffmpeg
entiende funciona (`.m4a`, `.mp3`, `.wav`, `.ogg`, `.opus`, `.webm`,
`.mp4`, etc.). **ffmpeg no viene con speechlib** — tiene que estar
instalado y en el PATH. Linux/Mac: `apt install ffmpeg` o
`brew install ffmpeg`. Windows: `choco install ffmpeg` o
`winget install ffmpeg`.
