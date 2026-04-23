# Lecciones Aprendidas — Sesión 2026-04-23

## Qué es speechlib

Pipeline Python (repo en `c:\workspace\dev\speechlib`, branch
`refactor/speaker-domain`) que transcribe audio con diarización y
reconocimiento de speakers. Stack: **faster-whisper (ASR) + pyannote
(diarización) + speechbrain (speaker embeddings)**. CLI: `speechlib
run <audio>` (Typer). Esta sesión extendió el CLI con `--hotwords` y
comparó empíricamente vs. AssemblyAI para es-CL.

## Glosario

- **hotwords** (faster-whisper): string inyectado como logit bias en la
  decodificación de Whisper. Distinto de `initial_prompt` (contexto).
- **keyterms_prompt** (AssemblyAI): sesgo de términos con `speech_model=best`.
  Exclusivo de inglés.
- **word_boost** (AssemblyAI): fallback multiidioma. Lista + `boost_param`
  (low/default/high).
- **es-CL**: español chileno (nombres propios, chilenismos como "cachai").
- **loudnorm**: filtro FFmpeg EBU R128 (LUFS + true peak).

## Intención de la sesión

Procesar 1 video + 2 audios (~52 min total) de una reunión
Leansight–Aguas Andinas vía AAI. Al aparecer errores léxicos obstinados
es-CL, ejecutar el experimento E1 del plan de exploración: implementar
`--hotwords` en speechlib y comparar vs. AAI `word_boost` sobre los
mismos archivos normalizados.

## Qué funcionó

### 1. Compresión/normalización FFmpeg según la guía

- **Comando video** (mono, 16 kHz, loudnorm −14 LUFS; ver
  `devdocs/guia_compresion.md`):
  ```
  -c:v libx264 -crf 28 -preset medium -c:a aac -ac 1 -b:a 64k -ar 16000
  -af loudnorm=I=-14:TP=-1.0:LRA=9 -movflags +faststart
  ```
  Audios análogos con `-c:a aac -b:a 96k -af loudnorm...`. `ffprobe` NO
  devolvió `side_data` con `rotation: -90` → sin `-vf transpose=2`.
- **Resultado**: Video 675 MB → 58 MB (−91%, 2m59s); Audios 33 MB → 26 MB
  (−22%) y 10 MB → 8 MB (−23%).

### 2. Feature slice `--hotwords` en speechlib (TDD)

- **Problema**: faster-whisper acepta `hotwords` en
  `BatchedInferencePipeline.transcribe`, pero speechlib no lo exponía
  (experimento E1 de `devdocs/plan_exploracion_transcripcion.md`).
- **Solución**: test de aceptación primero
  (`tests/test_acceptance_hotwords.py`), propagación paralela al patrón
  de `--initial-prompt` en 4 sitios (`speechlib/transcribe.py`,
  `core_analysis.py` ×2, `__main__.py`).
- **Resultado**: suite completa 433/433. Commits atómicos:
  - `1c8e04f feat(transcribe): --hotwords CLI flag as logit-bias alternative`
  - `74fca53 fix(transcribe): hotwords se junta a string antes de batched.transcribe`

### 3. Ejecución real speechlib + hotwords (GPU RTX 2070 Super Max-Q)

Comando:
```
HOTWORDS=$(grep -v '^#\|^$' devdocs/aguas_andinas_keyterms.txt | paste -sd, -)
speechlib run <audio> --language es --model large-v3-turbo \
    --hotwords "$HOTWORDS" --verbose
```
49 keyterms (participantes + stack Esri/GIS + jerga + chilenismos).
Wallclock (breakdown del `=== Step Timer Report ===` con `--verbose`):

- Video 5.5 min → 20s (preprocess 0.6s, diarize 10.7s, transcribe 8.1s).
- Audio 35.7 min → 134s (preprocess 15.1s, diarize 69.9s,
  transcribe 39.1s, output 9.7s).
- Audio 11 min → ~60s (breakdown no archivado).

Sin GPU el pipeline funciona (faster-whisper cae a CPU int8 automático)
pero el budget no fue medido en esta sesión.

### 4. Corrección de errores léxicos es-CL (evidencia empírica)

Setup: mismo archivo `Voz 260423_104801_norm.m4a` (AAC 96 kbps mono
48 kHz, loudnorm −14 LUFS) consumido por las 3 pasadas. SDK:
`assemblyai==0.54.1`. Keyterms: 49 de `devdocs/aguas_andinas_keyterms.txt`
(mismos para AAI word_boost y speechlib hotwords). Counts extraídos con
`grep -oiE "pattern" file | sort | uniq -c | sort -rn` sobre los
`.txt`/`.vtt`, no lectura manual.

| Error léxico | AAI v1 | AAI v2 (word_boost) | speechlib + hotwords |
|---|---|---|---|
| Pandisi | 1 | 1 | **0** |
| Aguasis | 2 | 2 | **0** |
| WEAP | 1 | 1 | **0** |
| Gira (por Jira) | 1 | 1 | **0** |
| Yera (por Jira) | 1 | 1 | **0** |
| Sarmida (video) | 1 | 1 | **0** |

| Término correcto | AAI | speechlib |
|---|---|---|
| SAP | 27 | **33** (+6) |
| Jira | 2 | **5** (+3) |
| cachai | 2 | **6** (+4) |
| chiquillos | 2 | **5** (+3) |
| GIS (uppercase) | 2 | **3** |

Speechlib capturó "Harald" y "cachai" en el audio de 11 min que ambas
pasadas de AAI perdieron (0→1, 0→1).

## Qué no funcionó

### 1. `keyterms_prompt` en AssemblyAI español

Pattern del prompt existente (`devdocs/assemblyai_prompt.md`) + `speech_model=best`
→ API rechaza: `TranscriptError: keyterms_prompt is only supported for the
following languages when using best speech_model: en, en_au, en_uk, en_us.
Use word_boost instead`. SDK v0.54.1 lo acepta pero server-side falla.
**Lección**: `keyterms_prompt` es exclusivo de inglés; para es, solo
`word_boost`.

### 2. `word_boost` con `boost_param="high"` en es-CL

Hipótesis: 49 keyterms + boost="high" corrigen errores léxicos. Realidad:
`diff v1 v2` del audio largo dio 376 líneas distintas mayormente
puntuación y re-segmentación (176 vs 184 utterances). Counts de errores
léxicos graves (Pandisi, Aguasis, WEAP, Gira, Yera) y de términos
correctos (SAP=27, Jira=2, cachai=2, chiquillos=2) **idénticos** entre v1
y v2. **Lección**: AAI `word_boost` en español cambia fraseo, no léxico
sobre nombres propios chilenos. Para errores léxicos obstinados en es-CL,
AAI no tiene herramienta efectiva en 2026-04-23.

### 3. Test de aceptación con MagicMock tolerante al tipo

- **Hipótesis**: `MagicMock()` sobre `BatchedInferencePipeline` captura el
  contract de `batched.transcribe(hotwords=...)`.
- **Realidad**: el test pasó verde con `hotwords=["Patricio", ...]`
  (lista) porque MagicMock intercepta la llamada sin validar tipos — el
  código real de faster-whisper nunca ejecuta. El test asertaba solo
  *presencia* del kwarg, no tipo. Al correr sin mock,
  `faster_whisper/transcribe.py` hizo `hotwords.strip()` y falló con
  `AttributeError: 'list' object has no attribute 'strip'`.
- **Lección**: duck-typed mocks no validan contract de librerías
  externas. El test debe asertear el tipo post-boundary (`str`
  space-joined en este caso); la conversión vive en el adapter
  `speechlib/transcribe.py:53` (`hotwords_str = " ".join(hotwords) if
  hotwords else None`). Corregido en commit `74fca53`.

## Hallazgos técnicos clave

1. **faster-whisper `hotwords` es `str`, no `list[str]`**. En
   `faster_whisper/transcribe.py:1544-1548` (v1.x), el código real es:
   ```python
   if hotwords and not prefix:
       hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
       if len(hotwords_tokens) >= self.max_length // 2:
           hotwords_tokens = hotwords_tokens[: self.max_length // 2 - 1]
       prompt.extend(hotwords_tokens)
   ```
   El valor se tokeniza y se trunca a `max_length // 2 - 1` cuando
   excede el umbral. Cualquier tipo sin método `.strip()` rompe el call.

2. **AssemblyAI `speech_model=best` + `language_code=es` rechaza
   `keyterms_prompt` server-side**. Mensaje exacto: *"keyterms_prompt is
   only supported for the following languages when using best
   speech_model: en, en_au, en_uk, en_us. Use word_boost instead"*.

3. **`word_boost` (AAI best+es) tiene efecto marginal; `--hotwords`
   (faster-whisper) sesga efectivamente en es-CL**. Las tablas de
   "Qué funcionó #4" son la evidencia: mismo audio, mismos 49 términos,
   AAI v1=v2 en léxico obstinado pero speechlib eliminó los 6 errores y
   mejoró términos correctos.

4. **Budget speechlib en RTX 2070 Super Max-Q + `large-v3-turbo`**:
   ~4× realtime para audio es de ~35 min (134s wallclock: preprocess
   ~15s, diarize ~70s, transcribe ~39s, output ~10s).

## Arquitectura actual del pipeline

```
audio (m4a/mp4)
  → preprocess: convert_to_wav → mono → resample_to_16k → loudnorm
  → diarization (GPU): pyannote speaker-diarization-community-1
  → (opcional) speaker recognition: SpeechBrain ECAPA + voices_folder
  → transcription (GPU): faster-whisper BatchedInferencePipeline
      large-v3-turbo (+ initial_prompt? + hotwords? ← NUEVO, ambos default None)
  → assign_text_to_segments: word-level overlap alignment
  → output: VTT/TXT con tags [SPEAKER_XX]
```

**Propagación `hotwords`**: `CLI --hotwords "csv"` → `_parse_speakers` →
`list[str]` → `core_analysis` → `_transcribe_segments` →
`transcribe_full_aligned` → `" ".join(hotwords)` → `str` →
`BatchedInferencePipeline.transcribe(hotwords=...)`.
