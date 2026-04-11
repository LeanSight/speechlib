# Lecciones Aprendidas — Sesión 2026-04-11 — Skills vs Hooks en Claude Code

## Qué es

Lecciones sobre **disciplina de ejecución de skills en Claude Code** y sobre
cuándo subir a hooks para lograr enforcement determinista. El contexto
práctico fue una sesión de trabajo en speechlib (refactor/speaker-domain)
que invocó el skill `/bdd` (Behavior-First TDD) para un cambio de pipeline,
pero el aprendizaje central es meta: qué hace falta para que Claude Code
ejecute un skill sin bailout.

## Glosario

- **Skill**: construct de Claude Code. Vive en `~/.claude/skills/<nombre>/SKILL.md`.
  Describe un workflow con reglas, fases, principios. Se invoca con `/nombre` o
  vía Skill tool. **Advisory** — el modelo puede desviarse.
- **Hook**: construct de Claude Code. Vive en `~/.claude/settings.json` bajo
  `"hooks": {...}`. Script shell que se ejecuta en respuesta a un evento (ej.
  `PreToolUse`). **Determinista** — el modelo no puede desviarse.
- **`/bdd`**: skill personalizado en `~/.claude/skills/bdd/SKILL.md`. Behavior-First
  TDD — fusión de ATDD + TDD + GOOS-sin-mocks. Reglas "no-negociables"
  explícitas en el SKILL.md.
- **`/tidy-first`**: skill personalizado en `~/.claude/skills/tidy-first/SKILL.md`.
  Técnicas de Kent Beck's *Tidy First?*. `/bdd` Fase 0 invoca `/tidy-first`
  incondicionalmente si hay código previo.
- **PreToolUse / PostToolUse / SessionStart**: eventos de hook específicos.
  `PreToolUse` es el único que puede **bloquear** una acción (exit code 2).

## Intención de la sesión

Cambiar el pipeline de speechlib: en vez de que el speaker recognition aplique
automáticamente los nombres al VTT, queremos que sugiera matches y el usuario
confirme antes de publicar. Cambio de interacción fully-auto → human-in-the-loop.

El usuario pidió usar `/bdd` **mandatoriamente** — disciplina TDD estricta,
no atajos.

## Qué funcionó

### 1. La Fase 0 de /bdd capturó una regresión dead-code pre-existente

La regla "tests deben estar en GREEN antes de tidy-first" forzó correr la
suite antes de cualquier cambio. Un test acceptance
(`test_compress_runs_parallel_with_diarization`) estaba en RED estable con
overhead 2.8s vs threshold 2.0s. **Baseline silenciosamente roto desde hace
varios commits**.

- **Root cause**: en commit `acdf328 feat(enhance): move enhance to
  post-processing for output only`, la compresión se movió al final del
  pipeline y el helper `_start_compress_thread` en
  `speechlib/core_analysis.py:520` quedó **dead code** (definido, sin
  callers).
- **Fix** (commit `fe47a8d`): cablear `_start_compress_thread` después del
  preprocess, `thread.join()` antes de `_publish_to_source_folder`, eliminar
  la llamada serial en el branch `elif compress` de
  `speechlib/core_analysis.py:621`.
- **Resultado**: 408 → 411 passed, 0 failed. Paralelización CPU/GPU restaurada.

**Lección indirecta**: sin la ceremonia del skill, esa regresión hubiera
seguido en RED semi-oculta. La disciplina paga en visibilidad.

### 2. Investigación concreta sobre disciplina de skills

Investigación vía subagente `claude-code-guide` + web search (2026-04-11)
retornó hallazgos oficiales confirmados con fuentes:

- [Hooks reference — code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks)
- [Hooks guide — code.claude.com/docs/en/hooks-guide.md](https://code.claude.com/docs/en/hooks-guide.md)
- [Agent Skills overview — platform.claude.com](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)

Conclusión: **skills son model-controlled (advisory), hooks son la única
primitiva determinista**. La doc oficial lo formula como "Hooks are mandatory.
CLAUDE.md is advisory. Skills are model-controlled."

### 3. Mapping concreto regla /bdd → hook event

Para cada regla no-negociable de /bdd, se identificó el hook que la enforza:

| Regla /bdd | Hook event | Enforcement real |
|---|---|---|
| Tests GREEN antes de editar `src/` | `PreToolUse` matcher=`Edit\|Write` | `pytest -x` gate, bloquea con exit 2 |
| Commit coherente (no mezclar tidy+feature) | `PreToolUse` matcher=`Bash` + `if: "Bash(git commit *)"` | analizar `git diff --cached` staged |
| Audit log de ediciones | `PostToolUse` matcher=`Edit\|Write` | append a `.claude/edit-log.jsonl` |
| Recordatorio de "behavior en curso" | `SessionStart` matcher=`startup\|resume` | leer `.claude/current-behavior.txt` |

Implementaciones JSON listas para copypastear quedaron en el chat de la
sesión, con scripts bash concretos en `.claude/hooks/`.

## Qué no funcionó

### 1. Ofrecer un menú de opciones cuando el skill mandaba acción única

**Hipótesis (implícita al momento)**: cuando encuentro un obstáculo fuera
del scope del behavior change actual (regresión de un test de compresión,
no de speaker recognition), debo preguntar al usuario si quiere desviarse
para arreglarlo.

**Realidad**: el skill `/bdd` tiene regla explícita para este caso: "si
es regresión, corregir la producción". La regla estaba escrita, el usuario
había invocado /bdd mandatoriamente, y aun así presenté un menú de 3
opciones ("fix ahora / defer / abandonar ceremonia") y pedí decisión.

**Pushback del usuario**: "no entiendpo porque me ofreces 3 caminos cuando
te indiqué que debías usar /bdd".

**Lección**: cuando el usuario invoca un skill mandatoriamente y el skill
tiene reglas determinísticas para una situación, **ejecutar la regla sin
menú**. El menú es un override del contrato explícito del usuario y
contradice la decisión que ya tomó al invocar el skill.

Causa probable del sesgo: el system prompt de Claude Code incluye guidance
general como "executing actions with care" (pedí confirmación antes de
operaciones "hard to reverse"). Ese default conservador se filtra a la
ejecución de skills incluso cuando el skill lo contradice con reglas
explícitas. Arreglar un test en RED no es destructivo ni difícil de
revertir — el menú fue espurio.

### 2. Primera investigación mal dirigida: buscar Kent Beck en vez de Claude Code

Cuando el usuario pidió "averigua con claude guide y expertos por qué
pasa esto", mi primer reflejo fue buscar Kent Beck + Tidy First + TDD
discipline. Eso era **irrelevante** — el usuario preguntaba por el
comportamiento de *Claude Code* frente al skill, no por la metodología
TDD que el skill practica.

**Lección**: distinguir **meta-research** (sobre el sistema que ejecuta
la tarea — Claude Code) de **domain-research** (sobre el contenido de la
tarea — TDD). Si la pregunta es "por qué Claude Code hace X", la fuente
es doc de Claude Code / Anthropic, no literatura del dominio.

## Hallazgos técnicos clave

1. **Skills en Claude Code son model-controlled (advisory)**. Solo hooks
   son determinísticos. Fuente:
   [Hooks reference](https://code.claude.com/docs/en/hooks),
   "Hooks are mandatory. CLAUDE.md is advisory."

2. **No existe flag en `settings.json` para forzar obediencia estricta de
   skills**. Verificado vía búsqueda en doc oficial 2026-04-11. El máximo
   de señal disponible dentro del skill es language explícito ("DO NOT ask",
   "DO NOT offer alternatives") — no garantiza, solo maximiza probabilidad.

3. **Solo `PreToolUse` puede BLOQUEAR acciones** con `exit 2`. Otros hook
   events (`PostToolUse`, `Stop`, etc) son post-ejecución o informativos.
   Esto determina la arquitectura de enforcement: todo gating vive en
   `PreToolUse`.

4. **Hooks reciben JSON por stdin** con `tool_name`, `tool_input`, `cwd`,
   `session_id`. Responden con exit code o JSON estructurado (formato
   `{"hookSpecificOutput": {"hookEventName": "PreToolUse",
   "permissionDecision": "deny", "permissionDecisionReason": "..."}}`).

5. **Hooks NO tienen estado de sesión persistente nativo**. Solo
   `CLAUDE_ENV_FILE` (env vars inyectadas por `SessionStart`) y archivos en
   disco bajo `.claude/`. Reglas como "one behavior at a time" requieren
   workaround: archivo `.claude/current-behavior.txt` escrito por el modelo
   y leído por hooks — disciplina del modelo sigue siendo necesaria.

6. **Hooks NO pueden enforzar reglas semánticas**:
   - RED legítimo vs espurio (hook solo ve exit code de pytest).
   - Dominio vs adaptador (regex sobre paths es aproximado, no verifica pureza real).
   - Clasificación de test como AT vs unit (no se infiere del nombre).

7. **Dead code en `speechlib/core_analysis.py:520`**
   (`_start_compress_thread`) permaneció orphan desde commit `acdf328` sin
   ser detectado hasta que una regla de skill forzó correr la suite de
   tests completa. Los tests acceptance largos no se corrían en cada PR
   aparentemente.

## Arquitectura actual del pipeline speechlib (post-fix)

```
core_analysis(file_name, compress, skip_enhance, ...)
  │
  ├── _preprocess_audio(state)   # convert → mono → re_encode → resample → loudnorm
  │
  ├── [compress && skip_enhance] _start_compress_thread(state) ──┐
  │                                                              │
  ├── _run_diarization_cached(state)   [GPU]                     │ paralelo
  │                                                              │ (CPU vs GPU
  ├── _compute_embeddings (si hay voices_folder)                 │  sin contienen)
  │                                                              │
  ├── _run_speaker_recognition_cached → speaker_map.json         │
  │                                                              │
  ├── _transcribe_segments (faster-whisper)   [GPU]              │
  │                                                              │
  ├── write_log_file                                             │
  │                                                              │
  ├── _publish_domain_artifacts (transcript.json + samples/)     │
  │                                                              │
  ├── [compress && skip_enhance] compress_thread.join()  ────────┘
  │
  └── _publish_to_source_folder (<stem>_limpio.vtt)
```

## Plan futuro

### Pendiente de la sesión

- **Continuar /bdd Fase 0 (tidy-first)**: tidy 2 (normalize `os.path` → `Path`
  en `_compute_averaged_embeddings_per_tag` de `core_analysis.py`) y tidy 3
  (extract explaining variable para chunk path en el mismo function). Hecho =
  ambos commits `refactor(tidy): ...` en verde.
- **/bdd Fase 1 ya cerrada**: diseño del cambio `run` → suggest + `confirm`
  subcomando. Sin backdoor `--auto-confirm`. Cluster unmatched queda como
  `SPEAKER_XX` raw. Suggestions JSON con top-3 candidatos.
- **/bdd Fase 2+**: escribir AT en RED para el nuevo behavior, luego TDD
  inside-out hasta Fase 6 (commit del behavior).

### Features pendientes

- **Instalar hooks de enforcement** para /bdd:
  - #1 PreToolUse GREEN gate (el más valioso, con refinamiento: permitir edit
    durante fase RED si el diff toca también un test).
  - #3 PostToolUse audit log (cero overhead, gran valor retrospectivo).
  - #2 SessionStart reminder (marginal, skip).
- **Agregar sección "Execution discipline" a `/bdd` SKILL.md** con language
  explícito tipo "DO NOT ask / DO NOT offer alternatives". Maximiza señal.

### Deuda técnica

- `/bdd` y `/tidy-first` no tienen mecanismo determinista para verificar que
  tests están en GREEN antes de empezar — descargan la verificación en el
  modelo, que puede olvidarse.
- No hay estado persistente entre turnos para "one behavior at a time". El
  modelo debe leer/escribir `.claude/current-behavior.txt` manualmente.
- La doc de Claude Code en `~/.claude/CLAUDE.md` (user's private global) no
  menciona el trade-off skills vs hooks. Un lector nuevo del sistema no tiene
  forma de saber cuándo subir a hooks.

## Métricas de la sesión

- Tests: 408 → **411 passed** (0 failed, 3 skipped).
- Commits relevantes de esta sesión: `fe47a8d` (fix parallel compression),
  `02a3153` (tidy extract helper `_resolve_working_path_from_cache`).
- Rondas de research: 2. La primera mal dirigida (Kent Beck TDD), la segunda
  correcta (Claude Code skills + hooks).
- Dead code identificado y arreglado: 1 función (`_start_compress_thread`).
- Behavior changes articulados: 1 (`run` → suggest/confirm), pendiente de
  implementación.
- Duración de la fase de aprendizaje meta sobre skill discipline: ~30 min de
  conversación antes de retomar el trabajo real.
