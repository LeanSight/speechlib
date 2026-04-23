# AXIOMAS DEL PROYECTO
Última destilación: 2026-04-23 (sesión: --hotwords slice + AAI vs faster-whisper es-CL)
Branch: refactor/speaker-domain

## Axiomas del dominio

- Claude Code skills son **model-controlled (advisory)**. Solo hooks son deterministas. Fuente: [code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks) — "Hooks are mandatory. CLAUDE.md is advisory."
- Solo `PreToolUse` hook event puede **bloquear** acciones (`exit 2`). Los demás son informativos o post-ejecución y no pueden revocar.
- Hooks reciben JSON por stdin (`tool_name`, `tool_input`, `cwd`, `session_id`) y responden con exit code (0=continuar, 2=bloquear) o JSON estructurado con `permissionDecision: "deny"` y razón.
- Hooks NO tienen estado de sesión persistente nativo. Solo `CLAUDE_ENV_FILE` (vars inyectadas por `SessionStart`) + archivos en disco bajo `.claude/`.
- El system prompt default de Claude Code incluye "executing actions with care" que genera sesgo conservador a pedir confirmación, y ese sesgo se filtra a la ejecución de skills incluso cuando el SKILL.md tiene reglas "no-negociables" escritas.
- `faster_whisper.BatchedInferencePipeline.transcribe(hotwords=…)` espera `str`, no `list[str]`. Internamente hace `tokenizer.encode(" " + hotwords.strip())` y trunca a `max_length // 2` tokens. Evidencia: `faster_whisper/transcribe.py:1545`.
- AssemblyAI `speech_model=best` + `language_code=es` **rechaza `keyterms_prompt` server-side**. Solo soportado en `en, en_au, en_uk, en_us`. El SDK acepta el parámetro pero el API devuelve `TranscriptError: "…Use word_boost instead"`. Verificado 2026-04-23 con assemblyai SDK 0.54.1.
- AssemblyAI `word_boost` con `boost_param="high"` en es-CL tiene **efecto casi nulo sobre errores léxicos obstinados** (nombres propios chilenos y jerga). Verificado empíricamente: 6 errores idénticos entre v1 (sin boost) y v2 (boost="high", 49 terms curados) sobre el mismo audio 35.7 min es-CL.
- faster-whisper + `--hotwords` sí sesga efectivamente ASR en es-CL: sobre los mismos 49 términos y el mismo audio, eliminó los 6 errores léxicos obstinados (Pandisi, Aguasis×2, WEAP, Gira, Yera) y aumentó conteos de términos correctos (SAP 27→33, Jira 2→5, cachai 2→6, chiquillos 2→5).

## Decisiones activas

| Decisión | Por qué irreducible | Invalida si |
|---|---|---|
| /bdd Fase 0 obligatorio aunque haya que arreglar regresiones fuera del scope del behavior actual | La disciplina capturó dead code oculto (`_start_compress_thread` orphan desde commit `acdf328`) que no se veía sin forzar la suite completa | El fix requiere días de side quest o el usuario explícitamente releva la ceremonia |
| `compress + skip_enhance` corre thread-parallel con diarize en core_analysis.py | compress es CPU-bound (ffmpeg), diarize GPU-bound, no contienden. Overhead ~0 vs ~3s serial medido en test acceptance | Aparece un enhance model que no contiende con diarize, o el overhead serial baja a <2s |
| Ejecutar skills mandatorios sin menús de opciones cuando las reglas son determinísticas | El usuario ya eligió la ceremonia al invocar el skill. Ofrecer menú contradice esa decisión | El skill no tiene regla determinística para el caso (verdadera ambigüedad, no indecisión del modelo) |
| Pipeline speechlib cambia a `run` → suggest+confirm (sin backdoor `--auto-confirm`, unmatched queda `SPEAKER_XX` raw) | Usuario explícito: "cambio duro, no opcional" | Scripts automatizados en producción que dependan del flujo auto hacen regresión imposible de absorber |
| Para transcripción es-CL usar `speechlib run --hotwords` (faster-whisper local), NO AssemblyAI | AAI rechaza `keyterms_prompt` en es y `word_boost` es marginal; faster-whisper + hotwords corrige errores léxicos empíricamente (6→0 en audio 35.7 min, 2026-04-23) | AAI libera `keyterms_prompt` para es; aparece otro proveedor cloud con logit bias efectivo en es; o faster-whisper deja de soportar el kwarg |
| `--hotwords` recibe CSV en CLI, propaga como `list[str]` interno, se joinea a `str` space-separated justo antes de `batched.transcribe` | faster-whisper espera `str`; mantener `list[str]` en la API interna es más semántico y localiza el leakage de la librería en el adapter | Cambia el contract de faster-whisper o la API pública de speechlib expone hotwords directamente |

## Restricciones verificadas

- Claude Code 2026 expone **21+ hook events**. Inventario completo: SessionStart/End, UserPromptSubmit, Stop/StopFailure, PreToolUse, PostToolUse/Failure, PermissionRequest/Denied, SubagentStart/Stop, TaskCreated/Completed, ConfigChange, CwdChanged, FileChanged, PreCompact/PostCompact, InstructionsLoaded, Elicitation/Result, Notification, WorktreeCreate/Remove. (verificado via claude-code-guide agent research 2026-04-11).
- `~/.claude/settings.json` **NO tiene flag** para "forzar obediencia estricta de skills". (verificado: búsqueda doc oficial 2026-04-11).
- Formato canónico de hooks en settings.json: `{"hooks": {"<event>": [{"matcher": "<regex>", "hooks": [{"type": "command", "command": "<shell>"}]}]}}`. (verificado: [code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks)).
- `speechlib/core_analysis.py:520` define `_start_compress_thread` que debe llamarse post-preprocess y join-earse antes de publish_to_source_folder cuando `compress && skip_enhance`. (verificado: test `test_compress_runs_parallel_with_diarization` pasa con overhead <2s después del fix `fe47a8d`).
- RTX 2070 Super Max-Q + faster-whisper `large-v3-turbo` procesa audio es a ~4× realtime. Medido 2026-04-23: 35.7 min audio → 134s wallclock incluyendo todo el pipeline (preprocess 15s + diarize 70s + transcribe 39s + output 10s).
- AssemblyAI SDK v0.54.1 expone `word_boost`, `boost_param`, `prompt`, `keyterms_prompt`, `keyterms_prompt_options` en `TranscriptionConfig`. El primero funciona en multiidioma; el 4º falla server-side fuera del inglés. Verificado vía `inspect.signature` + llamada real 2026-04-23.

## Anti-patrones confirmados

- **Ofrecer menú de N opciones cuando el skill tiene regla determinística** → viola el contrato de ejecución del skill. Alternativa: ejecutar la acción mandada directamente.
- **Buscar literatura del dominio (Kent Beck TDD) cuando la pregunta es sobre el sistema que ejecuta la tarea (Claude Code)** → confusión meta vs dominio, research irrelevante. Alternativa: identificar si la pregunta es sobre Claude Code CLI o sobre la metodología.
- **Escribir skills sin clausula explícita "DO NOT ask / DO NOT offer alternatives"** → el sesgo conservador del modelo prevalece sobre las reglas del skill. Alternativa: agregar language explícito + subir enforcement a hooks PreToolUse donde aplique.
- **`compress + skip_enhance` llamando a `compress_audio` serial al final del pipeline** → overhead 2.8s vs baseline 19.8s. Alternativa: `_start_compress_thread` después del preprocess, `thread.join()` antes de `_publish_to_source_folder`.
- **Dejar helpers huérfanos después de un refactor que cambia el flujo** (ej. `_start_compress_thread` post commit `acdf328`) → dead code invisible hasta que un test e2e falla. Alternativa: correr suite completa (no solo tests afectados por el diff) en el PR que cambia el flujo.
- **MagicMock duck-typed sobre librerías externas con type contracts estrictos** → el test pasa con cualquier tipo (ej. list aceptada donde faster-whisper espera str), el runtime real revienta. Evidencia: commit `74fca53` — `AttributeError: 'list' object has no attribute 'strip'` en `faster_whisper/transcribe.py:1545`. Alternativa: asertear tipo post-boundary (ej. `kwargs["hotwords"] == "term1 term2"` verifica str space-joined).
- **Usar AssemblyAI `word_boost` para corregir nombres/jerga en es-CL** → efecto marginal (cambia puntuación y segmentación, no léxico). Alternativa: faster-whisper + `--hotwords` (verificado empíricamente 2026-04-23).

## Dependencias

- [AXIOMA: skills son advisory, solo hooks son mandatorios] → [DECISIÓN: /bdd SKILL.md necesita sección "Execution discipline" con "DO NOT ask / DO NOT offer alternatives" explícito]
- [AXIOMA: solo PreToolUse bloquea] → [DECISIÓN: enforcement de "tests GREEN antes de editar src" requiere hook PreToolUse matcher=Edit|Write con pytest gate]
- [RESTRICCIÓN: hooks sin estado de sesión persistente] → [DECISIÓN: "one behavior at a time" se enforza con archivo `.claude/current-behavior.txt` + SessionStart reminder, no con hook puro]
- [AXIOMA: "executing actions with care" default existe en el system prompt] → [DECISIÓN: ceremonias de skill necesitan override explícito en SKILL.md o enforcement via hooks]
- [AXIOMA: hooks no pueden verificar semántica (RED legítimo vs espurio, dominio vs adaptador)] → [DECISIÓN: algunas reglas de /bdd permanecen como disciplina del modelo, no enforcement determinístico]
- [AXIOMA: faster-whisper `hotwords` espera `str`] → [DECISIÓN: `--hotwords` recibe CSV, propaga list[str] interno, joinea a str en `transcribe.py` antes de batched.transcribe]
- [AXIOMA: AAI es rechaza keyterms_prompt + word_boost marginal en es-CL] → [DECISIÓN: speechlib+hotwords default para es-CL, no AAI]
- [RESTRICCIÓN: RTX 2070 Super Max-Q procesa 35 min a ~134s] → [DECISIÓN: transcripción local viable; no hace falta cloud ASR para audios <1h en es-CL]
