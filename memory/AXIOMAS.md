# AXIOMAS DEL PROYECTO
Última destilación: 2026-04-11 (sesión: skill execution discipline + hooks)
Branch: refactor/speaker-domain

## Axiomas del dominio

- Claude Code skills son **model-controlled (advisory)**. Solo hooks son deterministas. Fuente: [code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks) — "Hooks are mandatory. CLAUDE.md is advisory."
- Solo `PreToolUse` hook event puede **bloquear** acciones (`exit 2`). Los demás son informativos o post-ejecución y no pueden revocar.
- Hooks reciben JSON por stdin (`tool_name`, `tool_input`, `cwd`, `session_id`) y responden con exit code (0=continuar, 2=bloquear) o JSON estructurado con `permissionDecision: "deny"` y razón.
- Hooks NO tienen estado de sesión persistente nativo. Solo `CLAUDE_ENV_FILE` (vars inyectadas por `SessionStart`) + archivos en disco bajo `.claude/`.
- El system prompt default de Claude Code incluye "executing actions with care" que genera sesgo conservador a pedir confirmación, y ese sesgo se filtra a la ejecución de skills incluso cuando el SKILL.md tiene reglas "no-negociables" escritas.

## Decisiones activas

| Decisión | Por qué irreducible | Invalida si |
|---|---|---|
| /bdd Fase 0 obligatorio aunque haya que arreglar regresiones fuera del scope del behavior actual | La disciplina capturó dead code oculto (`_start_compress_thread` orphan desde commit `acdf328`) que no se veía sin forzar la suite completa | El fix requiere días de side quest o el usuario explícitamente releva la ceremonia |
| `compress + skip_enhance` corre thread-parallel con diarize en core_analysis.py | compress es CPU-bound (ffmpeg), diarize GPU-bound, no contienden. Overhead ~0 vs ~3s serial medido en test acceptance | Aparece un enhance model que no contiende con diarize, o el overhead serial baja a <2s |
| Ejecutar skills mandatorios sin menús de opciones cuando las reglas son determinísticas | El usuario ya eligió la ceremonia al invocar el skill. Ofrecer menú contradice esa decisión | El skill no tiene regla determinística para el caso (verdadera ambigüedad, no indecisión del modelo) |
| Pipeline speechlib cambia a `run` → suggest+confirm (sin backdoor `--auto-confirm`, unmatched queda `SPEAKER_XX` raw) | Usuario explícito: "cambio duro, no opcional" | Scripts automatizados en producción que dependan del flujo auto hacen regresión imposible de absorber |

## Restricciones verificadas

- Claude Code 2026 expone **21+ hook events**. Inventario completo: SessionStart/End, UserPromptSubmit, Stop/StopFailure, PreToolUse, PostToolUse/Failure, PermissionRequest/Denied, SubagentStart/Stop, TaskCreated/Completed, ConfigChange, CwdChanged, FileChanged, PreCompact/PostCompact, InstructionsLoaded, Elicitation/Result, Notification, WorktreeCreate/Remove. (verificado via claude-code-guide agent research 2026-04-11).
- `~/.claude/settings.json` **NO tiene flag** para "forzar obediencia estricta de skills". (verificado: búsqueda doc oficial 2026-04-11).
- Formato canónico de hooks en settings.json: `{"hooks": {"<event>": [{"matcher": "<regex>", "hooks": [{"type": "command", "command": "<shell>"}]}]}}`. (verificado: [code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks)).
- `speechlib/core_analysis.py:520` define `_start_compress_thread` que debe llamarse post-preprocess y join-earse antes de publish_to_source_folder cuando `compress && skip_enhance`. (verificado: test `test_compress_runs_parallel_with_diarization` pasa con overhead <2s después del fix `fe47a8d`).

## Anti-patrones confirmados

- **Ofrecer menú de N opciones cuando el skill tiene regla determinística** → viola el contrato de ejecución del skill. Alternativa: ejecutar la acción mandada directamente.
- **Buscar literatura del dominio (Kent Beck TDD) cuando la pregunta es sobre el sistema que ejecuta la tarea (Claude Code)** → confusión meta vs dominio, research irrelevante. Alternativa: identificar si la pregunta es sobre Claude Code CLI o sobre la metodología.
- **Escribir skills sin clausula explícita "DO NOT ask / DO NOT offer alternatives"** → el sesgo conservador del modelo prevalece sobre las reglas del skill. Alternativa: agregar language explícito + subir enforcement a hooks PreToolUse donde aplique.
- **`compress + skip_enhance` llamando a `compress_audio` serial al final del pipeline** → overhead 2.8s vs baseline 19.8s. Alternativa: `_start_compress_thread` después del preprocess, `thread.join()` antes de `_publish_to_source_folder`.
- **Dejar helpers huérfanos después de un refactor que cambia el flujo** (ej. `_start_compress_thread` post commit `acdf328`) → dead code invisible hasta que un test e2e falla. Alternativa: correr suite completa (no solo tests afectados por el diff) en el PR que cambia el flujo.

## Dependencias

- [AXIOMA: skills son advisory, solo hooks son mandatorios] → [DECISIÓN: /bdd SKILL.md necesita sección "Execution discipline" con "DO NOT ask / DO NOT offer alternatives" explícito]
- [AXIOMA: solo PreToolUse bloquea] → [DECISIÓN: enforcement de "tests GREEN antes de editar src" requiere hook PreToolUse matcher=Edit|Write con pytest gate]
- [RESTRICCIÓN: hooks sin estado de sesión persistente] → [DECISIÓN: "one behavior at a time" se enforza con archivo `.claude/current-behavior.txt` + SessionStart reminder, no con hook puro]
- [AXIOMA: "executing actions with care" default existe en el system prompt] → [DECISIÓN: ceremonias de skill necesitan override explícito en SKILL.md o enforcement via hooks]
- [AXIOMA: hooks no pueden verificar semántica (RED legítimo vs espurio, dominio vs adaptador)] → [DECISIÓN: algunas reglas de /bdd permanecen como disciplina del modelo, no enforcement determinístico]
