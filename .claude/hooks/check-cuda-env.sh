#!/bin/bash
# PreToolUse hook: warns if mamba env 'speechlib' is not active
# when running python/uv speechlib commands. Does NOT block.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)

# Only check speechlib-related commands
if ! echo "$COMMAND" | grep -qE '(python -m speechlib|python.+speechlib)'; then
  exit 0
fi

# If speechlib mamba env is already active, all good
if [ "$CONDA_DEFAULT_ENV" = "speechlib" ]; then
  exit 0
fi

echo "WARNING: El ambiente mamba 'speechlib' no esta activo (CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV)." >&2
echo "CUDA/RTX no estara disponible. Activa con: mamba activate speechlib" >&2
exit 0
