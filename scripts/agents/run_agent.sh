#!/usr/bin/env bash
# Run a single agent. Usage: ./run_agent.sh <agent_name>
# Must be run from repo root.
set -uo pipefail

NAME="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROMPT_FILE="$ROOT/scripts/agents/prompts/${NAME}.md"
LOG_FILE="$ROOT/logs/agents/${NAME}.log"

if [ ! -f "$PROMPT_FILE" ]; then
  echo "ERROR: prompt not found: $PROMPT_FILE" >&2
  exit 1
fi

echo "[$NAME] Starting at $(date)" > "$LOG_FILE"
echo "[$NAME] Prompt: $PROMPT_FILE" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"

# Unset CLAUDECODE to allow nested launch, pipe prompt via stdin
unset CLAUDECODE 2>/dev/null || true

cat "$PROMPT_FILE" | claude \
  -p \
  --dangerously-skip-permissions \
  --model sonnet \
  --allowedTools "Bash" \
  >> "$LOG_FILE" 2>&1

echo "[$NAME] Exited at $(date)" >> "$LOG_FILE"
