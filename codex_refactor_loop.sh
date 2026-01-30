#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/lichenyu/leviathan"
LOG="$ROOT/codex_refactor.log"
PIDFILE="$ROOT/codex_refactor.pid"
STOPFILE="$ROOT/codex_refactor.stop"
ENV_FILE="$ROOT/.env"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"
DURATION_SECONDS="${DURATION_SECONDS:-86400}"
REFRACTOR_TARGET_LINES="${REFRACTOR_TARGET_LINES:-2000}"
REFRACTOR_IDEAL_LINES="${REFRACTOR_IDEAL_LINES:-1500}"
RESTART_MAIN_LOOP="${RESTART_MAIN_LOOP:-1}"
MAIN_LOOP_SCRIPT="${MAIN_LOOP_SCRIPT:-$ROOT/codex_24h_loop.sh}"
SANDBOX_MODE="${SANDBOX_MODE:-workspace-write}"
RESUME_FILE="${RESUME_FILE:-$ROOT/codex_refactor_resume.md}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

start_ts=$(date +%s)
end_ts=$((start_ts + DURATION_SECONDS))
echo "$$" > "$PIDFILE"

cat >> "$LOG" <<EOF
=== codex refactor loop started: $(date) ===
root: $ROOT
sleep: ${SLEEP_SECONDS}s
duration: ${DURATION_SECONDS}s
target_lines: ${REFRACTOR_TARGET_LINES}
ideal_lines: ${REFRACTOR_IDEAL_LINES}
EOF

iter=0
while [ "$(date +%s)" -lt "$end_ts" ]; do
  iter=$((iter + 1))
  if [ -f "$STOPFILE" ]; then
    echo "Stopfile detected at $(date); exiting." >> "$LOG"
    break
  fi

  echo "=== Refactor Iteration $iter @ $(date) ===" >> "$LOG"

  (
    cd "$ROOT"
    codex exec --sandbox "$SANDBOX_MODE" --full-auto \
      "You are refactoring the repository at $ROOT.
Goal:
- Reduce oversized scripts/modules to <= ${REFRACTOR_TARGET_LINES} lines (ideal <= ${REFRACTOR_IDEAL_LINES}) without changing behavior.
- Start with the lowest-risk files first (utilities, scripts, logging/metrics helpers) before core simulation logic.
- Stop when all files exceed no more than the target threshold or when the loop ends.
 - Light re-organization is allowed (create folders, move scripts/modules) if it improves clarity or reduces file size, but keep changes incremental and behavior-identical.

Process:
- First, scan for files exceeding the target (use wc -l or rg --files + wc -l).
- Pick one low-risk file to refactor per iteration.
- Refactor by extracting helpers into new modules, preserving APIs and behavior.
- Update imports and references accordingly.

Testing (incremental):
- After each refactor step, run 'python test_graph_system.py' if it exists and is fast.
- If the change touches core simulation logic, also run 'pytest -q'.
- If tests fail, fix them before finishing the iteration.

Continuity:
- Update ${RESUME_FILE} with: current focus, files refactored, tests run, results, next step, and risks.

Commit hygiene:
- If you make code changes and tests pass, make a Git commit before finishing.
- Use this message format: codex(refactor ${iter}): <short, specific change summary>.
- If no code changes were made, state 'No commit (no code changes)'.

Output a brief summary of changes and tests (run cheap tests if available, otherwise say what you'd run)."
  ) >> "$LOG" 2>&1

  {
    echo "--- git status ---"
    cd "$ROOT" && git status -sb
    echo "--- git diff --stat ---"
    cd "$ROOT" && git diff --stat
    echo "--- end iteration ---"
  } >> "$LOG" 2>&1

  sleep "$SLEEP_SECONDS"
done

echo "=== codex refactor loop ended: $(date) ===" >> "$LOG"

if [ "$RESTART_MAIN_LOOP" = "1" ] && [ -x "$MAIN_LOOP_SCRIPT" ]; then
  nohup bash "$MAIN_LOOP_SCRIPT" > "$ROOT/codex_24h.nohup.out" 2>&1 &
  echo "=== main loop restarted: $(date) ===" >> "$LOG"
fi
