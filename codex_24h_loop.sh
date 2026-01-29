#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/lichenyu/leviathan"
LOG="$ROOT/codex_24h.log"
PIDFILE="$ROOT/codex_24h.pid"
STOPFILE="$ROOT/codex_24h.stop"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"
DURATION_SECONDS="${DURATION_SECONDS:-86400}"

start_ts=$(date +%s)
end_ts=$((start_ts + DURATION_SECONDS))
echo "$$" > "$PIDFILE"

cat >> "$LOG" <<EOF
=== codex 24h loop started: $(date) ===
root: $ROOT
sleep: ${SLEEP_SECONDS}s
duration: ${DURATION_SECONDS}s
EOF

iter=0
while [ "$(date +%s)" -lt "$end_ts" ]; do
  iter=$((iter + 1))
  if [ -f "$STOPFILE" ]; then
    echo "Stopfile detected at $(date); exiting." >> "$LOG"
    break
  fi

  echo "=== Iteration $iter @ $(date) ===" >> "$LOG"

  (
    cd "$ROOT"
    codex exec --sandbox workspace-write --full-auto \
      "You are improving the repository at $ROOT.
Make ONE small, high-impact improvement per iteration (bug fix, test, docs, or refactor).
Rules:
- Do NOT modify or delete .DS_Store or notebooks/test_cluster.ipynb.
- Do NOT delete files or run destructive commands.
- No network access or package installs.
- Keep changes within the repo.
- If tests exist and are cheap, run them; otherwise explain what you'd run.
Output a brief summary of changes and tests."
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

echo "=== codex 24h loop ended: $(date) ===" >> "$LOG"
