#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/lichenyu/leviathan"
LOG="$ROOT/codex_24h.log"
PIDFILE="$ROOT/codex_24h.pid"
STOPFILE="$ROOT/codex_24h.stop"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"
DURATION_SECONDS="${DURATION_SECONDS:-86400}"
OBJECTIVE_FILE="${OBJECTIVE_FILE:-$ROOT/codex_objective.txt}"
OBJECTIVE_DEFAULT="Meaningful behavior = self-improving strategy design without collapsing agent/environment diversity; only do performance optimizations/refactors when the gain is large and measurable."
OBJECTIVE="${OBJECTIVE:-$OBJECTIVE_DEFAULT}"
REQUIRE_OPENAI_API_KEY="${REQUIRE_OPENAI_API_KEY:-1}"

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
    objective="$OBJECTIVE"
    if [ -f "$OBJECTIVE_FILE" ]; then
      file_objective="$(cat "$OBJECTIVE_FILE" | tr -d '\r')"
      if [ -n "$file_objective" ]; then
        objective="$file_objective"
      fi
    fi
    if [ "$REQUIRE_OPENAI_API_KEY" = "1" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
      echo "OPENAI_API_KEY is not set. Set it (or export REQUIRE_OPENAI_API_KEY=0) to continue." >> "$LOG"
      exit 1
    fi
    echo "objective: $objective" >> "$LOG"
    cd "$ROOT"
    codex exec --sandbox workspace-write --full-auto \
      "You are improving the repository at $ROOT.
Direction:
- Focus on meaningful behavior. $objective
- Preserve agent and environment diversity; avoid changes that narrow strategy space.
- Prefer changes that improve strategy generation, learning, planning, coordination, memory, or evaluation.
- Performance/refactor changes only if you can justify large, measurable gains (and say how you'd measure).
Evidence-first rule:
- Prioritize improvements based on real test/eval results, not blind edits.
- If you cannot run the relevant test/eval for a proposed change, do NOT change code; instead improve tests, eval plan, or instrumentation so the result can be measured next.
Size constraint (script length):
- Keep any individual script file you create or modify at ~1500 lines or fewer; do NOT exceed 2000 lines.
- If a target script is already large or would exceed the limit, you must refactor into smaller files/modules.
Testing (stage-gated):
- Stage 1 (always): run 'python test_graph_system.py' if it exists and is fast.
- Stage 2 (conditional): if Stage 1 passes AND changes touch core behavior/learning/planning logic, run 'pytest -q'.
- Stage 3 (conditional): if you changed core behavior or reward/memory dynamics, run a minimal simulation script if one exists; otherwise describe what you'd run.
  When Stage 3 is skipped, say 'Stage 3 skipped (not run): <reason>' instead of 'API unavailable'.
Evaluation design (explicit + iterative):
- Maintain or update a concise evaluation plan (e.g., `EVAL_PLAN.md`) with: metrics, baselines, thresholds, and expected deltas.
- Stage eval scale: small (few agents, 1–3 runs) → medium (more agents, 3–5 runs) → large only if gains look real.
- When you change behavior logic, update the eval plan and run the smallest relevant eval; scale up only if results improve.
Rules (lightweight safety):
- No destructive commands and no deleting files.
- Network access is allowed only for Stage 3 minimal simulation / benchmark checks; otherwise no network access.
- Keep changes within the repo.
- Avoid large binary edits (.DS_Store, images, big notebooks) unless necessary.
Continuity:
- Read 'codex_resume.md' at the start of each iteration for context.
- Update 'codex_resume.md' before finishing with: current focus, recent changes, tests/evals, results, next step, and risks.
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

echo "=== codex 24h loop ended: $(date) ===" >> "$LOG"
