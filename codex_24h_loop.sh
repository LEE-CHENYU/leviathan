#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/lichenyu/leviathan"
LOG="$ROOT/codex_24h.log"
PIDFILE="$ROOT/codex_24h.pid"
STOPFILE="$ROOT/codex_24h.stop"
ENV_FILE="$ROOT/.env"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"
DURATION_SECONDS="${DURATION_SECONDS:-86400}"
OBJECTIVE_FILE="${OBJECTIVE_FILE:-$ROOT/codex_objective.txt}"
OBJECTIVE_DEFAULT="Meaningful behavior = self-improving strategy design without collapsing agent/environment diversity; only do performance optimizations/refactors when the gain is large and measurable."
OBJECTIVE="${OBJECTIVE:-$OBJECTIVE_DEFAULT}"
REQUIRE_OPENAI_API_KEY="${REQUIRE_OPENAI_API_KEY:-1}"
RUN_E2E_OUTSIDE_CODEX="${RUN_E2E_OUTSIDE_CODEX:-1}"
E2E_SCRIPT="${E2E_SCRIPT:-$ROOT/scripts/run_e2e_stage3.sh}"
SANDBOX_MODE="${SANDBOX_MODE:-danger-full-access}"

STAGE3_PROMPT="Stage 3 (end-to-end, source of truth): run every iteration unless it is impossible to run (missing keys/network/downstream outage).
  Command: 'python scripts/run_e2e_smoke.py'
  Use the e2e summary at execution_histories/e2e_smoke/latest_summary.json as the primary feedback for evaluation and iteration decisions.
  If Stage 3 can't run due to critical infra/LLM failures (DNS/auth/rate-limit), do NOT change behavior; instead improve the test/instrumentation so Stage 3 can run.
  Minor file/log issues (e.g., missing/empty log files) are non-blocking; you may proceed with behavior changes but must note the issue and keep a Stage 3 rerun on the next step list.
  When Stage 3 is skipped, say 'Stage 3 skipped (not run): <reason>' instead of 'API unavailable'."
if [ "$RUN_E2E_OUTSIDE_CODEX" = "1" ]; then
  STAGE3_PROMPT="Stage 3 (end-to-end, source of truth): already executed outside Codex in this iteration via $E2E_SCRIPT.
  Do NOT run 'python scripts/run_e2e_smoke.py' inside Codex.
  Read execution_histories/e2e_smoke/latest_summary.json and use it as the primary feedback for evaluation and iteration decisions.
  If the summary is missing or clearly stale, report that and request a rerun outside Codex.
  Minor file/log issues (e.g., missing/empty log files) are non-blocking; you may proceed with behavior changes but must note the issue and keep a Stage 3 rerun on the next step list."
fi

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
    if [ "$RUN_E2E_OUTSIDE_CODEX" = "1" ] && [ -x "$E2E_SCRIPT" ]; then
      echo "--- stage3 external start: $(date) ---" >> "$LOG"
      if "$E2E_SCRIPT" >> "$LOG" 2>&1; then
        echo "--- stage3 external ok ---" >> "$LOG"
      else
        code=$?
        echo "--- stage3 external failed (exit $code) ---" >> "$LOG"
      fi
    fi
    cd "$ROOT"
    codex exec --sandbox "$SANDBOX_MODE" --full-auto \
      "You are improving the repository at $ROOT.
Direction:
- Focus on meaningful behavior. $objective
- Preserve agent and environment diversity; avoid changes that narrow strategy space.
- Prefer changes that improve strategy generation, learning, planning, coordination, memory, or evaluation.
- Performance/refactor changes only if you can justify large, measurable gains (and say how you'd measure).
Evidence-first rule:
- Prioritize improvements based on real test/eval results, not blind edits.
- If you cannot run the relevant test/eval due to critical infra/LLM failures (DNS/auth/rate-limit), do NOT change behavior; instead improve tests, eval plan, or instrumentation so the result can be measured next. Minor LLM errors should be treated as provisional (note them and rerun), not as blockers.
Size constraint (script length):
- Keep any individual script file you create or modify at ~1500 lines or fewer; do NOT exceed 2000 lines.
- If a target script is already large or would exceed the limit, you must refactor into smaller files/modules.
Testing (stage-gated, but bias toward running):
- Run Stage 1/2/3 early in the iteration (before proposing code changes) to establish a baseline.
- Stage 1 (always): run 'python test_graph_system.py' if it exists and is fast.
- Stage 2 (prefer to run): after Stage 1, run 'pytest -q' unless it is clearly broken/missing deps; report failures explicitly.
- ${STAGE3_PROMPT}
Evaluation design (explicit + iterative):
- Maintain or update a concise evaluation plan (e.g., EVAL_PLAN.md) with: metrics, baselines, thresholds, and expected deltas.
- Stage eval scale: small (few agents, 1–3 runs) → medium (more agents, 3–5 runs) → large only if gains look real.
- When you change behavior logic, update the eval plan and run the smallest relevant eval; scale up only if results improve.
Rules (lightweight safety):
- No destructive commands and no deleting files.
- Network access is allowed only for Stage 3 minimal simulation / benchmark checks; otherwise no network access.
- Keep changes within the repo.
- Avoid large binary edits (.DS_Store, images, big notebooks) unless necessary.
Continuity:
- Read 'codex_resume.md' at the start of each iteration for context.
- Read any recent research summaries in 'research/arxiv/latest.md' if present for idea feed.
- Update 'codex_resume.md' before finishing with: current focus, recent changes, tests/evals, results, next step, and risks.
Commit hygiene:
- If you make code changes and tests pass, make a Git commit before finishing.
- Use this message format: codex(iter ${iter}): <short, specific change summary>.
- If no code changes were made, state 'No commit (no code changes)'.
Output a brief summary of changes and tests (run cheap tests if available, otherwise say what you'd run)."
  ) >> "$LOG" 2>&1

  (
    echo "=== Commit instructions ==="
    echo "After changes are made and tests pass, commit with a clear message in this format:"
    echo "  git add -A"
    echo "  git commit -m \"codex(iter ${iter}): <short, specific change summary>\""
    echo "If no code changes were made, skip committing and say so explicitly."
    echo "=== End commit instructions ==="
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
