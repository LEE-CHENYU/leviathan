#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/lichenyu/leviathan"
LOG="$ROOT/arxiv_24h.log"
PIDFILE="$ROOT/arxiv_24h.pid"
STOPFILE="$ROOT/arxiv_24h.stop"
ENV_FILE="$ROOT/.env"
DURATION_HOURS="${DURATION_HOURS:-24}"
INTERVAL_MINUTES="${INTERVAL_MINUTES:-360}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/config/arxiv_loop.yaml}"
TOPIC="${ARXIV_TOPIC:-}"
QUERY="${ARXIV_QUERY:-}"
PYTHON_BIN="${PYTHON_BIN:-/Users/lichenyu/miniconda3/bin/python}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export ARXIV_STOPFILE="$STOPFILE"

echo "$$" > "$PIDFILE"

echo "=== arXiv loop started: $(date) ===" >> "$LOG"

(
  args=("$ROOT/scripts/arxiv_research.py" --config "$CONFIG_FILE" \
    --duration-hours "$DURATION_HOURS" --interval-minutes "$INTERVAL_MINUTES")
  if [ -n "$TOPIC" ]; then
    args+=(--topic "$TOPIC")
  fi
  if [ -n "$QUERY" ]; then
    args+=(--query "$QUERY")
  fi
  "$PYTHON_BIN" "${args[@]}"
) >> "$LOG" 2>&1

echo "=== arXiv loop ended: $(date) ===" >> "$LOG"
