#!/usr/bin/env bash
# Launch Claude Code CLI agents — codex-style loop, one Claude invocation per round.
# Each round: fetch state → Claude reasons → Claude acts via curl → sleep → repeat
#
# Usage:
#   ./scripts/agents/launch_agents.sh              # launch all 3 agents
#   ./scripts/agents/launch_agents.sh stop          # stop all agents
#   ./scripts/agents/launch_agents.sh status        # check agent status
#   ./scripts/agents/launch_agents.sh logs          # tail all agent logs
#
# Architecture: agent_loop.sh polls the server, invokes `claude -p` each round
# with fresh world state. Claude CLI is the "brain" — it analyzes state, generates
# strategy, submits actions, votes, and proposes mechanisms via curl.
#
# Inspired by archive/codex/codex_24h_loop.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
AGENT_DIR="$ROOT/scripts/agents"
LOG_DIR="$ROOT/logs/agents"
PID_DIR="$ROOT/logs/agents/pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

AGENTS=(
  "expansionist"
  "diplomat"
  "strategist"
)

stop_agents() {
  echo "Stopping all agents..."
  for name in "${AGENTS[@]}"; do
    pidfile="$PID_DIR/${name}.pid"
    if [ -f "$pidfile" ]; then
      pid=$(cat "$pidfile")
      if kill -0 "$pid" 2>/dev/null; then
        # Kill the loop and any child claude processes
        pkill -P "$pid" 2>/dev/null || true
        kill "$pid" 2>/dev/null || true
        echo "  stopped $name (pid $pid)"
      else
        echo "  $name already stopped"
      fi
      rm -f "$pidfile"
    else
      echo "  $name not running"
    fi
  done
}

status_agents() {
  for name in "${AGENTS[@]}"; do
    pidfile="$PID_DIR/${name}.pid"
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      pid=$(cat "$pidfile")
      # Count child processes (claude instances)
      children=$(pgrep -P "$pid" 2>/dev/null | wc -l | tr -d ' ')
      log="$LOG_DIR/${name}.log"
      last_line=$(tail -1 "$log" 2>/dev/null || echo "no log")
      echo "  $name: RUNNING (pid $pid, $children children)"
      echo "    last: $last_line"
    else
      echo "  $name: STOPPED"
    fi
  done
}

launch_agent() {
  local name="$1"
  local loop_script="$AGENT_DIR/agent_loop.sh"
  local pid_file="$PID_DIR/${name}.pid"

  if [ ! -f "$loop_script" ]; then
    echo "ERROR: loop script not found: $loop_script"
    return 1
  fi

  # Kill existing if running
  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "  $name already running (pid $(cat "$pid_file")), restarting..."
    pkill -P "$(cat "$pid_file")" 2>/dev/null || true
    kill "$(cat "$pid_file")" 2>/dev/null || true
    sleep 1
  fi

  echo "  launching $name (codex-style loop)"

  # Launch the loop script in background
  nohup bash "$loop_script" "$name" > /dev/null 2>&1 &
  local loop_pid=$!
  echo "$loop_pid" > "$pid_file"
  echo "  $name started (pid $loop_pid) → logs: $LOG_DIR/${name}.log"
}

case "${1:-launch}" in
  stop)
    stop_agents
    ;;
  status)
    status_agents
    ;;
  logs)
    echo "Tailing agent logs (Ctrl+C to stop)..."
    tail -f "$LOG_DIR"/expansionist.log "$LOG_DIR"/diplomat.log "$LOG_DIR"/strategist.log
    ;;
  launch|start)
    echo "Launching Leviathan agents (codex-style — Claude CLI invoked each round)"
    echo "Logs: $LOG_DIR/"
    echo ""
    for name in "${AGENTS[@]}"; do
      launch_agent "$name"
    done
    echo ""
    echo "All agents launched. Monitor with:"
    echo "  $0 status"
    echo "  $0 logs"
    echo "  $0 stop"
    ;;
  *)
    echo "Usage: $0 [launch|stop|status|logs]"
    exit 1
    ;;
esac
