#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/codex_24h.log"
PIDFILE="$ROOT/codex_24h.pid"
STOPFILE="$ROOT/codex_24h.stop"
RESUME="$ROOT/codex_resume.md"
STATE_DIR="$HOME/.clawdbot"
STATE_FILE="$STATE_DIR/codex_watchdog.state"
ENV_FILE="$ROOT/.env"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

WHATSAPP_TARGET="${WATCH_WHATSAPP_TARGET:-}"
TELEGRAM_TARGET="${WATCH_TELEGRAM_TARGET:-}"

send_message() {
  local msg="$1"
  if [ "${WATCH_DRY_RUN:-0}" = "1" ]; then
    printf '%s\n' "$msg"
    return
  fi
  if [ -n "$WHATSAPP_TARGET" ]; then
    "$ROOT/scripts/clawdbot_env.sh" message send --channel whatsapp --target "$WHATSAPP_TARGET" --message "$msg" >/dev/null 2>&1 || true
  fi
  if [ -n "$TELEGRAM_TARGET" ]; then
    "$ROOT/scripts/clawdbot_env.sh" message send --channel telegram --target "$TELEGRAM_TARGET" --message "$msg" >/dev/null 2>&1 || true
  fi
}

mkdir -p "$STATE_DIR"

last_iter=0
last_status=""
if [ -f "$STATE_FILE" ]; then
  last_iter=$(awk -F= '/^iter=/{print $2; exit}' "$STATE_FILE" || true)
  last_status=$(awk -F= '/^status=/{print $2; exit}' "$STATE_FILE" || true)
fi
last_iter=${last_iter:-0}

current_iter=0
if [ -f "$LOG" ]; then
  current_iter=$(tail -n 2000 "$LOG" | awk '$1=="===" && $2=="Iteration" {iter=$3} END{print iter+0}')
fi

status="stopped"
pid=""
if [ -f "$PIDFILE" ]; then
  pid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
    status="running"
  else
    status="stopped"
  fi
fi
if [ -f "$STOPFILE" ]; then
  status="stopfile"
fi

should_send=0
if [ "$current_iter" -gt "$last_iter" ]; then
  should_send=1
fi
if [ "$status" != "$last_status" ]; then
  should_send=1
fi

if [ "$should_send" -eq 1 ]; then
  msg="codex loop update ($(date))\nstatus: $status\niter: $current_iter"
  if [ -n "$pid" ]; then
    msg+="\npid: $pid"
  fi
  if [ -f "$RESUME" ]; then
    resume=$(sed -n '1,80p' "$RESUME")
    if [ -n "$resume" ]; then
      msg+="\n\ncodex_resume.md:\n$resume"
    fi
  fi

  if [ -n "$WHATSAPP_TARGET" ] || [ -n "$TELEGRAM_TARGET" ]; then
    send_message "$msg"
  fi
fi

printf 'iter=%s\nstatus=%s\n' "$current_iter" "$status" > "$STATE_FILE"
