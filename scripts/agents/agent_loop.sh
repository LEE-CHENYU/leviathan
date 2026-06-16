#!/usr/bin/env bash
# ── Codex-style agent loop ──────────────────────────────────────────────────
# Each round: fetch world state → invoke Claude CLI → Claude reasons + acts → sleep
#
# Usage:
#   ./scripts/agents/agent_loop.sh <agent_name>
#   e.g.: ./scripts/agents/agent_loop.sh diplomat
#
# Requires: claude CLI, python3, curl, jq (optional)
# Inspired by archive/codex/codex_24h_loop.sh
set -uo pipefail

NAME="${1:?Usage: agent_loop.sh <agent_name>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="${LEVIATHAN_BASE:-https://leviathan.fly.dev}"

PROMPT_FILE="$ROOT/scripts/agents/prompts/${NAME}_round.md"
MEMORY_FILE="$ROOT/logs/agents/${NAME}_memory.md"
CRED_FILE="$ROOT/logs/agents/credentials.json"
LOG_FILE="$ROOT/logs/agents/${NAME}.log"
PID_FILE="$ROOT/logs/agents/pids/${NAME}.pid"
SUMMARIZER="$ROOT/scripts/agents/summarize_state.py"

# ── Credential mapping ──
case "$NAME" in
  expansionist) CRED_KEY="Expansionist-Claude" ;;
  diplomat)     CRED_KEY="Diplomat-Claude" ;;
  strategist)   CRED_KEY="Strategist-Claude" ;;
  *)            CRED_KEY="$NAME" ;;
esac

# ── Validate prerequisites ──
if [ ! -f "$PROMPT_FILE" ]; then
  echo "ERROR: prompt not found: $PROMPT_FILE" >&2
  exit 1
fi

# ── Load or register credentials ──
save_credentials() {
  python3 -c "
import json, os
f='$CRED_FILE'
d={}
if os.path.exists(f):
    with open(f) as fh: d=json.load(fh)
d['$CRED_KEY']={'api_key':'$API_KEY','member_id':$MEMBER_ID}
with open(f,'w') as fh: json.dump(d,fh,indent=2)
" 2>/dev/null
}

load_credentials() {
  if [ -f "$CRED_FILE" ]; then
    API_KEY=$(python3 -c "import json; print(json.load(open('$CRED_FILE')).get('$CRED_KEY',{}).get('api_key',''))" 2>/dev/null)
    MEMBER_ID=$(python3 -c "import json; print(json.load(open('$CRED_FILE')).get('$CRED_KEY',{}).get('member_id',''))" 2>/dev/null)
  fi

  # Validate existing key against server
  if [ -n "${API_KEY:-}" ] && [ -n "${MEMBER_ID:-}" ]; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/v1/agents/me" -H "X-API-Key: $API_KEY" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" != "200" ]; then
      echo "[$NAME] Saved key invalid (HTTP $HTTP_CODE) — re-registering..."
      API_KEY=""
      MEMBER_ID=""
    fi
  fi

  if [ -z "${API_KEY:-}" ] || [ -z "${MEMBER_ID:-}" ]; then
    echo "[$NAME] No valid credentials — registering..."
    REG=$(curl -s -X POST "$BASE/v1/agents/register" \
      -H "Content-Type: application/json" \
      -d "{\"name\": \"$CRED_KEY\", \"description\": \"Claude CLI agent ($NAME)\"}")
    API_KEY=$(echo "$REG" | python3 -c "import sys,json; print(json.load(sys.stdin)['api_key'])" 2>/dev/null)
    MEMBER_ID=$(echo "$REG" | python3 -c "import sys,json; print(json.load(sys.stdin)['member_id'])" 2>/dev/null)
    if [ -z "$API_KEY" ]; then
      echo "[$NAME] Registration failed: $REG"
      return 1
    fi
    echo "[$NAME] Registered: member_id=$MEMBER_ID"
    save_credentials
  fi
}

re_register() {
  echo "[$NAME] Re-registering (member dead)..."
  REG=$(curl -s -X POST "$BASE/v1/agents/register" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$CRED_KEY\", \"description\": \"Claude CLI agent ($NAME)\"}")
  NEW_KEY=$(echo "$REG" | python3 -c "import sys,json; print(json.load(sys.stdin).get('api_key',''))" 2>/dev/null)
  NEW_MID=$(echo "$REG" | python3 -c "import sys,json; print(json.load(sys.stdin).get('member_id',''))" 2>/dev/null)
  if [ -n "$NEW_KEY" ] && [ -n "$NEW_MID" ]; then
    API_KEY="$NEW_KEY"
    MEMBER_ID="$NEW_MID"
    echo "[$NAME] Re-registered: member_id=$MEMBER_ID"
    save_credentials
    return 0
  fi
  echo "[$NAME] Re-registration failed: $REG"
  return 1
}

# ── Init ──
mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
echo "$$" > "$PID_FILE"
load_credentials

{
  echo "============================================================"
  echo "[$NAME] Agent loop started: $(date)"
  echo "[$NAME] member_id=$MEMBER_ID, base=$BASE"
  echo "============================================================"
} > "$LOG_FILE"

last_round=-1
consecutive_dead=0

# ── Main loop ──
while true; do
  # 1. Poll deadline
  DL=$(curl -s "$BASE/v1/world/rounds/current/deadline" 2>/dev/null)
  STATE=$(echo "$DL" | python3 -c "import sys,json; print(json.load(sys.stdin).get('state',''))" 2>/dev/null || echo "")
  ROUND=$(echo "$DL" | python3 -c "import sys,json; print(json.load(sys.stdin).get('round_id',-1))" 2>/dev/null || echo "-1")

  if [ "$STATE" != "accepting" ] || [ "$ROUND" = "$last_round" ]; then
    sleep 5
    continue
  fi

  echo "" >> "$LOG_FILE"
  echo "──────────────────────────────────────────────────" >> "$LOG_FILE"
  echo "[$NAME] ROUND $ROUND — $(date '+%H:%M:%S')" >> "$LOG_FILE"

  # 2. Check if alive by looking at snapshot
  ALIVE=$(curl -s "$BASE/v1/world/snapshot" 2>/dev/null | \
    python3 -c "
import sys,json
snap=json.load(sys.stdin)
mid=$MEMBER_ID
alive=any(m.get('id')==mid for m in snap.get('members',[]))
print('yes' if alive else 'no')
" 2>/dev/null || echo "unknown")

  if [ "$ALIVE" = "no" ]; then
    consecutive_dead=$((consecutive_dead + 1))
    echo "[$NAME] Member $MEMBER_ID is DEAD (attempt $consecutive_dead)" >> "$LOG_FILE"
    if re_register; then
      consecutive_dead=0
      # Clear memory on death
      echo "# Fresh start — previous member died" > "$MEMORY_FILE"
    else
      echo "[$NAME] Re-registration failed, waiting 30s..." >> "$LOG_FILE"
      sleep 30
    fi
    last_round=$ROUND
    continue
  fi
  consecutive_dead=0

  # 3. Generate compact world summary
  WORLD_SUMMARY=$(python3 "$SUMMARIZER" "$MEMBER_ID" 2>/dev/null || echo "Error fetching world state")

  # 4. Read agent memory
  AGENT_MEMORY=""
  if [ -f "$MEMORY_FILE" ]; then
    AGENT_MEMORY=$(cat "$MEMORY_FILE")
  fi

  # 5. Read personality prompt
  PERSONALITY=$(cat "$PROMPT_FILE")

  # 6. Build full round prompt
  ROUND_PROMPT="$PERSONALITY

---

## Environment Variables (already set — use them in your curl commands)
- \$BASE = $BASE
- \$API_KEY = $API_KEY
- \$MEMBER_ID = $MEMBER_ID
- \$ROUND = $ROUND
- \$MEMORY_FILE = $MEMORY_FILE

## Current World State (Round $ROUND)
$WORLD_SUMMARY

## Your Memory (from previous rounds)
$AGENT_MEMORY

---

Now analyze the situation and act. Use bash curl commands to submit your action, vote on mechanisms, and optionally propose a mechanism. Update your memory file when done. Be concise and efficient."

  # 7. Invoke Claude CLI
  echo "[$NAME] Invoking Claude for round $ROUND..." >> "$LOG_FILE"
  CLAUDE_START=$(date +%s)

  echo "$ROUND_PROMPT" | env -u CLAUDECODE \
    BASE="$BASE" API_KEY="$API_KEY" MEMBER_ID="$MEMBER_ID" ROUND="$ROUND" MEMORY_FILE="$MEMORY_FILE" \
    claude \
    -p \
    --dangerously-skip-permissions \
    --model haiku \
    --allowedTools "Bash(description:*)" \
    >> "$LOG_FILE" 2>&1

  CLAUDE_END=$(date +%s)
  CLAUDE_DURATION=$((CLAUDE_END - CLAUDE_START))
  echo "[$NAME] Claude done (${CLAUDE_DURATION}s)" >> "$LOG_FILE"

  last_round=$ROUND

  # 8. Brief sleep before next poll (server pace is ~30s)
  sleep 5
done
