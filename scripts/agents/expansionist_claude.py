#!/usr/bin/env python3
"""
Expansionist-Claude — persistent Leviathan agent.

Personality: THE EXPANSIONIST
- Always expand territory first
- Attack weak neighbors when vitality is high
- Offer resources to strong neighbors when vitality is critical
- Vote YES on productivity/territory mechanisms, NO on redistribution
"""

import time
import sys
import requests

BASE = "https://leviathan.fly.dev"
AGENT_NAME = "Expansionist-Claude"
AGENT_DESC = "Territory-focused expansion agent"

API_KEY = None
MEMBER_ID = None
headers = {}


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def register():
    global API_KEY, MEMBER_ID, headers
    log(f"Registering as '{AGENT_NAME}'...")
    try:
        r = requests.post(f"{BASE}/v1/agents/register", json={
            "name": AGENT_NAME,
            "description": AGENT_DESC,
        }, timeout=10)
        if r.status_code == 200:
            data = r.json()
            API_KEY = data["api_key"]
            MEMBER_ID = data["member_id"]
            headers = {"X-API-Key": API_KEY}
            log(f"Registered: member_id={MEMBER_ID}, key={API_KEY[:16]}...")
            return True
        elif r.status_code == 409:
            log("All slots taken (409). Will retry in 30s...")
            return False
        else:
            log(f"Registration failed: {r.status_code} {r.text[:120]}")
            return False
    except Exception as e:
        log(f"Registration error: {e}")
        return False


def _get(url: str, desc: str, timeout=10, retries=2):
    """GET with 429 backoff."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = 5 * (attempt + 1)
                log(f"Rate limited on {desc}, waiting {wait}s...")
                time.sleep(wait)
            else:
                log(f"{desc} error: {r.status_code}")
                return None
        except Exception as e:
            log(f"{desc} exception: {e}")
            if attempt < retries:
                time.sleep(2)
    return None


def get_deadline():
    return _get(f"{BASE}/v1/world/rounds/current/deadline", "deadline", timeout=5)


def get_snapshot():
    return _get(f"{BASE}/v1/world/snapshot", "snapshot", timeout=10)


def submit_action(code: str, round_id: int):
    try:
        r = requests.post(f"{BASE}/v1/world/actions", headers=headers, json={
            "code": code,
            "idempotency_key": f"round-{round_id}-expand",
        }, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            log(f"Action submit error: {r.status_code} {r.text[:120]}")
            return None
    except Exception as e:
        log(f"Action submit exception: {e}")
        return None


def get_pending_mechanisms():
    try:
        r = requests.get(f"{BASE}/v1/world/mechanisms?status=pending_vote", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log(f"Mechanisms poll error: {e}")
    return []


def get_mechanism_detail(mechanism_id: str):
    try:
        r = requests.get(f"{BASE}/v1/world/mechanisms/{mechanism_id}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def cast_vote(mechanism_id: str, vote: bool, round_id: int):
    try:
        r = requests.post(
            f"{BASE}/v1/world/mechanisms/{mechanism_id}/vote",
            headers=headers,
            json={
                "vote": vote,
                "idempotency_key": f"vote-{mechanism_id}-r{round_id}",
            },
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()
        else:
            log(f"Vote error: {r.status_code} {r.text[:80]}")
    except Exception as e:
        log(f"Vote exception: {e}")
    return None


def decide_vote(detail) -> bool:
    """Vote YES on productivity/territory mechanisms, NO on redistribution."""
    if detail is None:
        return True  # assume good faith if no detail

    desc = (detail.get("description") or "").lower()
    code = (detail.get("code") or "").lower()
    combined = desc + " " + code

    # Explicit NO signals: redistribution, land transfer, reduce territory
    no_keywords = [
        "redistribut", "tax", "equal", "transfer land",
        "reduce expansion", "shrink", "punish", "penalty",
        "land_num -= ", "land_num = land_num -",
    ]
    for kw in no_keywords:
        if kw in combined:
            log(f"  Voting NO: found '{kw}' in mechanism")
            return False

    # Canary flags are a strong signal against
    canary = detail.get("canary_report") or {}
    flags = canary.get("divergence_flags", [])
    if flags:
        log(f"  Voting NO: canary flags {flags}")
        return False

    # YES signals: productivity, land, territory, resources, growth
    yes_keywords = [
        "product", "land", "territory", "expand", "growth",
        "bonus", "vitality", "resource", "cargo", "protect",
    ]
    for kw in yes_keywords:
        if kw in combined:
            log(f"  Voting YES: found '{kw}' in mechanism")
            return True

    # Default: YES (optimistic)
    return True


def build_action_code(snapshot, round_id: int) -> str:
    """Analyze snapshot and build strategy code for this round."""
    members = snapshot.get("members", [])

    me_state = None
    for m in members:
        if m.get("id") == MEMBER_ID:
            me_state = m
            break

    if me_state is None:
        log("WARNING: my member not found in snapshot — may be dead")
        return None

    vitality = me_state.get("vitality", 50)
    cargo = me_state.get("cargo", 0)
    land_num = me_state.get("land_num", 0)

    others = [m for m in members if m.get("id") != MEMBER_ID]
    weakest = min(others, key=lambda m: m.get("vitality", 999)) if others else None
    strongest = max(others, key=lambda m: m.get("vitality", 0)) if others else None

    log(f"State: vitality={vitality:.1f}, cargo={cargo:.1f}, land={land_num}")

    # Strategy decision
    if vitality < 25 and strongest:
        strategy = "survive"
        target_id = strongest["id"]
        log(f"Strategy: SURVIVE — offer to strongest (id={target_id}, v={strongest['vitality']:.1f})")
    elif vitality > 60 and weakest and weakest.get("vitality", 100) < 30 and len(others) > 0:
        strategy = "attack"
        target_id = weakest["id"]
        log(f"Strategy: ATTACK weakest (id={target_id}, v={weakest['vitality']:.1f}) then EXPAND")
    else:
        strategy = "expand"
        log(f"Strategy: EXPAND (default)")

    # Build the code
    if strategy == "survive":
        code = f"""
def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    others = [m for m in execution_engine.current_members if m.id != me.id]
    if others:
        strongest = max(others, key=lambda m: m.vitality)
        execution_engine.offer(me, strongest)
    execution_engine.expand(me)
"""
    elif strategy == "attack":
        code = f"""
def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    others = [m for m in execution_engine.current_members if m.id != me.id]
    if others:
        weakest = min(others, key=lambda m: m.vitality)
        if weakest.vitality < 30:
            execution_engine.attack(me, weakest)
    execution_engine.expand(me)
"""
    else:
        code = """
def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    execution_engine.expand(me)
"""

    return code.strip()


def is_member_alive(snapshot) -> bool:
    if snapshot is None:
        return True  # can't confirm death
    members = snapshot.get("members", [])
    return any(m.get("id") == MEMBER_ID for m in members)


def main():
    global API_KEY, MEMBER_ID, headers

    log("=" * 60)
    log("  THE EXPANSIONIST — Leviathan Persistent Agent")
    log("  Objective: territorial domination, never stop")
    log("=" * 60)

    # Register with retry
    while True:
        if register():
            break
        time.sleep(30)

    last_round = -1

    while True:
        try:
            # Step a: Poll until accepting
            dl = get_deadline()
            if dl is None:
                time.sleep(2)
                continue

            state = dl.get("state")
            round_id = dl.get("round_id", -1)
            remaining = dl.get("seconds_remaining", 0)

            if state != "accepting" or round_id == last_round:
                time.sleep(2)
                continue

            log(f"--- Round {round_id} open ({remaining:.1f}s remaining) ---")

            # Step b: Get world snapshot (best-effort; use fallback if unavailable)
            snapshot = get_snapshot()

            # Step c: Check if still alive
            if not is_member_alive(snapshot):
                log("My member is dead! Re-registering...")
                API_KEY = None
                MEMBER_ID = None
                headers = {}
                while True:
                    if register():
                        break
                    time.sleep(30)
                # Re-fetch snapshot after re-register
                snapshot = get_snapshot()
                if snapshot is None:
                    last_round = round_id
                    time.sleep(2)
                    continue

            # Step d: Build and submit action
            code = build_action_code(snapshot, round_id)
            if code is None:
                # Fallback: just expand
                code = """
def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    execution_engine.expand(me)
""".strip()

            result = submit_action(code, round_id)
            if result:
                status = result.get("status", "unknown")
                log(f"Action submitted: {status}")
            else:
                log("Action submit failed")

            last_round = round_id

            # Step f: Vote on pending mechanisms
            pending = get_pending_mechanisms()
            if pending:
                log(f"Voting on {len(pending)} pending mechanism(s)...")
                for mech in pending:
                    mid = mech.get("mechanism_id", "")
                    detail = get_mechanism_detail(mid)
                    vote = decide_vote(detail)
                    vote_result = cast_vote(mid, vote, round_id)
                    if vote_result:
                        yes = vote_result.get("current_votes", {}).get("yes", "?")
                        no = vote_result.get("current_votes", {}).get("no", "?")
                        log(f"  Voted {'YES' if vote else 'NO'} on {mid[:12]}... (yes={yes}, no={no})")

        except KeyboardInterrupt:
            log("Interrupted by user. Exiting.")
            sys.exit(0)
        except Exception as e:
            log(f"Unexpected error: {type(e).__name__}: {e}")
            time.sleep(3)


if __name__ == "__main__":
    main()
