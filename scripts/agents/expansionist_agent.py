#!/usr/bin/env python3
"""
Expansionist-Claude — persistent Leviathan agent.
Personality: THE EXPANSIONIST — territory control above all else.
"""

import time
import requests

BASE = "https://leviathan.fly.dev"
AGENT_NAME = "Expansionist-Claude"
AGENT_DESC = "Territory-focused expansion agent — claims land relentlessly"

# ── Registration ──────────────────────────────────────────────────────────────

def register():
    """Register or re-register with the server. Returns (api_key, member_id)."""
    while True:
        try:
            r = requests.post(f"{BASE}/v1/agents/register", json={
                "name": AGENT_NAME,
                "description": AGENT_DESC,
            }, timeout=10)
            if r.status_code == 409:
                print("[register] All member slots taken — waiting 15s for a slot to open...")
                time.sleep(15)
                continue
            r.raise_for_status()
            data = r.json()
            api_key = data["api_key"]
            member_id = data["member_id"]
            print(f"[register] Joined as member {member_id} | key {api_key[:12]}...")
            return api_key, member_id
        except Exception as e:
            print(f"[register] Error: {e} — retrying in 5s...")
            time.sleep(5)

# ── Snapshot helpers ──────────────────────────────────────────────────────────

def get_snapshot():
    try:
        r = requests.get(f"{BASE}/v1/world/snapshot", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[snapshot] Error: {e}")
        return None

def get_deadline():
    try:
        r = requests.get(f"{BASE}/v1/world/rounds/current/deadline", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[deadline] Error: {e}")
        return None

# ── Strategy code generation ──────────────────────────────────────────────────

def build_action_code(me_info, all_members):
    """
    Expansionist strategy:
    - ALWAYS expand (primary goal)
    - Attack weakest neighbor if we're strong (vitality > 60, target vitality < 30)
    - Offer to strongest if we're in danger (vitality < 25)
    """
    vitality = me_info.get("vitality", 50)
    cargo = me_info.get("cargo", 0)
    land_num = me_info.get("land_num", 1)

    others = [m for m in all_members if m.get("id") != me_info.get("id")]

    weak_targets = [m for m in others if m.get("vitality", 100) < 30]
    has_weak_target = len(weak_targets) > 0

    strategy_comment = "expand"
    if vitality < 25 and others:
        strategy_comment = "desperate_offer"
    elif vitality > 60 and cargo > 10 and has_weak_target:
        strategy_comment = "attack_then_expand"

    print(f"  State: vitality={vitality:.1f} cargo={cargo:.1f} land={land_num} "
          f"strategy={strategy_comment} neighbors={len(others)} weak={len(weak_targets)}")

    code = f"""def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    others = [m for m in execution_engine.current_members if m.id != me.id]

    strategy = "{strategy_comment}"

    if strategy == "desperate_offer" and others:
        # Low vitality — curry favour with the strongest to avoid being eaten
        strongest = max(others, key=lambda m: m.vitality)
        execution_engine.offer(me, strongest)
        return  # conserve cargo, skip expansion this round

    if strategy == "attack_then_expand" and others:
        # Strong enough to bully — hit the weakest first, then expand
        weakest = min(others, key=lambda m: m.vitality)
        if weakest.vitality < 35:
            execution_engine.attack(me, weakest)

    # Primary objective: always grab more land
    execution_engine.expand(me)
"""
    return code

# ── Mechanism voting ──────────────────────────────────────────────────────────

GOOD_KEYWORDS = ["productivity", "land", "produce", "food", "growth", "territory", "expand"]
BAD_KEYWORDS  = ["redistribute", "tax", "reduce", "share", "equal", "shrink", "transfer"]

def vote_on_mechanisms(headers):
    try:
        r = requests.get(f"{BASE}/v1/world/mechanisms?status=pending_vote", timeout=10)
        if r.status_code != 200:
            return
        mechanisms = r.json()
        if not mechanisms:
            return
        for mech in mechanisms:
            mech_id = mech.get("mechanism_id") or mech.get("id")
            desc = (mech.get("description") or "").lower()
            vote = True
            for bad in BAD_KEYWORDS:
                if bad in desc:
                    vote = False
                    break
            # Override: if desc mentions expanding/land productivity, definitely yes
            for good in GOOD_KEYWORDS:
                if good in desc:
                    vote = True
                    break
            try:
                vr = requests.post(
                    f"{BASE}/v1/world/mechanisms/{mech_id}/vote",
                    headers=headers,
                    json={"vote": vote, "idempotency_key": f"vote-{mech_id}"},
                    timeout=10,
                )
                print(f"  [vote] mechanism {mech_id}: {vote} | desc='{desc[:60]}'")
            except Exception as e:
                print(f"  [vote] failed for {mech_id}: {e}")
    except Exception as e:
        print(f"[mechanisms] Error listing: {e}")

# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    # Try pre-registered credentials first, fall back to register()
    import json, os
    cred_file = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "agents", "credentials.json")
    try:
        creds = json.load(open(cred_file))
        if "Expansionist-Claude" in creds:
            c = creds["Expansionist-Claude"]
            api_key, member_id = c["api_key"], c["member_id"]
            print(f"[expansionist] Using saved credentials: member_id={member_id}")
        else:
            api_key, member_id = register()
    except Exception:
        api_key, member_id = register()
    headers = {"X-API-Key": api_key}
    last_round = -1

    print(f"\n[expansionist] Starting persistent loop. Member index: {member_id}\n")

    while True:
        try:
            # (a) Poll for accepting state
            dl = get_deadline()
            if dl is None:
                time.sleep(5)
                continue

            state = dl.get("state")
            round_id = dl.get("round_id", -1)
            secs = dl.get("seconds_remaining", 0)

            if state != "accepting" or round_id == last_round:
                time.sleep(5)
                continue

            print(f"\n[round {round_id}] window open ({secs:.1f}s remaining)")

            # (b) Get world snapshot
            snap = get_snapshot()
            if snap is None:
                time.sleep(5)
                continue

            members = snap.get("members", [])

            # (c) Find myself — if not present, we're dead
            me_list = [m for m in members if m.get("id") == member_id]
            if not me_list:
                print(f"[round {round_id}] Member {member_id} not found — DEAD. Re-registering...")
                api_key, member_id = register()
                headers = {"X-API-Key": api_key}
                last_round = -1
                continue

            me_info = me_list[0]

            # (d) Generate strategy code
            code = build_action_code(me_info, members)

            # (e) Submit action
            idem_key = f"round-{round_id}-expand"
            r = requests.post(f"{BASE}/v1/world/actions", headers=headers, json={
                "code": code,
                "idempotency_key": idem_key,
            }, timeout=10)

            if r.status_code == 401 or r.status_code == 403:
                print(f"[round {round_id}] Auth error ({r.status_code}) — re-registering...")
                api_key, member_id = register()
                headers = {"X-API-Key": api_key}
                last_round = -1
                continue

            resp = r.json() if r.ok else {"status": "error", "detail": r.text[:100]}
            print(f"  [action] {resp.get('status', '?')} | round={resp.get('round_id', round_id)}")

            last_round = round_id

            # (f) Vote on pending mechanisms
            vote_on_mechanisms(headers)

        except KeyboardInterrupt:
            print("\n[expansionist] Interrupted — stopping.")
            break
        except Exception as e:
            print(f"[loop] Unexpected error: {e} — retrying in 3s...")
            time.sleep(3)

        # (g) Brief sleep before next poll
        time.sleep(5)


if __name__ == "__main__":
    run()
