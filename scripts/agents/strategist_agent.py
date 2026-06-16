#!/usr/bin/env python3
"""
Strategist-Claude: Adaptive data-driven agent for Leviathan.
Runs indefinitely — polls each round, analyzes state, acts, votes.
"""
import time
import requests
from collections import deque

BASE = "https://leviathan.fly.dev"
AGENT_NAME = "Strategist-Claude"
AGENT_DESC = "Adaptive data-driven strategist"

# ── Memory ──────────────────────────────────────────────────────────────────

memory = {
    "vitality_history": deque(maxlen=5),
    "cargo_history": deque(maxlen=5),
    "land_history": deque(maxlen=5),
    "attackers": set(),       # member ids that attacked me
    "allies": set(),          # member ids that traded with me
    "recent_deaths": [],      # ids that died last round
    "round_count": 0,
}

# ── Registration ─────────────────────────────────────────────────────────────

def register():
    while True:
        try:
            r = requests.post(f"{BASE}/v1/agents/register", json={
                "name": AGENT_NAME,
                "description": AGENT_DESC,
            }, timeout=10)
            if r.status_code == 200:
                data = r.json()
                print(f"[REGISTER] agent_id={data['agent_id']} member_id={data['member_id']}")
                return data["api_key"], data["member_id"]
            elif r.status_code == 409:
                print("[REGISTER] All slots taken. Waiting 30s...")
                time.sleep(30)
            else:
                print(f"[REGISTER] Error {r.status_code}: {r.text[:200]}")
                time.sleep(10)
        except Exception as e:
            print(f"[REGISTER] Exception: {e}. Retrying in 10s...")
            time.sleep(10)

# ── Strategy Logic ────────────────────────────────────────────────────────────

def choose_strategy(me, others, metrics, mem):
    """Return (strategy_name, action_code_string)."""
    vit = me["vitality"]
    hist = list(mem["vitality_history"])

    # Trend: positive = growing, negative = declining
    declining = len(hist) >= 3 and all(hist[i] >= hist[i+1] for i in range(len(hist)-1, max(len(hist)-3, 0), -1))

    # Identify weak isolated targets (low vitality, low land)
    candidates = [m for m in others if m["vitality"] < 40 and m["land_num"] < 3]
    weakest_id = min(others, key=lambda m: m["vitality"])["id"] if others else None
    strongest_id = max(others, key=lambda m: m["vitality"])["id"] if others else None

    gini = metrics.get("gini_coefficient", 0)
    conflict = metrics.get("conflict_count", 0)
    recent_deaths = mem.get("recent_deaths", [])

    # --- DEFENSIVE ---
    if vit < 35 or declining:
        strategy = "DEFENSIVE"
        code = f"""
def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    others = [m for m in execution_engine.current_members if m.id != me.id]
    execution_engine.expand(me)
    if others:
        strongest = max(others, key=lambda m: m.vitality)
        execution_engine.offer(me, strongest)
"""
    # --- OPPORTUNISTIC: someone just died or lost big ---
    elif recent_deaths:
        strategy = "OPPORTUNISTIC"
        code = f"""
def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    execution_engine.expand(me)
    execution_engine.expand(me)
"""
    # --- AGGRESSIVE: strong and there's a weak target ---
    elif vit > 65 and candidates:
        target_id = min(candidates, key=lambda m: m["vitality"])["id"]
        strategy = "AGGRESSIVE"
        code = f"""
def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    others = [m for m in execution_engine.current_members if m.id != me.id]
    target = next((m for m in others if m.id == {target_id}), None)
    if target:
        execution_engine.attack(me, target)
    execution_engine.expand(me)
"""
    # --- BALANCED ---
    else:
        strategy = "BALANCED"
        # Trade with the rising agent (top vitality) for alliance
        trade_id = strongest_id
        code = f"""
def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    others = [m for m in execution_engine.current_members if m.id != me.id]
    execution_engine.expand(me)
    if others:
        partner = next((m for m in others if m.id == {trade_id}), None)
        if partner:
            execution_engine.offer(me, partner)
"""
    return strategy, code.strip()

# ── Mechanism Proposal ────────────────────────────────────────────────────────

def propose_productivity_boost(headers, round_id):
    """Propose +10% base productivity boost — benefits everyone."""
    code = """
def propose_modification(execution_engine):
    for member in execution_engine.current_members:
        member.cargo = member.cargo * 1.1
"""
    try:
        r = requests.post(f"{BASE}/v1/world/mechanisms/propose", headers=headers, json={
            "code": code.strip(),
            "description": "Collective productivity boost +10% cargo for all members",
            "idempotency_key": f"round-{round_id}-mech-productivity",
        }, timeout=10)
        print(f"[PROPOSE] {r.status_code} {r.text[:100]}")
    except Exception as e:
        print(f"[PROPOSE] Error: {e}")

# ── Voting ────────────────────────────────────────────────────────────────────

def vote_on_mechanisms(headers):
    try:
        r = requests.get(f"{BASE}/v1/world/mechanisms?status=pending_vote", timeout=10)
        if r.status_code != 200:
            return
        mechanisms = r.json() if isinstance(r.json(), list) else r.json().get("mechanisms", [])
        for mech in mechanisms:
            mid = mech.get("id") or mech.get("mechanism_id")
            if not mid:
                continue
            # Get detailed canary report
            try:
                dr = requests.get(f"{BASE}/v1/world/mechanisms/{mid}", timeout=10)
                if dr.status_code != 200:
                    continue
                detail = dr.json()
            except Exception:
                continue

            canary = detail.get("canary_result") or detail.get("canary") or {}
            deaths = canary.get("agents_died", [])
            vitality_change = canary.get("vitality_change_pct", 0)
            execution_error = canary.get("execution_error") or detail.get("execution_error")

            vote = True
            reason = "canary clean"
            if execution_error:
                vote, reason = False, f"execution error: {execution_error[:60]}"
            elif deaths:
                vote, reason = False, f"agents died in canary: {deaths}"
            elif vitality_change < -10:
                vote, reason = False, f"vitality dropped {vitality_change:.1f}%"

            try:
                vr = requests.post(f"{BASE}/v1/world/mechanisms/{mid}/vote",
                                   headers=headers, json={"vote": vote, "idempotency_key": f"vote-{mid}"}, timeout=10)
                print(f"[VOTE] mech={mid} vote={vote} ({reason}) -> {vr.status_code}")
            except Exception as e:
                print(f"[VOTE] Error voting on {mid}: {e}")
    except Exception as e:
        print(f"[VOTE] Error fetching mechanisms: {e}")

# ── Event Processing ──────────────────────────────────────────────────────────

def process_events(last_round, member_id, mem):
    try:
        r = requests.get(f"{BASE}/v1/world/events?since_round={last_round}", timeout=10)
        if r.status_code != 200:
            return
        events = r.json() if isinstance(r.json(), list) else r.json().get("events", [])
        deaths_this_round = []
        for ev in events:
            etype = ev.get("type", "")
            if etype == "attack":
                if ev.get("target_id") == member_id:
                    mem["attackers"].add(ev.get("actor_id"))
            elif etype == "offer":
                if ev.get("target_id") == member_id:
                    mem["allies"].add(ev.get("actor_id"))
            elif etype in ("death", "member_died"):
                did = ev.get("member_id") or ev.get("id")
                if did and did != member_id:
                    deaths_this_round.append(did)
        mem["recent_deaths"] = deaths_this_round
    except Exception as e:
        print(f"[EVENTS] Error: {e}")

# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    # Try saved credentials first, fall back to register()
    import json, os
    cred_file = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "agents", "credentials.json")
    try:
        creds = json.load(open(cred_file))
        if "Strategist-Claude" in creds:
            c = creds["Strategist-Claude"]
            api_key, member_id = c["api_key"], c["member_id"]
            print(f"[START] Using saved credentials: member_id={member_id}")
        else:
            api_key, member_id = register()
    except Exception:
        api_key, member_id = register()
    headers = {"X-API-Key": api_key}
    last_round = -1

    print(f"[START] Playing as member_id={member_id}")

    while True:
        try:
            # (a) Poll deadline
            dl_r = requests.get(f"{BASE}/v1/world/rounds/current/deadline", timeout=10)
            if dl_r.status_code != 200:
                time.sleep(5)
                continue
            dl = dl_r.json()
            if dl["state"] != "accepting" or dl["round_id"] == last_round:
                time.sleep(5)
                continue

            round_id = dl["round_id"]

            # (b) Snapshot
            snap_r = requests.get(f"{BASE}/v1/world/snapshot", timeout=10)
            if snap_r.status_code != 200:
                time.sleep(5)
                continue
            snap = snap_r.json()
            members = snap.get("members", [])

            # Check if I'm still alive
            me_data = next((m for m in members if m["id"] == member_id), None)
            if me_data is None:
                print(f"[DEAD] Member {member_id} is gone. Re-registering...")
                api_key, member_id = register()
                headers = {"X-API-Key": api_key}
                memory["vitality_history"].clear()
                memory["cargo_history"].clear()
                memory["land_history"].clear()
                memory["attackers"].clear()
                memory["allies"].clear()
                memory["round_count"] = 0
                last_round = -1
                continue

            others = [m for m in members if m["id"] != member_id]

            # (c) Metrics
            met_r = requests.get(f"{BASE}/v1/world/metrics", timeout=10)
            metrics = met_r.json() if met_r.status_code == 200 else {}

            # Update history
            memory["vitality_history"].append(me_data["vitality"])
            memory["cargo_history"].append(me_data["cargo"])
            memory["land_history"].append(me_data["land_num"])
            memory["round_count"] += 1

            # (d-e) Choose strategy
            strategy, code = choose_strategy(me_data, others, metrics, memory)

            # (f) Submit action
            ar = requests.post(f"{BASE}/v1/world/actions", headers=headers, json={
                "code": code,
                "idempotency_key": f"round-{round_id}-strategist",
            }, timeout=10)
            status = ar.json().get("status", "?") if ar.status_code == 200 else f"err{ar.status_code}"

            print(f"[R{round_id}] {strategy:12s} | vit={me_data['vitality']:.1f} "
                  f"cargo={me_data['cargo']:.1f} land={me_data['land_num']} | {status}")

            # (g) Process events
            process_events(last_round, member_id, memory)

            # (h) Vote on mechanisms
            vote_on_mechanisms(headers)

            # (i) Propose mechanism every 10 rounds if in top 3
            if memory["round_count"] % 10 == 0 and others:
                rank = sorted(members, key=lambda m: m["vitality"], reverse=True)
                my_rank = next((i for i, m in enumerate(rank) if m["id"] == member_id), 99)
                if my_rank < 3:
                    propose_productivity_boost(headers, round_id)

            last_round = round_id

        except KeyboardInterrupt:
            print("\n[EXIT] Stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
