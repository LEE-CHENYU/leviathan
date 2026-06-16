#!/usr/bin/env python3
"""Strategist-Claude: adaptive data-driven agent for Leviathan at leviathan.fly.dev.

Runs indefinitely, choosing DEFENSIVE/BALANCED/AGGRESSIVE/OPPORTUNISTIC strategy
each round based on vitality trend, threats, and world inequality metrics.
"""

import json
import time
import sys
from collections import deque
from typing import Optional, Dict, List, Any

import requests

BASE_URL = "https://leviathan.fly.dev"
AGENT_NAME = "Strategist-Claude"
AGENT_DESC = "Adaptive data-driven strategist"
POLL_INTERVAL = 2  # seconds between deadline polls


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def get(path: str, key: Optional[str] = None, **kwargs) -> Optional[Any]:
    headers = {"X-API-Key": key} if key else {}
    try:
        r = requests.get(f"{BASE_URL}{path}", headers=headers, timeout=8, **kwargs)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  [GET {path}] error: {e}")
    return None


def post(path: str, body: dict, key: Optional[str] = None) -> Optional[Any]:
    headers = {"X-API-Key": key} if key else {}
    try:
        r = requests.post(f"{BASE_URL}{path}", json=body, headers=headers, timeout=8)
        if r.status_code == 200:
            return r.json()
        print(f"  [POST {path}] {r.status_code}: {r.text[:120]}")
    except Exception as e:
        print(f"  [POST {path}] error: {e}")
    return None


# ── Registration ──────────────────────────────────────────────────────────────

def register() -> Optional[Dict]:
    """Register agent; returns {api_key, member_id, agent_id} or None."""
    result = post("/v1/agents/register", {
        "name": AGENT_NAME,
        "description": AGENT_DESC,
    })
    if result and "api_key" in result:
        print(f"  [+] Registered: member_id={result['member_id']} key={result['api_key'][:12]}...")
        return result
    print(f"  [!] Registration failed: {result}")
    return None


# ── Action code generation ─────────────────────────────────────────────────────

def make_action_code(strategy: str, my_id: int, threat_id: Optional[int], weak_id: Optional[int]) -> str:
    """Generate agent_action code based on chosen strategy."""

    if strategy == "DEFENSIVE":
        return f"""\
def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    execution_engine.expand(me)
"""
    elif strategy == "BALANCED":
        target_clause = ""
        if weak_id is not None:
            target_clause = f"""\
    target = next((m for m in members if getattr(m, 'id', None) == {weak_id}), None)
    if target and getattr(me, 'cargo', 0) > 5:
        execution_engine.offer(me, target)
        return
"""
        return f"""\
def agent_action(execution_engine, member_index):
    members = execution_engine.current_members
    me = members[member_index]
{target_clause}    execution_engine.expand(me)
"""
    elif strategy == "AGGRESSIVE":
        attack_clause = ""
        if weak_id is not None:
            attack_clause = f"""\
    target = next((m for m in members if getattr(m, 'id', None) == {weak_id}), None)
    if target and getattr(target, 'vitality', 100) < getattr(me, 'vitality', 0) * 0.7:
        execution_engine.attack(me, target)
        return
"""
        return f"""\
def agent_action(execution_engine, member_index):
    members = execution_engine.current_members
    me = members[member_index]
{attack_clause}    execution_engine.expand(me)
"""
    else:  # OPPORTUNISTIC
        return f"""\
def agent_action(execution_engine, member_index):
    members = execution_engine.current_members
    me = members[member_index]
    execution_engine.expand(me)
    # Grab unclaimed territory opportunistically
    execution_engine.expand(me)
"""


def make_mechanism_code(my_id: int, round_num: int) -> str:
    """Propose a mechanism that gives a small vitality boost to the top performers."""
    return f"""\
def propose_modification(execution_engine):
    # Safety bonus: reward members with land for responsible expansion
    members = execution_engine.current_members
    for m in members:
        land = getattr(m, 'land_num', 0)
        if land > 2:
            m.vitality = m.vitality + 0.5
"""


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(memory: Dict, snapshot: Dict, metrics: Optional[Dict], my_id: int) -> Dict:
    """Deep analysis returning strategy + threat/opportunity IDs."""
    members = snapshot.get("members", [])
    me = next((m for m in members if m.get("id") == my_id), None)

    my_vitality = me.get("vitality", 0) if me else 0
    my_cargo = me.get("cargo", 0) if me else 0
    my_land = me.get("land_num", 0) if me else 0

    # Update history
    memory["vitality_hist"].append(my_vitality)
    memory["cargo_hist"].append(my_cargo)

    hist = list(memory["vitality_hist"])
    declining = len(hist) >= 3 and hist[-1] < hist[-2] < hist[-3]

    # Identify others
    others = [m for m in members if m.get("id") != my_id and m.get("vitality", 0) > 0]
    others_sorted = sorted(others, key=lambda m: m.get("vitality", 0))

    threat_id = None
    weak_id = None
    if others_sorted:
        weak = others_sorted[0]
        strong = others_sorted[-1]
        weak_id = weak.get("id")
        if strong.get("vitality", 0) > my_vitality * 1.3:
            threat_id = strong.get("id")

    # Track attacks from events
    attackers = memory.get("attackers", set())

    # Choose strategy
    gini = (metrics or {}).get("gini_coefficient", 0)
    conflict = (metrics or {}).get("conflict_count", 0)

    if my_vitality < 35 or declining:
        strategy = "DEFENSIVE"
    elif my_vitality > 65 and weak_id and others_sorted[0].get("vitality", 100) < 30:
        strategy = "AGGRESSIVE"
    elif conflict > 3 and my_vitality < 50:
        strategy = "DEFENSIVE"
    elif weak_id and others_sorted[0].get("vitality", 100) < 20:
        strategy = "OPPORTUNISTIC"
    else:
        strategy = "BALANCED"

    return {
        "strategy": strategy,
        "my_vitality": my_vitality,
        "my_cargo": my_cargo,
        "my_land": my_land,
        "threat_id": threat_id,
        "weak_id": weak_id,
        "gini": gini,
        "declining": declining,
        "population": len(members),
    }


# ── Voting ────────────────────────────────────────────────────────────────────

def vote_on_mechanisms(api_key: str, round_id: int):
    pending = get("/v1/world/mechanisms?status=pending_vote") or []
    for mech in pending:
        mid = mech.get("mechanism_id", "")
        if not mid:
            continue
        detail = get(f"/v1/world/mechanisms/{mid}")
        if not detail:
            continue
        # Evaluate canary report
        canary = detail.get("canary_report") or {}
        flags = canary.get("divergence_flags", [])
        deaths = canary.get("deaths", 0) or 0
        vitality_change_pct = canary.get("vitality_change_pct", 0) or 0
        vote_yes = len(flags) == 0 and deaths == 0 and vitality_change_pct > -10
        result = post(f"/v1/world/mechanisms/{mid}/vote", {
            "vote": vote_yes,
            "idempotency_key": f"strategist-vote-{mid[:8]}-r{round_id}",
        }, key=api_key)
        if result:
            decision = "YES" if vote_yes else "NO"
            print(f"  [#] Voted {decision} on mechanism {mid[:8]}... "
                  f"(flags={len(flags)}, deaths={deaths}, vit_chg={vitality_change_pct:.1f}%)")


# ── Event ingestion ───────────────────────────────────────────────────────────

def ingest_events(memory: Dict, since_round: int):
    events = get(f"/v1/world/events?since_round={since_round}") or []
    for ev in events:
        etype = ev.get("event_type", "")
        data = ev.get("data", {})
        if etype == "attack":
            attacker = data.get("attacker_id")
            if attacker:
                memory.setdefault("attackers", set()).add(attacker)
        elif etype == "member_died":
            did = data.get("member_id")
            if did:
                memory.setdefault("recent_deaths", set()).add(did)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    memory: Dict = {
        "vitality_hist": deque(maxlen=5),
        "cargo_hist": deque(maxlen=5),
        "attackers": set(),
        "recent_deaths": set(),
        "rounds_played": 0,
    }

    api_key = None
    my_member_id = None
    last_round_id = 0

    print(f"\n{'='*60}")
    print(f"  STRATEGIST-CLAUDE — Leviathan Persistent Agent")
    print(f"  Target: {BASE_URL}")
    print(f"{'='*60}\n")

    # Initial registration
    reg = register()
    if reg:
        api_key = reg["api_key"]
        my_member_id = reg["member_id"]
    else:
        print("  [!] Could not register initially — will retry each loop")

    while True:
        try:
            # Re-register if we have no key
            if not api_key:
                reg = register()
                if reg:
                    api_key = reg["api_key"]
                    my_member_id = reg["member_id"]
                    memory = {
                        "vitality_hist": deque(maxlen=5),
                        "cargo_hist": deque(maxlen=5),
                        "attackers": set(),
                        "recent_deaths": set(),
                        "rounds_played": memory.get("rounds_played", 0),
                    }
                else:
                    time.sleep(5)
                    continue

            # Poll until accepting
            deadline = None
            while True:
                deadline = get("/v1/world/rounds/current/deadline")
                if deadline and deadline.get("state") == "accepting":
                    secs = deadline.get("seconds_remaining", 0)
                    if secs > 1.0:
                        break
                time.sleep(POLL_INTERVAL)

            round_id = deadline["round_id"]
            if round_id == last_round_id:
                time.sleep(POLL_INTERVAL)
                continue

            # Gather world state
            snapshot = get("/v1/world/snapshot") or {}
            metrics = get("/v1/world/metrics")

            # Verify we're still alive
            members = snapshot.get("members", [])
            me = next((m for m in members if m.get("id") == my_member_id), None)
            if not me:
                print(f"  [!] Member {my_member_id} not found — re-registering")
                api_key = None
                my_member_id = None
                continue

            # Ingest last round events
            if last_round_id > 0:
                ingest_events(memory, last_round_id)

            # Analyze & choose strategy
            analysis = analyze(memory, snapshot, metrics, my_member_id)
            strategy = analysis["strategy"]

            print(f"\n--- Round {round_id} | "
                  f"Strategy: {strategy} | "
                  f"vitality={analysis['my_vitality']:.1f} "
                  f"cargo={analysis['my_cargo']:.1f} "
                  f"land={analysis['my_land']} "
                  f"pop={analysis['population']} "
                  f"gini={analysis['gini']:.3f}")

            # Generate + submit action
            code = make_action_code(
                strategy, my_member_id,
                analysis["threat_id"], analysis["weak_id"]
            )
            result = post("/v1/world/actions", {
                "code": code,
                "idempotency_key": f"round-{round_id}-strategist",
            }, key=api_key)
            status = (result or {}).get("status", "error")
            print(f"  [>] Action submitted: {status}")

            # Vote on pending mechanisms
            vote_on_mechanisms(api_key, round_id)

            # Every 10 rounds, propose a mechanism if in top 3 by vitality
            rounds_played = memory["rounds_played"] + 1
            memory["rounds_played"] = rounds_played
            if rounds_played % 10 == 0:
                all_vitalities = sorted(
                    [m.get("vitality", 0) for m in members], reverse=True
                )
                top3_threshold = all_vitalities[2] if len(all_vitalities) >= 3 else 0
                if analysis["my_vitality"] >= top3_threshold:
                    mech_code = make_mechanism_code(my_member_id, round_id)
                    mech_result = post("/v1/world/mechanisms/propose", {
                        "code": mech_code,
                        "description": f"Strategist expansion reward mechanism (round {round_id})",
                        "idempotency_key": f"strategist-mech-{round_id}",
                    }, key=api_key)
                    if mech_result:
                        print(f"  [#] Mechanism proposed: {mech_result.get('mechanism_id', '?')[:12]}...")

            last_round_id = round_id
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\n  [~] Interrupted by user — stopping.")
            sys.exit(0)
        except Exception as e:
            print(f"  [!] Unexpected error: {type(e).__name__}: {e} — continuing")
            time.sleep(3)


if __name__ == "__main__":
    main()
