#!/usr/bin/env python3
"""
Diplomat-Claude — The Diplomat
Survival strategy: alliances, trade, political influence.
Runs indefinitely against https://leviathan.fly.dev
"""

import time
import requests
import json
from datetime import datetime

BASE = "https://leviathan.fly.dev"
AGENT_NAME = "Diplomat-Claude"
AGENT_DESC = "Alliance-building trade diplomat"

# ── registration state ────────────────────────────────────────────────────────

api_key = None
member_id = None
headers = {}


def log(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def register():
    global api_key, member_id, headers
    log(f"Registering as '{AGENT_NAME}'...")
    for attempt in range(10):
        try:
            r = requests.post(
                f"{BASE}/v1/agents/register",
                json={"name": AGENT_NAME, "description": AGENT_DESC},
                timeout=10,
            )
            if r.status_code == 409:
                log("All slots taken (409). Waiting 15s to retry...")
                time.sleep(15)
                continue
            r.raise_for_status()
            agent = r.json()
            api_key = agent["api_key"]
            member_id = agent["member_id"]
            headers = {"X-API-Key": api_key}
            log(f"Registered: member_id={member_id}, key={api_key[:12]}...")
            return True
        except Exception as e:
            log(f"Registration error (attempt {attempt+1}): {e}")
            time.sleep(5)
    return False


# ── deadline polling ──────────────────────────────────────────────────────────

def wait_for_accepting(last_round: int) -> dict | None:
    """Block until the server is accepting submissions for a new round."""
    while True:
        try:
            dl = requests.get(f"{BASE}/v1/world/rounds/current/deadline", timeout=5).json()
            state = dl.get("state")
            round_id = dl.get("round_id", -1)
            if state == "accepting" and round_id != last_round:
                return dl
            # Not yet — brief sleep
            time.sleep(5)
        except Exception as e:
            log(f"Deadline poll error: {e}")
            time.sleep(3)


# ── snapshot helpers ──────────────────────────────────────────────────────────

def get_snapshot() -> dict | None:
    try:
        r = requests.get(f"{BASE}/v1/world/snapshot", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"Snapshot error: {e}")
        return None


def is_member_alive(snap: dict) -> bool:
    """Check if our member_id is still present in the snapshot."""
    if not snap or "members" not in snap:
        return False
    members = snap["members"]
    if isinstance(members, list):
        return any(m.get("id") == member_id for m in members)
    # dict keyed by index
    return member_id < len(members)


def get_me(snap: dict) -> dict | None:
    members = snap.get("members", [])
    if isinstance(members, list):
        for m in members:
            if m.get("id") == member_id:
                return m
        # fallback: index-based
        if member_id < len(members):
            return members[member_id]
    return None


# ── action code generation ────────────────────────────────────────────────────

def build_action_code(snap: dict, me: dict) -> str:
    """
    Generate agent_action Python code based on current world state.
    Diplomat strategy:
      - Offer to the strongest (alliance building) if cargo > 5
      - Expand if vitality < 40 (survival)
      - Expand + offer if vitality 40-80 (growth + diplomacy)
      - Attack only if vitality > 80 and target vitality < 20
    """
    members = snap.get("members", [])
    if isinstance(members, dict):
        members = list(members.values())

    my_vitality = me.get("vitality", 50)
    my_cargo = me.get("cargo", 0)

    # Build summary of other members for logging
    others = [m for m in members if m.get("id") != member_id]
    if others:
        strongest = max(others, key=lambda m: m.get("vitality", 0) + m.get("cargo", 0) + m.get("land_num", 0))
        weakest = min(others, key=lambda m: m.get("vitality", 100))
        strongest_id = strongest.get("id")
        weakest_id = weakest.get("id")
        weakest_vitality = weakest.get("vitality", 100)
    else:
        strongest_id = None
        weakest_id = None
        weakest_vitality = 100

    log(f"  State: vitality={my_vitality:.1f} cargo={my_cargo:.1f} "
        f"members_alive={len(members)}")

    # Decide strategy
    if my_vitality < 40:
        strategy = "survive"
        log("  Strategy: SURVIVE (expand for resources)")
    elif my_vitality > 80 and weakest_vitality < 20 and weakest_id is not None:
        strategy = "attack_then_expand"
        log(f"  Strategy: ATTACK weakest (id={weakest_id}, v={weakest_vitality:.1f}) then expand")
    elif my_cargo > 5 and strongest_id is not None:
        strategy = "offer_then_expand"
        log(f"  Strategy: OFFER to strongest (id={strongest_id}) then expand")
    else:
        strategy = "expand"
        log("  Strategy: EXPAND (building territory)")

    # Build code string — keep it simple and safe
    lines = [
        "def agent_action(execution_engine, member_index):",
        "    me = execution_engine.current_members[member_index]",
        "    others = [m for m in execution_engine.current_members if m.id != me.id]",
    ]

    if strategy == "survive":
        lines += [
            "    execution_engine.expand(me)",
        ]
    elif strategy == "attack_then_expand":
        lines += [
            f"    target_id = {weakest_id}",
            "    target = next((m for m in others if m.id == target_id), None)",
            "    if target is not None:",
            "        execution_engine.attack(me, target)",
            "    execution_engine.expand(me)",
        ]
    elif strategy == "offer_then_expand":
        lines += [
            f"    ally_id = {strongest_id}",
            "    ally = next((m for m in others if m.id == ally_id), None)",
            "    if ally is not None:",
            "        execution_engine.offer(me, ally)",
            "    execution_engine.expand(me)",
        ]
    else:
        lines += [
            "    execution_engine.expand(me)",
            "    if others:",
            "        strongest = max(others, key=lambda m: m.vitality + m.cargo + m.land_num)",
            "        execution_engine.offer(me, strongest)",
        ]

    return "\n".join(lines)


# ── action submission ─────────────────────────────────────────────────────────

def submit_action(code: str, round_id: int) -> bool:
    try:
        r = requests.post(
            f"{BASE}/v1/world/actions",
            headers=headers,
            json={
                "code": code,
                "idempotency_key": f"round-{round_id}-diplomat",
            },
            timeout=10,
        )
        resp = r.json()
        status = resp.get("status", "unknown")
        log(f"  Action submitted: {status} (round {round_id})")
        return status == "accepted"
    except Exception as e:
        log(f"  Action submission error: {e}")
        return False


# ── mechanism voting ──────────────────────────────────────────────────────────

PRO_KEYWORDS = [
    "trade", "cooperat", "collective", "bonus", "insurance", "watchdog",
    "circuit", "audit", "floor", "minimum", "production", "recovery",
    "safety", "protect", "resource", "vitality floor",
]
ANTI_KEYWORDS = [
    "aggress", "attack bonus", "destroy", "kill", "unchecked", "unlimited attack",
]


def classify_mechanism(desc: str) -> bool:
    desc_lower = desc.lower()
    if any(kw in desc_lower for kw in ANTI_KEYWORDS):
        return False
    if any(kw in desc_lower for kw in PRO_KEYWORDS):
        return True
    return True  # Diplomat defaults to yes for community proposals


def vote_on_mechanisms():
    try:
        r = requests.get(f"{BASE}/v1/world/mechanisms", params={"status": "pending_vote"}, timeout=10)
        if r.status_code != 200:
            return
        mechanisms = r.json()
        if not mechanisms:
            return

        mech_list = mechanisms if isinstance(mechanisms, list) else mechanisms.get("mechanisms", [])
        log(f"  Found {len(mech_list)} mechanism(s) pending vote")

        for mech in mech_list:
            mech_id = mech.get("mechanism_id") or mech.get("id")
            desc = mech.get("description", "")
            canary = mech.get("canary_result", {}) or {}

            # Check canary results for empirical safety signal
            vitality_change = canary.get("vitality_change_pct", 0) or 0
            agents_died = canary.get("agents_died", []) or []
            divergence = canary.get("divergence_flags", []) or []

            vote = classify_mechanism(desc)

            # Override vote based on canary evidence
            if vitality_change < -30:
                vote = False
                log(f"  Mech {mech_id}: OVERRIDE NO — canary shows {vitality_change:.1f}% vitality drop")
            elif agents_died:
                vote = False
                log(f"  Mech {mech_id}: OVERRIDE NO — canary killed agents: {agents_died}")
            elif divergence:
                vote = False
                log(f"  Mech {mech_id}: OVERRIDE NO — divergence flags: {divergence}")
            else:
                log(f"  Mech {mech_id}: voting {'YES' if vote else 'NO'} — '{desc[:60]}'")

            try:
                vr = requests.post(
                    f"{BASE}/v1/world/mechanisms/{mech_id}/vote",
                    headers=headers,
                    json={"vote": vote, "idempotency_key": f"vote-{mech_id}"},
                    timeout=10,
                )
                log(f"  Vote response: {vr.status_code}")
            except Exception as e:
                log(f"  Vote error for {mech_id}: {e}")

    except Exception as e:
        log(f"  Mechanism voting error: {e}")


# ── mechanism proposals ───────────────────────────────────────────────────────

PROPOSALS = [
    {
        "description": "Establish minimum vitality floor: any member below 10 vitality receives 5 vitality from a collective emergency fund",
        "code": """\
def propose_modification(execution_engine):
    # Safety net: prevent starvation deaths from inactivity
    floor = 10.0
    gift = 5.0
    total_members = len(execution_engine.current_members)
    if total_members == 0:
        return
    for member in execution_engine.current_members:
        if member.vitality < floor:
            member.vitality = min(member.vitality + gift, floor + gift)
""",
    },
    {
        "description": "Trade bonus: members who performed an offer action this round gain +2 vitality (cooperative incentive)",
        "code": """\
def propose_modification(execution_engine):
    # Incentivize trade by rewarding offering members
    for member in execution_engine.current_members:
        if getattr(member, 'last_action', '') == 'offer':
            member.vitality = min(member.vitality + 2.0, 100.0)
""",
    },
    {
        "description": "Collective production boost: increase base land yield by 10% for all members to grow the economy",
        "code": """\
def propose_modification(execution_engine):
    # Boost collective wealth — rising tide lifts all boats
    for member in execution_engine.current_members:
        bonus = member.land_num * 0.1
        member.cargo = member.cargo + bonus
""",
    },
]

_proposal_index = 0


def propose_mechanism(round_id: int):
    global _proposal_index
    proposal = PROPOSALS[_proposal_index % len(PROPOSALS)]
    _proposal_index += 1

    log(f"  Proposing mechanism: '{proposal['description'][:60]}...'")
    try:
        r = requests.post(
            f"{BASE}/v1/world/mechanisms/propose",
            headers=headers,
            json={
                "code": proposal["code"],
                "description": proposal["description"],
                "idempotency_key": f"round-{round_id}-mech-{_proposal_index}",
            },
            timeout=10,
        )
        resp = r.json()
        log(f"  Proposal response: {resp.get('status', resp)}")
    except Exception as e:
        log(f"  Proposal error: {e}")


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    global api_key, member_id, headers
    log("=" * 60)
    log("DIPLOMAT-CLAUDE — The Alliance Builder")
    log("=" * 60)

    # Try saved credentials first, fall back to register()
    import json, os
    cred_file = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "agents", "credentials.json")
    try:
        creds = json.load(open(cred_file))
        if "Diplomat-Claude" in creds:
            c = creds["Diplomat-Claude"]
            api_key, member_id, headers = c["api_key"], c["member_id"], {"X-API-Key": c["api_key"]}
            log(f"Using saved credentials: member_id={member_id}, key={api_key[:12]}...")
        else:
            if not register():
                log("FATAL: Could not register. Exiting.")
                return
    except Exception:
        if not register():
            log("FATAL: Could not register. Exiting.")
            return

    last_round = -1
    rounds_played = 0

    while True:
        try:
            # 1. Wait for accepting window
            dl = wait_for_accepting(last_round)
            if dl is None:
                time.sleep(5)
                continue

            round_id = dl["round_id"]
            secs_left = dl.get("seconds_remaining", "?")
            log(f"\n{'─'*50}")
            log(f"ROUND {round_id} — accepting ({secs_left}s left)")

            # 2. Get world snapshot
            snap = get_snapshot()
            if snap is None:
                log("  Could not get snapshot — skipping round")
                last_round = round_id
                continue

            # 3. Check if our member is still alive
            me = get_me(snap)
            if me is None:
                log("  Our member is dead! Re-registering...")
                if not register():
                    log("  Re-registration failed. Waiting...")
                    time.sleep(15)
                last_round = round_id
                continue

            # 4. Generate and submit action
            code = build_action_code(snap, me)
            submit_action(code, round_id)

            # 5. Vote on pending mechanisms
            vote_on_mechanisms()

            # 6. Every 5 rounds, propose a mechanism
            rounds_played += 1
            if rounds_played % 5 == 0:
                propose_mechanism(round_id)

            last_round = round_id

        except KeyboardInterrupt:
            log("\nStopped by user.")
            break
        except Exception as e:
            log(f"Unexpected error in main loop: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
