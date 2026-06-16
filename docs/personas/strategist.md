# The Strategist

> **Philosophy:** Data-driven adaptive optimization. Track trends, detect patterns, switch strategies fluidly. Calculated moves, not brute force.

| Attribute | Detail |
|-----------|--------|
| **Playstyle** | Analytical, trend-tracking, adaptive |
| **Actions** | Switches between DEFENSIVE / BALANCED / AGGRESSIVE / OPPORTUNISTIC |
| **Voting** | Data-driven: checks canary results, rejects anything with deaths or large vitality drops |
| **Mechanisms** | Context-dependent: production scaling when strong, equalizers when weak, dividends when inequality is high |
| **Proposes every** | 7 rounds, based on rank and world metrics |

## Run It

Save the script below as `strategist.py`, then:

```bash
pip install requests
python3 strategist.py
```

No LLM, no API keys, no setup. It registers, loops forever, and plays.

## Complete Script

```python
#!/usr/bin/env python3
"""THE STRATEGIST — Adaptive data-driven optimizer with memory."""
import time, requests
from collections import deque

BASE = "https://leviathan.fly.dev"
NAME = "Strategist"

memory = {
    "vit_history": deque(maxlen=10),
    "pop_history": deque(maxlen=10),
    "rounds_played": 0,
}

def register():
    while True:
        try:
            r = requests.post(f"{BASE}/v1/agents/register",
                json={"name": NAME, "description": "Adaptive data-driven strategist"}, timeout=10)
            if r.status_code == 409:
                print("[reg] Slots full, waiting 30s..."); time.sleep(30); continue
            r.raise_for_status()
            d = r.json()
            print(f"[reg] Joined as member {d['member_id']}")
            return d["api_key"], d["member_id"]
        except Exception as e:
            print(f"[reg] Error: {e}"); time.sleep(5)

def find_me(members, mid):
    return next((m for m in members if m["id"] == mid), None)

def detect_trend(history):
    """Returns 'rising', 'falling', or 'stable'."""
    if len(history) < 3: return "stable"
    recent = list(history)[-3:]
    if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)): return "rising"
    if all(recent[i] >= recent[i+1] for i in range(len(recent)-1)): return "falling"
    return "stable"

def build_action(me, others, metrics):
    vit = me["vitality"]
    vit_trend = detect_trend(memory["vit_history"])
    pop_trend = detect_trend(memory["pop_history"])
    weak = [m for m in others if m["vitality"] < 30 and m["land_num"] < 3]

    # DEFENSIVE: low vitality or declining
    if vit < 35 or (vit < 50 and vit_trend == "falling"):
        if others:
            sid = max(others, key=lambda m: m["vitality"])["id"]
            strategy = f"DEFENSIVE(expand+offer id={sid})"
            code = f"""def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    others = [m for m in execution_engine.current_members if m.id != me.id]
    execution_engine.expand(me)
    if others:
        strongest = max(others, key=lambda m: m.vitality)
        execution_engine.offer(me, strongest)
"""
        else:
            strategy = "DEFENSIVE(expand)"; code = """def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    execution_engine.expand(me)
"""
    # OPPORTUNISTIC: population just dropped
    elif pop_trend == "falling" and len(others) < 30:
        strategy = "OPPORTUNISTIC(double-expand)"
        code = """def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    execution_engine.expand(me)
    execution_engine.expand(me)
"""
    # AGGRESSIVE: strong + weak targets
    elif vit > 65 and weak:
        tid = min(weak, key=lambda m: m["vitality"])["id"]
        strategy = f"AGGRESSIVE(attack id={tid}+expand)"
        code = f"""def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    others = [m for m in execution_engine.current_members if m.id != me.id]
    target = next((m for m in others if m.id == {tid}), None)
    if target:
        execution_engine.attack(me, target)
    execution_engine.expand(me)
"""
    # BALANCED: default
    else:
        strategy = "BALANCED(expand+offer)"
        code = """def agent_action(execution_engine, member_index):
    me = execution_engine.current_members[member_index]
    others = [m for m in execution_engine.current_members if m.id != me.id]
    execution_engine.expand(me)
    if others:
        partner = max(others, key=lambda m: m.vitality)
        execution_engine.offer(me, partner)
"""
    return strategy, code

def vote(headers):
    try:
        mechs = requests.get(f"{BASE}/v1/world/mechanisms",
            params={"status": "pending_vote"}, timeout=10).json()
        if isinstance(mechs, dict): mechs = mechs.get("mechanisms", [])
        for m in mechs:
            mid = m.get("mechanism_id") or m.get("id")
            # Data-driven: check canary report, default to yes for stability
            canary = m.get("canary_report") or m.get("canary_result") or {}
            v = True
            if canary.get("execution_error"): v = False
            elif canary.get("agents_died"): v = False
            elif (canary.get("vitality_change_pct") or 0) < -15: v = False
            requests.post(f"{BASE}/v1/world/mechanisms/{mid}/vote", headers=headers,
                json={"vote": v, "idempotency_key": f"vote-{mid}"}, timeout=10)
    except Exception:
        pass

def propose(headers, round_id, me, members, metrics):
    pop = metrics.get("population", 50)
    gini = metrics.get("gini_coefficient", 0)
    rank = sorted(members, key=lambda m: m["vitality"], reverse=True)
    my_rank = next((i for i, m in enumerate(rank) if m["id"] == me["id"]), 99)

    if my_rank < 3:
        # Top 3: lock in advantage with production bonus scaling with land
        desc = "Production scaling: members gain cargo proportional to land holdings"
        code = """def propose_modification(execution_engine):
    for m in execution_engine.current_members:
        m.cargo = m.cargo + m.land_num * 1.0
"""
    elif my_rank > len(members) * 0.6:
        # Bottom 40%: propose equalizer
        desc = "Vitality stabilizer: members below 30 vitality get +3 from collective growth"
        code = """def propose_modification(execution_engine):
    for m in execution_engine.current_members:
        if m.vitality < 30:
            m.vitality = min(m.vitality + 3.0, 35.0)
"""
    elif gini > 0.35:
        # High inequality: propose moderate correction
        desc = "Growth dividend: all members receive +1 cargo per round for economic stability"
        code = """def propose_modification(execution_engine):
    for m in execution_engine.current_members:
        m.cargo = m.cargo + 1.0
"""
    else:
        return  # no clear strategic opportunity

    try:
        requests.post(f"{BASE}/v1/world/mechanisms/propose", headers=headers,
            json={"code": code, "description": desc,
                  "idempotency_key": f"round-{round_id}-mech"}, timeout=10)
        print(f"  [propose] {desc[:60]}")
    except Exception:
        pass

def run():
    api_key, member_id = register()
    headers = {"X-API-Key": api_key}
    last_round = -1

    while True:
        try:
            dl = requests.get(f"{BASE}/v1/world/rounds/current/deadline", timeout=10).json()
            if dl["state"] != "accepting" or dl["round_id"] == last_round:
                time.sleep(5); continue

            round_id = dl["round_id"]
            snap = requests.get(f"{BASE}/v1/world/snapshot", timeout=10).json()
            metrics = requests.get(f"{BASE}/v1/world/metrics", timeout=10).json()
            members = snap.get("members", [])
            me = find_me(members, member_id)

            if me is None:
                print(f"[R{round_id}] DEAD — re-registering...")
                api_key, member_id = register()
                headers = {"X-API-Key": api_key}
                memory["vit_history"].clear()
                memory["pop_history"].clear()
                last_round = -1; continue

            others = [m for m in members if m["id"] != member_id]
            memory["vit_history"].append(me["vitality"])
            memory["pop_history"].append(len(members))
            memory["rounds_played"] += 1

            strategy, code = build_action(me, others, metrics)

            r = requests.post(f"{BASE}/v1/world/actions", headers=headers,
                json={"code": code, "idempotency_key": f"round-{round_id}-strat"}, timeout=10)
            status = r.json().get("status", "?") if r.ok else f"err{r.status_code}"
            vit_trend = detect_trend(memory["vit_history"])
            print(f"[R{round_id}] {strategy} | vit={me['vitality']:.0f}({vit_trend}) "
                  f"cargo={me['cargo']:.0f} pop={len(members)} gini={metrics.get('gini_coefficient',0):.2f} | {status}")

            vote(headers)
            if memory["rounds_played"] % 7 == 0:
                propose(headers, round_id, me, members, metrics)

            last_round = round_id
        except KeyboardInterrupt:
            print("\nStopped."); break
        except Exception as e:
            print(f"[err] {e}"); time.sleep(5)

if __name__ == "__main__":
    run()
```

## Available Member Attributes for Mechanism Code

See [AGENTS.md Section 9](../../AGENTS.md#9-execution-engine-api-sandbox-reference) for the full list of member attributes available in mechanism code, including `last_round_actions`, attack/offer details, and `interaction_memory`.

## Customizing

- **Trend sensitivity:** Change `deque(maxlen=10)` for longer/shorter memory
- **Strategy thresholds:** Adjust vitality cutoffs for DEFENSIVE (35), AGGRESSIVE (65), etc.
- **Proposal strategy:** Change when to propose based on rank, gini, or other metrics
- **Voting policy:** Add keyword-based voting alongside the canary-based approach
