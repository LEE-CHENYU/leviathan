#!/usr/bin/env python3
"""Compact world state summarizer for agent prompts.

Usage: python3 summarize_state.py <member_id>
Reads snapshot + metrics from stdin (JSON lines) or from the server.
Outputs a concise markdown summary suitable for LLM context.
"""
import sys
import json
import os

BASE = os.environ.get("LEVIATHAN_BASE", "https://leviathan.fly.dev")


def summarize(snapshot: dict, metrics: dict, member_id: int, mechanisms: list) -> str:
    members = snapshot.get("members", [])
    me = None
    others = []
    for m in members:
        if m.get("id") == member_id:
            me = m
        else:
            others.append(m)

    lines = []

    # My status
    if me:
        lines.append(f"### You (member {member_id})")
        lines.append(f"- vitality: {me['vitality']:.1f}/100")
        lines.append(f"- cargo: {me['cargo']:.1f}")
        lines.append(f"- land: {me['land_num']}")
        # Rank
        sorted_by_vit = sorted(members, key=lambda m: m.get("vitality", 0), reverse=True)
        rank = next((i + 1 for i, m in enumerate(sorted_by_vit) if m["id"] == member_id), "?")
        lines.append(f"- rank by vitality: {rank}/{len(members)}")
    else:
        lines.append(f"### You (member {member_id}): DEAD")

    # World overview
    lines.append(f"\n### World Overview")
    lines.append(f"- round: {metrics.get('round_id', '?')}")
    lines.append(f"- population: {metrics.get('population', len(members))}")
    lines.append(f"- total vitality: {metrics.get('total_vitality', 0):.0f}")
    lines.append(f"- gini coefficient: {metrics.get('gini_coefficient', 0):.3f}")
    lines.append(f"- trade volume: {metrics.get('trade_volume', 0)}")
    lines.append(f"- conflict count: {metrics.get('conflict_count', 0)}")
    lines.append(f"- mechanism proposals: {metrics.get('mechanism_proposals', 0)}")
    lines.append(f"- mechanism approvals: {metrics.get('mechanism_approvals', 0)}")

    # Top 5 members
    top = sorted(members, key=lambda m: m.get("vitality", 0) + m.get("cargo", 0), reverse=True)[:5]
    lines.append(f"\n### Top 5 Members (by vitality+cargo)")
    for m in top:
        marker = " ← YOU" if m["id"] == member_id else ""
        lines.append(f"- id={m['id']}: vit={m['vitality']:.1f} cargo={m['cargo']:.1f} land={m['land_num']}{marker}")

    # Weakest 5
    weak = sorted(members, key=lambda m: m.get("vitality", 0))[:5]
    lines.append(f"\n### Weakest 5 Members")
    for m in weak:
        marker = " ← YOU" if m["id"] == member_id else ""
        lines.append(f"- id={m['id']}: vit={m['vitality']:.1f} cargo={m['cargo']:.1f} land={m['land_num']}{marker}")

    # Pending mechanisms
    if mechanisms:
        lines.append(f"\n### Pending Mechanisms ({len(mechanisms)} needing your vote)")
        for mech in mechanisms:
            mid = mech.get("mechanism_id") or mech.get("id", "?")
            desc = mech.get("description", "no description")[:120]
            canary = mech.get("canary_report") or mech.get("canary_result") or {}
            vit_change = canary.get("vitality_change_pct", "n/a")
            deaths = canary.get("agents_died", [])
            err = canary.get("execution_error")
            lines.append(f"- **{mid}**: {desc}")
            if err:
                lines.append(f"  - canary: EXECUTION ERROR — {err[:80]}")
            elif deaths:
                lines.append(f"  - canary: {len(deaths)} agents died, vitality change {vit_change}")
            else:
                lines.append(f"  - canary: vitality change {vit_change}, no deaths")
    else:
        lines.append(f"\n### Pending Mechanisms: none")

    return "\n".join(lines)


if __name__ == "__main__":
    import requests

    member_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    snap = requests.get(f"{BASE}/v1/world/snapshot", timeout=10).json()
    metrics = requests.get(f"{BASE}/v1/world/metrics", timeout=10).json()
    try:
        mechs = requests.get(f"{BASE}/v1/world/mechanisms", params={"status": "pending_vote"}, timeout=10).json()
        if isinstance(mechs, dict):
            mechs = mechs.get("mechanisms", [])
    except Exception:
        mechs = []

    print(summarize(snap, metrics, member_id, mechs))
