"""Lightweight proxy objects for running agent code in a subprocess.

The EngineProxy mimics the IslandExecution interface but records
intended actions as JSON dicts instead of executing them.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class MemberProxy:
    """Read-only stand-in for a Member object."""
    id: int
    vitality: float
    cargo: float
    land_num: int


@dataclass
class LandProxy:
    """Read-only stand-in for the Land object."""
    shape: Tuple[int, int]


class EngineProxy:
    """Stand-in for IslandExecution that records actions as JSON."""

    def __init__(self, state: dict) -> None:
        self.current_members: List[MemberProxy] = [
            MemberProxy(**m) for m in state["members"]
        ]
        land_shape = state["land"]["shape"]
        self.land = LandProxy(shape=tuple(land_shape))
        self.actions: List[Dict[str, Any]] = []

    def attack(self, member: MemberProxy, target: MemberProxy) -> None:
        self.actions.append({"action": "attack", "member_id": member.id, "target_id": target.id})

    def offer(self, member: MemberProxy, target: MemberProxy) -> None:
        self.actions.append({"action": "offer", "member_id": member.id, "target_id": target.id})

    def offer_land(self, member: MemberProxy, target: MemberProxy) -> None:
        self.actions.append({"action": "offer_land", "member_id": member.id, "target_id": target.id})

    def expand(self, member: MemberProxy) -> None:
        self.actions.append({"action": "expand", "member_id": member.id})

    def send_message(self, from_id: int, to_id: int, message: str) -> None:
        self.actions.append({"action": "message", "from_id": from_id, "to_id": to_id, "message": message})
