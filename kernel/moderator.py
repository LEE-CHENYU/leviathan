"""Moderator configuration and state management."""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from kernel.schemas import WorldSnapshot


@dataclass
class QuotaConfig:
    max_actions_per_round: int = 0      # 0 = unlimited
    max_proposals_per_round: int = 1


class ModeratorState:
    """Tracks moderator-controlled world state."""

    def __init__(self, max_snapshots: int = 10) -> None:
        self.paused: bool = False
        self.banned_agents: Set[int] = set()
        self.quotas: QuotaConfig = QuotaConfig()
        self._snapshot_history: List[WorldSnapshot] = []
        self._max_snapshots = max_snapshots

    def is_banned(self, agent_id: int) -> bool:
        return agent_id in self.banned_agents

    def ban(self, agent_id: int) -> None:
        self.banned_agents.add(agent_id)

    def unban(self, agent_id: int) -> None:
        self.banned_agents.discard(agent_id)

    def store_snapshot(self, snapshot: WorldSnapshot) -> None:
        self._snapshot_history.append(snapshot)
        if len(self._snapshot_history) > self._max_snapshots:
            self._snapshot_history.pop(0)

    def get_snapshot_for_round(self, round_id: int) -> Optional[WorldSnapshot]:
        for s in self._snapshot_history:
            if s.round_id == round_id:
                return s
        return None

    def to_dict(self) -> dict:
        return {
            "paused": self.paused,
            "banned_agents": sorted(self.banned_agents),
            "quotas": {
                "max_actions_per_round": self.quotas.max_actions_per_round,
                "max_proposals_per_round": self.quotas.max_proposals_per_round,
            },
            "snapshot_history_size": len(self._snapshot_history),
        }

    @staticmethod
    def hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]
