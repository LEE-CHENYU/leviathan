"""Moderator configuration and state management."""

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from kernel.schemas import WorldSnapshot

if TYPE_CHECKING:
    from kernel.store import Store


@dataclass
class QuotaConfig:
    max_actions_per_round: int = 0      # 0 = unlimited
    max_proposals_per_round: int = 1


class ModeratorState:
    """Tracks moderator-controlled world state.

    If *store* is provided, rollback snapshots are persisted to SQLite.
    """

    def __init__(self, max_snapshots: int = 10, store: Optional["Store"] = None) -> None:
        self.paused: bool = False
        self.banned_agents: Set[int] = set()
        self.quotas: QuotaConfig = QuotaConfig()
        self._max_snapshots = max_snapshots
        self._store = store
        # In-memory fallback when no store is provided (tests, ephemeral)
        self._snapshot_history: List[WorldSnapshot] = []

    def is_banned(self, agent_id: int) -> bool:
        return agent_id in self.banned_agents

    def ban(self, agent_id: int) -> None:
        self.banned_agents.add(agent_id)

    def unban(self, agent_id: int) -> None:
        self.banned_agents.discard(agent_id)

    def store_snapshot(self, snapshot: WorldSnapshot) -> None:
        if self._store:
            self._store.store_snapshot(
                round_id=snapshot.round_id,
                data=dataclasses.asdict(snapshot),
                max_keep=self._max_snapshots,
            )
        else:
            self._snapshot_history.append(snapshot)
            if len(self._snapshot_history) > self._max_snapshots:
                self._snapshot_history.pop(0)

    def get_snapshot_for_round(self, round_id: int) -> Optional[WorldSnapshot]:
        if self._store:
            data = self._store.get_snapshot(round_id)
            if data:
                return WorldSnapshot(**data)
            return None
        for s in self._snapshot_history:
            if s.round_id == round_id:
                return s
        return None

    def snapshot_count(self) -> int:
        if self._store:
            return self._store.snapshot_count()
        return len(self._snapshot_history)

    def to_dict(self) -> dict:
        return {
            "paused": self.paused,
            "banned_agents": sorted(self.banned_agents),
            "quotas": {
                "max_actions_per_round": self.quotas.max_actions_per_round,
                "max_proposals_per_round": self.quotas.max_proposals_per_round,
            },
            "snapshot_history_size": self.snapshot_count(),
        }

    @staticmethod
    def hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]
