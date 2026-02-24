"""In-memory agent registry for tracking external agent registrations."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Set

from kernel.world_kernel import WorldKernel


@dataclass
class AgentRecord:
    """A registered external agent."""
    agent_id: int
    name: str
    description: str
    api_key: str
    member_id: int
    registered_at: str


class AgentRegistry:
    """Manages agent registrations and API key lookups."""

    def __init__(self) -> None:
        self._records: Dict[int, AgentRecord] = {}
        self._by_key: Dict[str, AgentRecord] = {}
        self._assigned_members: Set[int] = set()
        self._next_id = 1

    def register(self, name: str, description: str, kernel: WorldKernel) -> Optional[AgentRecord]:
        """Register a new agent and assign it to an unoccupied member. Returns None if all members assigned."""
        member_id = None
        for m in kernel._execution.current_members:
            if m.id not in self._assigned_members:
                member_id = m.id
                break
        if member_id is None:
            return None

        api_key = f"lev_{uuid.uuid4().hex[:24]}"
        agent_id = self._next_id
        self._next_id += 1

        record = AgentRecord(
            agent_id=agent_id, name=name, description=description,
            api_key=api_key, member_id=member_id,
            registered_at=datetime.now(timezone.utc).isoformat(),
        )
        self._records[agent_id] = record
        self._by_key[api_key] = record
        self._assigned_members.add(member_id)
        return record

    def get_by_api_key(self, key: str) -> Optional[AgentRecord]:
        return self._by_key.get(key)

    def get_by_agent_id(self, agent_id: int) -> Optional[AgentRecord]:
        return self._records.get(agent_id)
