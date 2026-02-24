"""Canonical data-transfer objects for the WorldKernel simulation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class WorldConfig:
    """Top-level configuration for world initialization."""

    init_member_number: int
    land_shape: Tuple[int, int]
    random_seed: Optional[int] = None


@dataclass
class ActionIntent:
    """An agent's intended action for a round."""

    agent_id: int
    round_id: int
    code: str
    idempotency_key: str


@dataclass
class ActionResult:
    """Outcome of executing a single agent action."""

    agent_id: int
    success: bool
    old_stats: Dict[str, float]
    new_stats: Dict[str, float]
    performance_change: float
    messages_sent: List[Tuple[int, str]]
    error: Optional[str] = None
    signature: Optional[Dict] = None


@dataclass
class MechanismProposal:
    """A proposed mechanism modification from an agent."""

    proposal_id: str
    agent_id: int
    code: str
    round_id: int


@dataclass
class MechanismResult:
    """Outcome of evaluating / executing a mechanism proposal."""

    proposal_id: str
    executed: bool
    error: Optional[str] = None


@dataclass
class WorldSnapshot:
    """Full serializable snapshot of world state at a given round."""

    world_id: str
    round_id: int
    members: List[Dict]
    land: Dict
    active_mechanisms: List[Dict]
    active_contracts: List[Dict]
    physics_constraints: List[Dict]
    state_hash: str


@dataclass
class RoundReceipt:
    """Immutable record of everything that happened in one simulation round."""

    round_id: int
    seed: int
    snapshot_hash_before: str
    snapshot_hash_after: str
    accepted_action_ids: List[str]
    rejected_action_ids: List[str]
    activated_mechanism_ids: List[str]
    judge_results: List[Dict]
    round_metrics: Dict[str, float]
    timestamp: str
