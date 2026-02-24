"""In-memory registry for mechanism proposals and their lifecycle."""

import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class MechanismRecord:
    """A mechanism proposal with its lifecycle state."""
    mechanism_id: str
    proposer_id: int
    code: str
    description: str
    status: str  # "submitted" | "approved" | "rejected" | "active"
    submitted_round: int
    judged_round: Optional[int] = None
    judge_reason: Optional[str] = None
    activated_round: Optional[int] = None


class MechanismRegistry:
    """Tracks mechanism proposals through their lifecycle.
    Lifecycle: submitted -> approved/rejected -> active
    """

    def __init__(self) -> None:
        self._records: Dict[str, MechanismRecord] = {}
        self._agent_round_submissions: Set[Tuple[int, int]] = set()

    def submit(self, proposer_id: int, code: str, description: str, round_id: int = 0) -> Optional[MechanismRecord]:
        """Submit a new mechanism proposal. Returns None if agent already proposed this round."""
        key = (proposer_id, round_id)
        if key in self._agent_round_submissions:
            return None
        mechanism_id = uuid.uuid4().hex
        record = MechanismRecord(
            mechanism_id=mechanism_id, proposer_id=proposer_id, code=code,
            description=description, status="submitted", submitted_round=round_id,
        )
        self._records[mechanism_id] = record
        self._agent_round_submissions.add(key)
        return record

    def get_pending(self) -> List[MechanismRecord]:
        return [r for r in self._records.values() if r.status == "submitted"]

    def mark_approved(self, mechanism_id: str, round_id: int, reason: str) -> None:
        rec = self._records[mechanism_id]
        rec.status = "approved"
        rec.judged_round = round_id
        rec.judge_reason = reason

    def mark_rejected(self, mechanism_id: str, round_id: int, reason: str) -> None:
        rec = self._records[mechanism_id]
        rec.status = "rejected"
        rec.judged_round = round_id
        rec.judge_reason = reason

    def activate(self, mechanism_id: str, round_id: int) -> None:
        rec = self._records[mechanism_id]
        rec.status = "active"
        rec.activated_round = round_id

    def get_active(self) -> List[MechanismRecord]:
        return [r for r in self._records.values() if r.status == "active"]

    def get_all(self) -> List[MechanismRecord]:
        return list(self._records.values())

    def get_by_id(self, mechanism_id: str) -> Optional[MechanismRecord]:
        return self._records.get(mechanism_id)
