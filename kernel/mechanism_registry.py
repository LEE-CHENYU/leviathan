"""Registry for mechanism proposals and their lifecycle.

Backed by ``Store`` (SQLite) when provided, otherwise in-memory only.
"""

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from kernel.store import Store


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

    If *store* is given, all mutations are persisted to SQLite and
    the registry loads existing records on init.
    """

    def __init__(self, store: Optional["Store"] = None) -> None:
        self._records: Dict[str, MechanismRecord] = {}
        self._agent_round_submissions: Set[Tuple[int, int]] = set()
        self._store = store
        if store:
            self._load_from_store()

    def _load_from_store(self) -> None:
        for row in self._store.get_mechanisms():  # type: ignore[union-attr]
            rec = MechanismRecord(**row)
            self._records[rec.mechanism_id] = rec
        self._agent_round_submissions = self._store.get_all_mechanism_submissions()  # type: ignore[union-attr]

    def _persist(self, rec: MechanismRecord) -> None:
        if not self._store:
            return
        self._store.upsert_mechanism(
            mechanism_id=rec.mechanism_id,
            proposer_id=rec.proposer_id,
            code=rec.code,
            description=rec.description,
            status=rec.status,
            submitted_round=rec.submitted_round,
            judged_round=rec.judged_round,
            judge_reason=rec.judge_reason,
            activated_round=rec.activated_round,
        )

    # ── Mutations ─────────────────────────────────

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
        if self._store:
            self._store.add_mechanism_submission(proposer_id, round_id)
        self._persist(record)
        return record

    def get_pending(self) -> List[MechanismRecord]:
        return [r for r in self._records.values() if r.status == "submitted"]

    def mark_approved(self, mechanism_id: str, round_id: int, reason: str) -> None:
        rec = self._records[mechanism_id]
        rec.status = "approved"
        rec.judged_round = round_id
        rec.judge_reason = reason
        self._persist(rec)

    def mark_rejected(self, mechanism_id: str, round_id: int, reason: str) -> None:
        rec = self._records[mechanism_id]
        rec.status = "rejected"
        rec.judged_round = round_id
        rec.judge_reason = reason
        self._persist(rec)

    def activate(self, mechanism_id: str, round_id: int) -> None:
        rec = self._records[mechanism_id]
        rec.status = "active"
        rec.activated_round = round_id
        self._persist(rec)

    def get_active(self) -> List[MechanismRecord]:
        return [r for r in self._records.values() if r.status == "active"]

    def get_all(self) -> List[MechanismRecord]:
        return list(self._records.values())

    def get_by_id(self, mechanism_id: str) -> Optional[MechanismRecord]:
        return self._records.get(mechanism_id)
