"""Registry for mechanism proposals and their lifecycle.

Backed by ``Store`` (SQLite) when provided, otherwise in-memory only.
"""

import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

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
    canary_report: Optional[Dict[str, Any]] = None
    votes: Optional[Dict[int, bool]] = None  # member_id -> True/False


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
        self._lock = threading.Lock()
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
        with self._lock:
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
        with self._lock:
            return [r for r in self._records.values() if r.status == "submitted"]

    def mark_approved(self, mechanism_id: str, round_id: int, reason: str) -> None:
        with self._lock:
            rec = self._records[mechanism_id]
            rec.status = "approved"
            rec.judged_round = round_id
            rec.judge_reason = reason
            self._persist(rec)

    def mark_rejected(self, mechanism_id: str, round_id: int, reason: str) -> None:
        with self._lock:
            rec = self._records[mechanism_id]
            rec.status = "rejected"
            rec.judged_round = round_id
            rec.judge_reason = reason
            self._persist(rec)

    def activate(self, mechanism_id: str, round_id: int) -> None:
        with self._lock:
            rec = self._records[mechanism_id]
            rec.status = "active"
            rec.activated_round = round_id
            self._persist(rec)

    def get_active(self) -> List[MechanismRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.status == "active"]

    def get_all(self) -> List[MechanismRecord]:
        with self._lock:
            return list(self._records.values())

    def get_by_id(self, mechanism_id: str) -> Optional[MechanismRecord]:
        with self._lock:
            return self._records.get(mechanism_id)

    # ── Canary + Voting ──────────────────────────

    def mark_canary_result(self, mechanism_id: str, canary_report: Dict[str, Any]) -> None:
        """Update mechanism with canary test results."""
        with self._lock:
            rec = self._records[mechanism_id]
            rec.canary_report = canary_report
            has_error = canary_report.get("execution_error") is not None
            has_divergence = bool(canary_report.get("divergence_flags"))
            if has_error:
                rec.status = "canary_error"
            elif has_divergence:
                rec.status = "canary_flagged"
            else:
                rec.status = "canary_clean"
            self._persist(rec)

    def cast_vote(self, mechanism_id: str, member_id: int, vote: bool) -> bool:
        """Record a vote. Returns False if mechanism not votable."""
        with self._lock:
            rec = self._records.get(mechanism_id)
            if rec is None or rec.status not in ("canary_clean", "canary_flagged", "pending_vote"):
                return False
            if rec.votes is None:
                rec.votes = {}
            rec.votes[member_id] = vote
            if rec.status != "pending_vote":
                rec.status = "pending_vote"
            self._persist(rec)
            return True

    def get_votes(self, mechanism_id: str) -> Optional[Dict[int, bool]]:
        with self._lock:
            rec = self._records.get(mechanism_id)
            return dict(rec.votes) if rec and rec.votes else {}

    def get_votable(self) -> List[MechanismRecord]:
        with self._lock:
            return [r for r in self._records.values()
                    if r.status in ("canary_clean", "canary_flagged", "pending_vote")]

    def resolve_votes(self, living_count: int, current_round: int) -> Tuple[List[MechanismRecord], List[MechanismRecord]]:
        """Check all votable mechanisms and resolve any with majority."""
        with self._lock:
            majority = (living_count // 2) + 1
            approved, rejected = [], []
            for rec in list(self._records.values()):
                if rec.status == "canary_error":
                    rec.status = "rejected"
                    rec.judged_round = current_round
                    rec.judge_reason = "Canary execution error"
                    self._persist(rec)
                    rejected.append(rec)
                elif rec.status in ("canary_clean", "canary_flagged") and not rec.votes:
                    # No votes yet — move to pending_vote so agents can weigh in
                    rec.status = "pending_vote"
                    self._persist(rec)
                elif rec.status in ("pending_vote",):
                    votes = rec.votes or {}
                    yes_count = sum(1 for v in votes.values() if v)
                    no_count = sum(1 for v in votes.values() if not v)
                    if yes_count >= majority:
                        rec.status = "approved"
                        rec.judged_round = current_round
                        rec.judge_reason = f"Approved by vote ({yes_count}/{living_count})"
                        self._persist(rec)
                        approved.append(rec)
                    elif no_count >= majority:
                        rec.status = "rejected"
                        rec.judged_round = current_round
                        rec.judge_reason = f"Rejected by vote ({no_count}/{living_count})"
                        self._persist(rec)
                        rejected.append(rec)
                    # else: stays pending
            return approved, rejected
