"""Registry for mechanism proposals and their lifecycle, with optional JSON persistence."""

import dataclasses
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


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

    If *persist_path* is given, state is saved to that JSON file on
    every mutation and loaded on init when the file already exists.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._records: Dict[str, MechanismRecord] = {}
        self._agent_round_submissions: Set[Tuple[int, int]] = set()
        self._persist_path: Optional[Path] = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    # ── Persistence helpers ─────────────────────────

    def _save(self) -> None:
        if not self._persist_path:
            return
        data = {
            "records": [dataclasses.asdict(r) for r in self._records.values()],
            "submissions": list(self._agent_round_submissions),
        }
        tmp = self._persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._persist_path)

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text())  # type: ignore[union-attr]
            for rd in data.get("records", []):
                rec = MechanismRecord(**rd)
                self._records[rec.mechanism_id] = rec
            for pair in data.get("submissions", []):
                self._agent_round_submissions.add(tuple(pair))
            logger.info("Loaded %d mechanisms from %s", len(self._records), self._persist_path)
        except Exception as exc:
            logger.warning("Failed to load mechanism registry from %s: %s", self._persist_path, exc)

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
        self._save()
        return record

    def get_pending(self) -> List[MechanismRecord]:
        return [r for r in self._records.values() if r.status == "submitted"]

    def mark_approved(self, mechanism_id: str, round_id: int, reason: str) -> None:
        rec = self._records[mechanism_id]
        rec.status = "approved"
        rec.judged_round = round_id
        rec.judge_reason = reason
        self._save()

    def mark_rejected(self, mechanism_id: str, round_id: int, reason: str) -> None:
        rec = self._records[mechanism_id]
        rec.status = "rejected"
        rec.judged_round = round_id
        rec.judge_reason = reason
        self._save()

    def activate(self, mechanism_id: str, round_id: int) -> None:
        rec = self._records[mechanism_id]
        rec.status = "active"
        rec.activated_round = round_id
        self._save()

    def get_active(self) -> List[MechanismRecord]:
        return [r for r in self._records.values() if r.status == "active"]

    def get_all(self) -> List[MechanismRecord]:
        return list(self._records.values())

    def get_by_id(self, mechanism_id: str) -> Optional[MechanismRecord]:
        return self._records.get(mechanism_id)
