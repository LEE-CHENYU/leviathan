"""Thread-safe round state shared between the API and simulation threads."""

import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional


@dataclass
class PendingAction:
    """An action submitted by an external agent, waiting for execution."""
    agent_id: int
    member_id: int
    code: str
    idempotency_key: str


class RoundState:
    """Manages the submission window between begin_round and settle_round.
    Thread-safe: all public methods acquire self._lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state: str = "settled"
        self.round_id: int = 0
        self.deadline: Optional[datetime] = None
        self._pending: List[PendingAction] = []
        self._seen_keys: Dict[str, bool] = {}

    def open_submissions(self, round_id: int, pace: float) -> None:
        with self._lock:
            self.state = "accepting"
            self.round_id = round_id
            self.deadline = datetime.now(timezone.utc) + timedelta(seconds=pace)
            self._pending = []
            self._seen_keys = {}

    def submit_action(self, action: PendingAction) -> bool:
        with self._lock:
            if self.state != "accepting":
                return False
            if action.idempotency_key in self._seen_keys:
                return True  # Idempotent accept
            self._seen_keys[action.idempotency_key] = True
            self._pending.append(action)
            return True

    def close_submissions(self) -> None:
        with self._lock:
            self.state = "executing"

    def drain_actions(self) -> List[PendingAction]:
        with self._lock:
            actions = list(self._pending)
            self._pending = []
            return actions

    def mark_settled(self) -> None:
        with self._lock:
            self.state = "settled"

    def get_pending_actions(self) -> List[PendingAction]:
        with self._lock:
            return list(self._pending)

    def seconds_remaining(self) -> float:
        with self._lock:
            if self.deadline is None:
                return 0.0
            remaining = (self.deadline - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, remaining)
