"""Persistent event log backed by ``Store`` (SQLite).

Drop-in replacement for the plain ``List[EventEnvelope]`` used previously.
Supports ``append()``, ``len()``, iteration, and ``since_round`` filtering.
"""

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from api.models import EventEnvelope

if TYPE_CHECKING:
    from kernel.store import Store


class EventLog:
    """Append-only event log backed by SQLite.

    Provides a list-like interface so existing code
    (``len(event_log)``, ``for e in event_log``, ``event_log.append(...)``)
    continues to work without changes.
    """

    def __init__(self, store: Optional["Store"] = None) -> None:
        self._store = store
        # In-memory fallback when no store is provided (tests, ephemeral)
        self._mem: List[EventEnvelope] = []
        if store:
            # Load existing events into memory cache for fast iteration
            for row in store.get_events():
                self._mem.append(EventEnvelope(**row))

    def append(self, envelope: EventEnvelope) -> None:
        """Append an event. Persists to SQLite if a store is available."""
        if self._store:
            event_id = self._store.append_event(
                event_type=envelope.event_type,
                round_id=envelope.round_id,
                timestamp=envelope.timestamp,
                payload=envelope.payload,
                world_id=envelope.world_id,
                phase=envelope.phase,
                payload_hash=envelope.payload_hash,
                prev_event_hash=envelope.prev_event_hash,
            )
            envelope.event_id = event_id
        self._mem.append(envelope)

    def since_round(self, round_id: int) -> List[EventEnvelope]:
        return [e for e in self._mem if e.round_id > round_id]

    def find_by_round(self, round_id: int, event_type: str = "round_settled") -> Optional[EventEnvelope]:
        for e in self._mem:
            if e.event_type == event_type and e.round_id == round_id:
                return e
        return None

    def __len__(self) -> int:
        return len(self._mem)

    def __iter__(self) -> Iterator[EventEnvelope]:
        return iter(self._mem)

    def __getitem__(self, index):
        return self._mem[index]
