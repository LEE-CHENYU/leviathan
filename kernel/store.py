"""SQLite-backed persistent store for Leviathan world state.

Covers authoritative state per docs/INVARIANTS.md E-1:
  - Event log (append-only round settlement history)
  - Mechanism registry (full proposal lifecycle)
  - Rollback snapshots (recent world snapshots for moderator rollback)

Design principles:
  - WAL mode for concurrent reads during writes (E-3)
  - Atomic transactions for all mutations (E-2)
  - Single writer assumed (the simulation loop)
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id    INTEGER PRIMARY KEY,
    event_type  TEXT    NOT NULL,
    round_id    INTEGER NOT NULL,
    timestamp   TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    world_id    TEXT,
    phase       TEXT,
    payload_hash    TEXT,
    prev_event_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_round ON events(round_id);

CREATE TABLE IF NOT EXISTS mechanisms (
    mechanism_id    TEXT PRIMARY KEY,
    proposer_id     INTEGER NOT NULL,
    code            TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    submitted_round INTEGER NOT NULL,
    judged_round    INTEGER,
    judge_reason    TEXT,
    activated_round INTEGER
);

CREATE TABLE IF NOT EXISTS mechanism_submissions (
    proposer_id INTEGER NOT NULL,
    round_id    INTEGER NOT NULL,
    PRIMARY KEY (proposer_id, round_id)
);

CREATE TABLE IF NOT EXISTS snapshots (
    round_id    INTEGER PRIMARY KEY,
    data        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    idempotency_key TEXT    PRIMARY KEY,
    round_id        INTEGER NOT NULL,
    result_json     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_idempotency_round ON idempotency_keys(round_id);
"""


class Store:
    """SQLite-backed persistent store.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for tests.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("Store opened: %s", db_path)

    def close(self) -> None:
        self._conn.close()

    # ── Events ────────────────────────────────────

    def append_event(
        self,
        event_type: str,
        round_id: int,
        timestamp: str,
        payload: Dict[str, Any],
        world_id: Optional[str] = None,
        phase: Optional[str] = None,
        payload_hash: Optional[str] = None,
        prev_event_hash: Optional[str] = None,
    ) -> int:
        """Append an event and return the assigned event_id."""
        payload_json = json.dumps(payload, default=str)
        cur = self._conn.execute(
            """INSERT INTO events
               (event_type, round_id, timestamp, payload,
                world_id, phase, payload_hash, prev_event_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_type, round_id, timestamp, payload_json,
             world_id, phase, payload_hash, prev_event_hash),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_events(self, since_round: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return events, optionally filtered to round_id > since_round."""
        if since_round is not None:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE round_id > ? ORDER BY event_id",
                (since_round,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY event_id"
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_event_by_round(self, round_id: int, event_type: str = "round_settled") -> Optional[Dict[str, Any]]:
        """Return a single event matching round_id and event_type."""
        row = self._conn.execute(
            "SELECT * FROM events WHERE round_id = ? AND event_type = ? LIMIT 1",
            (round_id, event_type),
        ).fetchone()
        return self._row_to_event(row) if row else None

    def event_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return row[0]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "event_type": row["event_type"],
            "round_id": row["round_id"],
            "timestamp": row["timestamp"],
            "payload": json.loads(row["payload"]),
            "world_id": row["world_id"],
            "phase": row["phase"],
            "payload_hash": row["payload_hash"],
            "prev_event_hash": row["prev_event_hash"],
        }

    # ── Mechanisms ────────────────────────────────

    def upsert_mechanism(
        self,
        mechanism_id: str,
        proposer_id: int,
        code: str,
        description: str,
        status: str,
        submitted_round: int,
        judged_round: Optional[int] = None,
        judge_reason: Optional[str] = None,
        activated_round: Optional[int] = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO mechanisms
               (mechanism_id, proposer_id, code, description, status,
                submitted_round, judged_round, judge_reason, activated_round)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(mechanism_id) DO UPDATE SET
                 status=excluded.status,
                 judged_round=excluded.judged_round,
                 judge_reason=excluded.judge_reason,
                 activated_round=excluded.activated_round""",
            (mechanism_id, proposer_id, code, description, status,
             submitted_round, judged_round, judge_reason, activated_round),
        )
        self._conn.commit()

    def add_mechanism_submission(self, proposer_id: int, round_id: int) -> bool:
        """Record a submission. Returns False if already exists."""
        try:
            self._conn.execute(
                "INSERT INTO mechanism_submissions (proposer_id, round_id) VALUES (?, ?)",
                (proposer_id, round_id),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def has_mechanism_submission(self, proposer_id: int, round_id: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM mechanism_submissions WHERE proposer_id = ? AND round_id = ?",
            (proposer_id, round_id),
        ).fetchone()
        return row is not None

    def get_mechanisms(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM mechanisms WHERE status = ?", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM mechanisms").fetchall()
        return [dict(r) for r in rows]

    def get_mechanism(self, mechanism_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM mechanisms WHERE mechanism_id = ?", (mechanism_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_mechanism_submissions(self) -> Set[Tuple[int, int]]:
        rows = self._conn.execute("SELECT proposer_id, round_id FROM mechanism_submissions").fetchall()
        return {(r["proposer_id"], r["round_id"]) for r in rows}

    # ── Snapshots ─────────────────────────────────

    def store_snapshot(self, round_id: int, data: Dict[str, Any], max_keep: int = 10) -> None:
        """Store a snapshot, evicting oldest if over max_keep."""
        self._conn.execute(
            "INSERT OR REPLACE INTO snapshots (round_id, data) VALUES (?, ?)",
            (round_id, json.dumps(data, default=str)),
        )
        # Evict oldest beyond max_keep
        self._conn.execute(
            """DELETE FROM snapshots WHERE round_id NOT IN
               (SELECT round_id FROM snapshots ORDER BY round_id DESC LIMIT ?)""",
            (max_keep,),
        )
        self._conn.commit()

    def get_snapshot(self, round_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT data FROM snapshots WHERE round_id = ?", (round_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else None

    def snapshot_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()
        return row[0]

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Return metadata for all available snapshots (round_id, summary)."""
        rows = self._conn.execute(
            "SELECT round_id, data FROM snapshots ORDER BY round_id DESC"
        ).fetchall()
        result = []
        for row in rows:
            data = json.loads(row["data"])
            entry = {"round_id": row["round_id"]}
            # Extract summary without loading full state
            if isinstance(data, dict):
                members = data.get("members", [])
                entry["member_count"] = len(members)
                entry["total_vitality"] = sum(
                    m.get("vitality", 0) for m in members if isinstance(m, dict)
                )
            result.append(entry)
        return result

    def get_snapshot_summary(self, round_id: int) -> Optional[Dict[str, Any]]:
        """Return summary (member_count, total_vitality) without full data."""
        row = self._conn.execute(
            "SELECT data FROM snapshots WHERE round_id = ?", (round_id,)
        ).fetchone()
        if not row:
            return None
        data = json.loads(row["data"])
        if not isinstance(data, dict):
            return {"round_id": round_id}
        members = data.get("members", [])
        return {
            "round_id": round_id,
            "member_count": len(members),
            "total_vitality": sum(
                m.get("vitality", 0) for m in members if isinstance(m, dict)
            ),
        }

    # ── Idempotency Keys ─────────────────────────

    def store_idempotency_key(
        self, key: str, round_id: int, result_json: str
    ) -> None:
        """Store an idempotency key with its result."""
        self._conn.execute(
            "INSERT OR REPLACE INTO idempotency_keys (idempotency_key, round_id, result_json) VALUES (?, ?, ?)",
            (key, round_id, result_json),
        )
        self._conn.commit()

    def get_idempotency_result(self, key: str) -> Optional[str]:
        """Return the stored result JSON for a key, or None."""
        row = self._conn.execute(
            "SELECT result_json FROM idempotency_keys WHERE idempotency_key = ?",
            (key,),
        ).fetchone()
        return row["result_json"] if row else None

    def get_idempotency_keys_for_round(self, round_id: int) -> Dict[str, str]:
        """Return all idempotency keys and results for a given round."""
        rows = self._conn.execute(
            "SELECT idempotency_key, result_json FROM idempotency_keys WHERE round_id = ?",
            (round_id,),
        ).fetchall()
        return {r["idempotency_key"]: r["result_json"] for r in rows}

    def clear_idempotency_keys_before_round(self, round_id: int) -> int:
        """Delete idempotency keys from rounds before the given round. Returns count deleted."""
        cur = self._conn.execute(
            "DELETE FROM idempotency_keys WHERE round_id < ?", (round_id,)
        )
        self._conn.commit()
        return cur.rowcount
