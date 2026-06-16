"""Deterministic hashing utilities for world snapshots and round receipts."""

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    """Return a deterministic, compact JSON representation as UTF-8 bytes.

    * Keys are sorted.
    * No extra whitespace.
    * Non-serializable values fall back to ``str()``.
    * Non-ASCII characters are preserved (``ensure_ascii=False``).
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def compute_state_hash(snapshot: dict) -> str:
    """Compute the SHA-256 hex digest of a world snapshot dict.

    The ``state_hash`` field is excluded from the hash computation so
    that the hash can be stored inside the snapshot itself without
    creating a circular dependency.
    """
    filtered = {k: v for k, v in snapshot.items() if k != "state_hash"}
    return hashlib.sha256(canonical_json(filtered)).hexdigest()


def compute_receipt_hash(receipt: dict) -> str:
    """Compute the SHA-256 hex digest of a round receipt dict."""
    return hashlib.sha256(canonical_json(receipt)).hexdigest()
