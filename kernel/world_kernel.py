"""WorldKernel -- facade wrapping IslandExecution for the distributed kernel API."""

import dataclasses
import uuid
from typing import Dict, List, Optional

from kernel.schemas import (
    ActionIntent,
    ActionResult,
    MechanismProposal,
    MechanismResult,
    RoundReceipt,
    WorldConfig,
    WorldSnapshot,
)
from kernel.receipt import compute_state_hash
from kernel.execution_sandbox import InProcessSandbox, SandboxContext


class WorldKernel:
    """High-level facade over the simulation engine.

    Provides a clean, deterministic API for:
    - Inspecting world state (snapshots)
    - Submitting agent actions and mechanism proposals
    - Settling rounds with produce/consume
    - Generating immutable round receipts
    """

    def __init__(self, config: WorldConfig, save_path: str) -> None:
        # Lazy import to avoid circular imports at module level
        from MetaIsland.metaIsland import IslandExecution

        self._execution = IslandExecution(
            init_member_number=config.init_member_number,
            land_shape=config.land_shape,
            save_path=save_path,
            random_seed=config.random_seed,
        )
        self._world_id = str(uuid.uuid4())
        self._round_id = 0
        self._sandbox = InProcessSandbox()
        self._idempotency_cache: Dict[str, ActionResult] = {}
        self._last_receipt: Optional[RoundReceipt] = None

    # ── Properties ────────────────────────────────

    @property
    def round_id(self) -> int:
        return self._round_id

    # ── Snapshot ──────────────────────────────────

    def get_snapshot(self) -> WorldSnapshot:
        """Build a full WorldSnapshot from the current engine state."""
        members = []
        for m in self._execution.current_members:
            members.append({
                "id": m.id,
                "vitality": float(m.vitality),
                "cargo": float(m.cargo),
                "land_num": int(m.land_num),
            })

        land_info = {"shape": list(self._execution.land.shape)}

        # Active mechanisms from execution history
        active_mechanisms: List[Dict] = []
        if self._execution.execution_history.get("rounds"):
            last_round = self._execution.execution_history["rounds"][-1]
            mods = last_round.get("mechanism_modifications", {})
            for mod in mods.get("executed", []):
                active_mechanisms.append({
                    "member_id": mod.get("member_id"),
                    "round": mod.get("round") or mod.get("executed_round"),
                })

        # Active contracts
        active_contracts: List[Dict] = []
        if hasattr(self._execution, "contracts"):
            for cid, cdata in self._execution.contracts.contracts.items():
                active_contracts.append({"id": cid, **cdata})

        # Physics constraints
        physics_constraints: List[Dict] = []
        if hasattr(self._execution, "physics"):
            for constraint in self._execution.physics.constraints:
                physics_constraints.append(dict(constraint))

        snap_dict = {
            "world_id": self._world_id,
            "round_id": self._round_id,
            "members": members,
            "land": land_info,
            "active_mechanisms": active_mechanisms,
            "active_contracts": active_contracts,
            "physics_constraints": physics_constraints,
            "state_hash": "",
        }
        snap_dict["state_hash"] = compute_state_hash(snap_dict)

        return WorldSnapshot(**snap_dict)
