"""Canary testing infrastructure for mechanism proposals.

Runs mechanism code against a deep copy of world state before live execution.
Exposes results (vitality changes, agent deaths, divergence flags) to agents.
Supports I-8 (Canary Before Commit) from docs/INVARIANTS.md.
"""

import copy
import random
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CanaryReport:
    """Results of a canary run for a single mechanism proposal."""
    proposal_id: str
    proposer_id: int
    snapshot_before: Dict[str, Any]
    snapshot_after: Dict[str, Any]
    vitality_change_pct: float
    agents_died: List[int]
    divergence_flags: List[str]
    judge_opinion: Optional[Tuple[str, str]] = None  # (concern_level, reason)
    execution_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposer_id": self.proposer_id,
            "snapshot_before": self.snapshot_before,
            "snapshot_after": self.snapshot_after,
            "vitality_change_pct": self.vitality_change_pct,
            "agents_died": self.agents_died,
            "divergence_flags": self.divergence_flags,
            "judge_opinion": self.judge_opinion,
            "execution_error": self.execution_error,
        }


def _compact_member_stats(members: list) -> Dict[str, Any]:
    """Build compact stats from a list of member objects."""
    stats = {}
    total_vitality = 0.0
    alive_ids = []
    for m in members:
        v = float(getattr(m, "vitality", 0))
        total_vitality += v
        mid = getattr(m, "id", None)
        if mid is not None:
            alive_ids.append(mid)
        stats[str(mid)] = {
            "vitality": v,
            "cargo": float(getattr(m, "cargo", 0)),
            "land_num": int(getattr(m, "land_num", 0)),
        }
    stats["_total_vitality"] = total_vitality
    stats["_alive_ids"] = alive_ids
    return stats


def _deep_copy_state(execution_engine) -> Dict[str, Any]:
    """Deep-copy mutable state-bearing attributes from the execution engine.

    Does NOT copy: LLM client, graph engine, file handles, execution_history.
    Forks the seeded PRNG for I-1 compliance.
    """
    state = {}
    # Members — deep copy the list and each member
    state["current_members"] = copy.deepcopy(execution_engine.current_members)

    # Land
    if hasattr(execution_engine, "land"):
        state["land"] = copy.deepcopy(execution_engine.land)

    # Relationship dict
    if hasattr(execution_engine, "relationship_dict"):
        state["relationship_dict"] = copy.deepcopy(execution_engine.relationship_dict)

    # Contracts
    if hasattr(execution_engine, "contracts"):
        state["contracts_pending"] = copy.deepcopy(
            getattr(execution_engine.contracts, "pending", {})
        )
        state["contracts_active"] = copy.deepcopy(
            getattr(execution_engine.contracts, "active", {})
        )

    # Physics constraints
    if hasattr(execution_engine, "physics"):
        state["physics_constraints"] = copy.deepcopy(
            getattr(execution_engine.physics, "constraints", [])
        )

    # Resources
    if hasattr(execution_engine, "resources"):
        state["resources"] = copy.deepcopy(execution_engine.resources)

    # Fork PRNG — preserve I-1 determinism
    if hasattr(execution_engine, "_rng"):
        state["_rng"] = copy.deepcopy(execution_engine._rng)

    return state


def _apply_state(execution_engine, state: Dict[str, Any]) -> None:
    """Apply deep-copied state onto an execution engine (for canary run)."""
    execution_engine.current_members = state["current_members"]
    if "land" in state:
        execution_engine.land = state["land"]
    if "relationship_dict" in state:
        execution_engine.relationship_dict = state["relationship_dict"]
    if "contracts_pending" in state and hasattr(execution_engine, "contracts"):
        execution_engine.contracts.pending = state["contracts_pending"]
    if "contracts_active" in state and hasattr(execution_engine, "contracts"):
        execution_engine.contracts.active = state["contracts_active"]
    if "physics_constraints" in state and hasattr(execution_engine, "physics"):
        execution_engine.physics.constraints = state["physics_constraints"]
    if "resources" in state:
        execution_engine.resources = state["resources"]
    if "_rng" in state and hasattr(execution_engine, "_rng"):
        execution_engine._rng = state["_rng"]


class CanaryRunner:
    """Runs mechanism code against a deep copy of world state.

    Parameters
    ----------
    divergence_vitality_threshold:
        Percentage drop in total vitality that triggers a divergence flag.
        Default 50.0 means a >50% drop is flagged.
    """

    def __init__(self, divergence_vitality_threshold: float = 50.0):
        self.divergence_vitality_threshold = divergence_vitality_threshold

    def run_canary(
        self,
        execution_engine,
        mechanism_code: str,
        proposal_id: str,
        proposer_id: int,
        judge=None,
    ) -> CanaryReport:
        """Run mechanism code against a deep copy of state.

        Parameters
        ----------
        execution_engine:
            The live IslandExecution instance. State is deep-copied, not modified.
        mechanism_code:
            The mechanism code string to test.
        proposal_id:
            Unique identifier for this proposal.
        proposer_id:
            ID of the proposing agent.
        judge:
            Optional Judge instance for advisory opinion.

        Returns
        -------
        CanaryReport with before/after snapshots and divergence analysis.
        """
        # 1. Snapshot before
        snapshot_before = _compact_member_stats(execution_engine.current_members)
        alive_before = set(snapshot_before["_alive_ids"])
        total_before = snapshot_before["_total_vitality"]

        # 2. Deep-copy mutable state
        saved_state = _deep_copy_state(execution_engine)

        # 3. Execute mechanism code against the copy (applied to the engine temporarily)
        execution_error = None
        try:
            exec_env = {
                "execution_engine": execution_engine,
                "np": np,
            }
            exec(mechanism_code, exec_env)
            if "propose_modification" in exec_env:
                exec_env["propose_modification"](execution_engine)
        except Exception as e:
            execution_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        # 4. Snapshot after
        snapshot_after = _compact_member_stats(execution_engine.current_members)
        alive_after = set(snapshot_after["_alive_ids"])
        total_after = snapshot_after["_total_vitality"]

        # 5. Restore original state (canary is ephemeral)
        _apply_state(execution_engine, saved_state)

        # 6. Compute divergence
        agents_died = sorted(alive_before - alive_after)

        if total_before > 0:
            vitality_change_pct = ((total_after - total_before) / total_before) * 100.0
        else:
            vitality_change_pct = 0.0

        divergence_flags = []
        if abs(vitality_change_pct) > self.divergence_vitality_threshold:
            divergence_flags.append(f"vitality_drop_{abs(vitality_change_pct):.0f}%")
        if agents_died:
            divergence_flags.append(f"agents_died:{agents_died}")
        if execution_error:
            divergence_flags.append("execution_error")

        # 7. Get judge advisory opinion if available
        judge_opinion = None
        if judge is not None and execution_error is None:
            try:
                judge_opinion = judge.judge_proposal_advisory(
                    mechanism_code, proposer_id, "mechanism"
                )
            except Exception:
                # Judge errors should not block canary reporting
                pass

        return CanaryReport(
            proposal_id=proposal_id,
            proposer_id=proposer_id,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
            vitality_change_pct=vitality_change_pct,
            agents_died=agents_died,
            divergence_flags=divergence_flags,
            judge_opinion=judge_opinion,
            execution_error=execution_error,
        )
