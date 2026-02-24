"""WorldKernel -- facade wrapping IslandExecution for the distributed kernel API."""

import dataclasses
import hashlib
import uuid
from typing import Dict, List, Optional, Tuple

from kernel.schemas import (
    ActionIntent,
    ActionResult,
    MechanismProposal,
    MechanismResult,
    RoundReceipt,
    WorldConfig,
    WorldSnapshot,
)
from kernel.receipt import compute_state_hash, compute_receipt_hash
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
        # Deterministic world_id when seed is provided, random otherwise
        if config.random_seed is not None:
            self._world_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"worldkernel-{config.random_seed}")
            )
        else:
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

    # ── Round lifecycle ───────────────────────────

    def begin_round(self) -> None:
        """Start a new round: increment round_id, reset caches."""
        self._round_id += 1
        self._execution.new_round()
        self._idempotency_cache = {}

    def accept_actions(self, actions: List[ActionIntent]) -> List[ActionResult]:
        """Execute agent actions and return results, with idempotency support."""
        results: List[ActionResult] = []
        for action in actions:
            # Check idempotency cache
            if action.idempotency_key in self._idempotency_cache:
                results.append(self._idempotency_cache[action.idempotency_key])
                continue

            # Resolve agent index
            agent_index = self._resolve_agent_index(action.agent_id)

            if agent_index is None:
                result = ActionResult(
                    agent_id=action.agent_id,
                    success=False,
                    old_stats={},
                    new_stats={},
                    performance_change=0.0,
                    messages_sent=[],
                    error=f"Agent {action.agent_id} not found",
                )
                self._idempotency_cache[action.idempotency_key] = result
                results.append(result)
                continue

            member = self._execution.current_members[agent_index]

            # Capture old stats
            old_survival = float(self._execution.compute_survival_chance(member))
            old_relation = float(self._execution.compute_relation_balance(member))
            old_stats = {
                "vitality": float(member.vitality),
                "cargo": float(member.cargo),
                "land": float(member.land_num),
                "survival_chance": old_survival,
                "relation_balance": old_relation,
            }

            # Run through sandbox
            ctx = SandboxContext(
                execution_engine=self._execution,
                member_index=agent_index,
            )
            sandbox_result = self._sandbox.execute_agent_code(action.code, ctx)

            if not sandbox_result.success:
                result = ActionResult(
                    agent_id=action.agent_id,
                    success=False,
                    old_stats=old_stats,
                    new_stats=old_stats,
                    performance_change=0.0,
                    messages_sent=[],
                    error=sandbox_result.error,
                )
                self._idempotency_cache[action.idempotency_key] = result
                results.append(result)
                continue

            # Capture new stats (member may still be alive after action)
            new_survival = float(self._execution.compute_survival_chance(member))
            new_relation = float(self._execution.compute_relation_balance(member))
            new_stats = {
                "vitality": float(member.vitality),
                "cargo": float(member.cargo),
                "land": float(member.land_num),
                "survival_chance": new_survival,
                "relation_balance": new_relation,
            }

            performance_change = new_survival - old_survival

            result = ActionResult(
                agent_id=action.agent_id,
                success=True,
                old_stats=old_stats,
                new_stats=new_stats,
                performance_change=performance_change,
                messages_sent=[],
            )
            self._idempotency_cache[action.idempotency_key] = result
            results.append(result)

        return results

    def accept_mechanisms(
        self, mechanisms: List[MechanismProposal]
    ) -> List[MechanismResult]:
        """Execute mechanism proposals and return results."""
        results: List[MechanismResult] = []
        for mech in mechanisms:
            ctx = SandboxContext(
                execution_engine=self._execution,
                member_index=0,
            )
            sandbox_result = self._sandbox.execute_mechanism_code(mech.code, ctx)

            results.append(
                MechanismResult(
                    proposal_id=mech.proposal_id,
                    executed=sandbox_result.success,
                    error=sandbox_result.error if not sandbox_result.success else None,
                )
            )
        return results

    def apply_intended_actions(self, actions: list) -> list:
        """Apply intended actions from an EngineProxy to the real engine.

        Each action is a dict like {"action": "attack", "member_id": 1, "target_id": 2}.
        Returns a list of result dicts with {"action": ..., "applied": bool, "error": ...}.
        """
        results = []
        for act in actions:
            action_type = act.get("action")
            try:
                if action_type == "attack":
                    m1 = self._find_member_by_id(act["member_id"])
                    m2 = self._find_member_by_id(act["target_id"])
                    if m1 and m2:
                        self._execution.attack(m1, m2)
                        results.append({"action": action_type, "applied": True})
                    else:
                        results.append({"action": action_type, "applied": False, "error": "member not found"})
                elif action_type == "offer":
                    m1 = self._find_member_by_id(act["member_id"])
                    m2 = self._find_member_by_id(act["target_id"])
                    if m1 and m2:
                        self._execution.offer(m1, m2)
                        results.append({"action": action_type, "applied": True})
                    else:
                        results.append({"action": action_type, "applied": False, "error": "member not found"})
                elif action_type == "offer_land":
                    m1 = self._find_member_by_id(act["member_id"])
                    m2 = self._find_member_by_id(act["target_id"])
                    if m1 and m2:
                        self._execution.offer_land(m1, m2)
                        results.append({"action": action_type, "applied": True})
                    else:
                        results.append({"action": action_type, "applied": False, "error": "member not found"})
                elif action_type == "expand":
                    m1 = self._find_member_by_id(act["member_id"])
                    if m1:
                        self._execution.expand(m1)
                        results.append({"action": action_type, "applied": True})
                    else:
                        results.append({"action": action_type, "applied": False, "error": "member not found"})
                elif action_type == "message":
                    self._execution.send_message(act["from_id"], act["to_id"], act["message"])
                    results.append({"action": action_type, "applied": True})
                else:
                    results.append({"action": action_type, "applied": False, "error": f"unknown action: {action_type}"})
            except Exception as e:
                results.append({"action": action_type, "applied": False, "error": str(e)})
        return results

    def settle_round(self, seed: int) -> RoundReceipt:
        """Run produce/consume and build a deterministic round receipt."""
        snap_before = self.get_snapshot()

        self._execution.produce()
        self._execution.consume()

        snap_after = self.get_snapshot()

        # Build deterministic timestamp from hash of seed + round_id
        ts_input = f"{seed}:{self._round_id}".encode("utf-8")
        deterministic_ts = hashlib.sha256(ts_input).hexdigest()[:16]

        # Collect action/mechanism IDs from cache
        accepted_ids = [
            key for key, res in self._idempotency_cache.items() if res.success
        ]
        rejected_ids = [
            key for key, res in self._idempotency_cache.items() if not res.success
        ]

        receipt = RoundReceipt(
            round_id=self._round_id,
            seed=seed,
            snapshot_hash_before=snap_before.state_hash,
            snapshot_hash_after=snap_after.state_hash,
            accepted_action_ids=accepted_ids,
            rejected_action_ids=rejected_ids,
            activated_mechanism_ids=[],
            judge_results=[],
            round_metrics={},
            timestamp=deterministic_ts,
        )
        self._last_receipt = receipt
        return receipt

    def get_round_receipt(self) -> Optional[RoundReceipt]:
        """Return the last round receipt."""
        return self._last_receipt

    # ── Private helpers ───────────────────────────

    def _find_member_by_id(self, member_id: int):
        """Find a Member object by permanent id."""
        for m in self._execution.current_members:
            if m.id == member_id:
                return m
        return None

    def _resolve_agent_index(self, agent_id: int) -> Optional[int]:
        """Find the index of a member by its id in current_members."""
        for idx, member in enumerate(self._execution.current_members):
            if member.id == agent_id:
                return idx
        return None
