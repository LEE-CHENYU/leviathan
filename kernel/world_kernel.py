"""WorldKernel -- facade wrapping IslandExecution for the distributed kernel API."""

import dataclasses
import hashlib
import json
import threading
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
from kernel.dag_runner import KernelDAGRunner
from kernel.round_metrics import compute_round_metrics
from kernel.constitution import Constitution
from kernel.oracle import OracleIdentity

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kernel.store import Store


class WorldKernel:
    """High-level facade over the simulation engine.

    Provides a clean, deterministic API for:
    - Inspecting world state (snapshots)
    - Submitting agent actions and mechanism proposals
    - Settling rounds with produce/consume
    - Generating immutable round receipts
    """

    def __init__(self, config: WorldConfig, save_path: str, store: Optional["Store"] = None) -> None:
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
        self._constitution = Constitution.default()
        if config.random_seed is not None:
            self._oracle = OracleIdentity.from_seed(config.random_seed)
        else:
            self._oracle = OracleIdentity.generate()
        self._store = store
        # Thread safety: protects kernel state from concurrent sim-thread
        # writes and API-thread reads.
        self._lock = threading.RLock()

    # ── Properties ────────────────────────────────

    @property
    def round_id(self) -> int:
        return self._round_id

    @property
    def constitution(self) -> Constitution:
        return self._constitution

    @property
    def oracle(self) -> OracleIdentity:
        return self._oracle

    # ── Snapshot ──────────────────────────────────

    def get_snapshot(self) -> WorldSnapshot:
        """Build a full WorldSnapshot from the current engine state."""
        with self._lock:
            members = []
            for m in self._execution.current_members:
                member_dict = {
                    "id": m.id,
                    "vitality": float(m.vitality),
                    "cargo": float(m.cargo),
                    "land_num": int(m.land_num),
                    "age": int(m.age),
                    "productivity": float(m.productivity),
                }
                # Include derived properties when available
                if hasattr(m, "strength"):
                    member_dict["strength"] = float(m.strength)
                if hasattr(m, "consumption"):
                    member_dict["consumption"] = float(m.consumption)
                if hasattr(m, "overall_productivity"):
                    member_dict["overall_productivity"] = float(m.overall_productivity)
                members.append(member_dict)

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
        with self._lock:
            self._round_id += 1
            self._execution.new_round()
            # Clean up old idempotency keys from SQLite (keep current round)
            if self._store:
                self._store.clear_idempotency_keys_before_round(self._round_id)
            self._idempotency_cache = {}

    def _cache_result(self, key: str, result: ActionResult) -> None:
        """Store an action result in both memory and SQLite."""
        self._idempotency_cache[key] = result
        if self._store:
            self._store.store_idempotency_key(
                key, self._round_id, json.dumps(dataclasses.asdict(result)),
            )

    def _lookup_idempotency(self, key: str) -> Optional[ActionResult]:
        """Check memory cache, then SQLite for an existing result."""
        if key in self._idempotency_cache:
            return self._idempotency_cache[key]
        if self._store:
            result_json = self._store.get_idempotency_result(key)
            if result_json:
                data = json.loads(result_json)
                result = ActionResult(**data)
                self._idempotency_cache[key] = result
                return result
        return None

    def accept_actions(self, actions: List[ActionIntent]) -> List[ActionResult]:
        """Execute agent actions and return results, with idempotency support."""
        with self._lock:
            results: List[ActionResult] = []
            for action in actions:
                # Check idempotency cache (memory + SQLite)
                cached = self._lookup_idempotency(action.idempotency_key)
                if cached is not None:
                    results.append(cached)
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
                    self._cache_result(action.idempotency_key, result)
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
                    self._cache_result(action.idempotency_key, result)
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
        with self._lock:
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
        with self._lock:
            return self._apply_intended_actions_unlocked(actions)

    def _apply_intended_actions_unlocked(self, actions: list) -> list:
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

    def settle_round(
        self,
        seed: int,
        judge_results: Optional[list] = None,
        mechanism_proposals: int = 0,
        mechanism_approvals: int = 0,
    ) -> RoundReceipt:
        """Run all settlement phases via DAG runner and build a round receipt."""
        with self._lock:
            return self._settle_round_unlocked(
                seed, judge_results, mechanism_proposals, mechanism_approvals,
            )

    def _settle_round_unlocked(
        self,
        seed: int,
        judge_results: Optional[list] = None,
        mechanism_proposals: int = 0,
        mechanism_approvals: int = 0,
    ) -> RoundReceipt:
        snap_before = self.get_snapshot()

        runner = KernelDAGRunner(self._execution)
        runner.run_settlement_phases()

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

        # Compute round metrics
        metrics = compute_round_metrics(
            members=snap_after.members,
            mechanism_proposals=mechanism_proposals,
            mechanism_approvals=mechanism_approvals,
        )

        receipt = RoundReceipt(
            round_id=self._round_id,
            seed=seed,
            snapshot_hash_before=snap_before.state_hash,
            snapshot_hash_after=snap_after.state_hash,
            accepted_action_ids=accepted_ids,
            rejected_action_ids=rejected_ids,
            activated_mechanism_ids=[],
            judge_results=judge_results or [],
            round_metrics=metrics,
            timestamp=deterministic_ts,
            constitution_hash=self._constitution.compute_hash(),
            world_public_key=self._oracle.world_public_key,
        )

        # Sign the receipt: serialize with oracle_signature=None, then sign
        receipt_dict = dataclasses.asdict(receipt)
        receipt_dict["oracle_signature"] = None
        canonical = json.dumps(receipt_dict, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, default=str).encode("utf-8")
        receipt.oracle_signature = self._oracle.sign(canonical)

        self._last_receipt = receipt
        return receipt

    def get_round_receipt(self) -> Optional[RoundReceipt]:
        """Return the last round receipt."""
        with self._lock:
            return self._last_receipt

    # ── Checkpoints ────────────────────────────────

    def get_available_checkpoints(self) -> List[Dict]:
        """Return checkpoint metadata that agents can see."""
        with self._lock:
            if not self._store:
                return []
            return self._store.list_snapshots()

    def restore_checkpoint(self, round_id: int) -> bool:
        """Restore world state to a checkpoint.

        Appends 'checkpoint_restored' event to log (preserves I-2: append-only).
        Does NOT delete any events — the restore is itself a new event.

        Returns True if restore succeeded, False if snapshot not found.
        """
        with self._lock:
            if not self._store:
                return False
            snapshot_data = self._store.get_snapshot(round_id)
            if snapshot_data is None:
                return False

            # Restore member state from snapshot
            members_data = snapshot_data.get("members", [])
            for mdata in members_data:
                if not isinstance(mdata, dict):
                    continue
                mid = mdata.get("id")
                if mid is None:
                    continue
                member = self._find_member_by_id(mid)
                if member is None:
                    continue
                for attr in ("vitality", "cargo", "land_num"):
                    if attr in mdata:
                        setattr(member, attr, mdata[attr])

            # Append checkpoint_restored event (preserves I-2)
            import datetime as _dt
            self._store.append_event(
                event_type="checkpoint_restored",
                round_id=self._round_id,
                timestamp=_dt.datetime.utcnow().isoformat(),
                payload={
                    "restored_to_round": round_id,
                    "current_round": self._round_id,
                },
                world_id=self._world_id,
                phase="checkpoint_restore",
            )
            return True

    def _collect_snapshot_data(self) -> Dict:
        """Collect current state into a snapshot dict for storage."""
        members = []
        for m in self._execution.current_members:
            members.append({
                "id": m.id,
                "vitality": float(m.vitality),
                "cargo": float(m.cargo),
                "land_num": int(m.land_num),
            })
        return {"members": members, "round_id": self._round_id}

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
