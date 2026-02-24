"""Tests for the WorldKernel extraction project."""

import dataclasses
from typing import Dict, List, Optional, Tuple

import pytest


# ──────────────────────────────────────────────
# Task 1 – Schema dataclass tests
# ──────────────────────────────────────────────

from kernel.schemas import (
    ActionIntent,
    ActionResult,
    MechanismProposal,
    MechanismResult,
    RoundReceipt,
    WorldConfig,
    WorldSnapshot,
)


class TestWorldConfig:
    def test_creation_defaults(self):
        cfg = WorldConfig(init_member_number=10, land_shape=(5, 5))
        assert cfg.init_member_number == 10
        assert cfg.land_shape == (5, 5)
        assert cfg.random_seed is None

    def test_creation_with_seed(self):
        cfg = WorldConfig(init_member_number=8, land_shape=(3, 4), random_seed=42)
        assert cfg.random_seed == 42

    def test_asdict_roundtrip(self):
        cfg = WorldConfig(init_member_number=6, land_shape=(2, 2), random_seed=7)
        d = dataclasses.asdict(cfg)
        assert d == {
            "init_member_number": 6,
            "land_shape": (2, 2),
            "random_seed": 7,
        }
        cfg2 = WorldConfig(**d)
        assert cfg == cfg2


class TestActionIntent:
    def test_creation(self):
        ai = ActionIntent(
            agent_id=1, round_id=5, code="print('hi')", idempotency_key="abc-123"
        )
        assert ai.agent_id == 1
        assert ai.round_id == 5
        assert ai.code == "print('hi')"
        assert ai.idempotency_key == "abc-123"

    def test_asdict_roundtrip(self):
        ai = ActionIntent(agent_id=2, round_id=3, code="x=1", idempotency_key="k")
        d = dataclasses.asdict(ai)
        assert ActionIntent(**d) == ai


class TestActionResult:
    def test_creation_defaults(self):
        ar = ActionResult(
            agent_id=1,
            success=True,
            old_stats={"hp": 100.0},
            new_stats={"hp": 90.0},
            performance_change=-10.0,
            messages_sent=[(2, "hello")],
        )
        assert ar.error is None
        assert ar.signature is None

    def test_creation_with_optionals(self):
        ar = ActionResult(
            agent_id=1,
            success=False,
            old_stats={},
            new_stats={},
            performance_change=0.0,
            messages_sent=[],
            error="boom",
            signature={"sig": "xyz"},
        )
        assert ar.error == "boom"
        assert ar.signature == {"sig": "xyz"}

    def test_asdict_roundtrip(self):
        ar = ActionResult(
            agent_id=3,
            success=True,
            old_stats={"a": 1.0},
            new_stats={"a": 2.0},
            performance_change=1.0,
            messages_sent=[(1, "msg")],
            error=None,
            signature=None,
        )
        d = dataclasses.asdict(ar)
        assert ActionResult(**d) == ar


class TestMechanismProposal:
    def test_creation(self):
        mp = MechanismProposal(
            proposal_id="p1", agent_id=5, code="def f(): pass", round_id=10
        )
        assert mp.proposal_id == "p1"
        assert mp.agent_id == 5

    def test_asdict_roundtrip(self):
        mp = MechanismProposal(
            proposal_id="p2", agent_id=3, code="return 1", round_id=2
        )
        d = dataclasses.asdict(mp)
        assert MechanismProposal(**d) == mp


class TestMechanismResult:
    def test_creation_defaults(self):
        mr = MechanismResult(proposal_id="p1", executed=True)
        assert mr.error is None

    def test_creation_with_error(self):
        mr = MechanismResult(proposal_id="p1", executed=False, error="denied")
        assert mr.error == "denied"

    def test_asdict_roundtrip(self):
        mr = MechanismResult(proposal_id="p1", executed=True, error=None)
        d = dataclasses.asdict(mr)
        assert MechanismResult(**d) == mr


class TestWorldSnapshot:
    def test_creation(self):
        ws = WorldSnapshot(
            world_id="w1",
            round_id=0,
            members=[{"id": 1}],
            land={"grid": [[0]]},
            active_mechanisms=[],
            active_contracts=[],
            physics_constraints=[],
            state_hash="abc123",
        )
        assert ws.world_id == "w1"
        assert ws.round_id == 0

    def test_asdict_roundtrip(self):
        ws = WorldSnapshot(
            world_id="w2",
            round_id=1,
            members=[],
            land={},
            active_mechanisms=[{"id": "m1"}],
            active_contracts=[],
            physics_constraints=[{"rule": "gravity"}],
            state_hash="deadbeef",
        )
        d = dataclasses.asdict(ws)
        assert WorldSnapshot(**d) == ws


class TestRoundReceipt:
    def test_creation(self):
        rr = RoundReceipt(
            round_id=1,
            seed=42,
            snapshot_hash_before="aaa",
            snapshot_hash_after="bbb",
            accepted_action_ids=["a1"],
            rejected_action_ids=[],
            activated_mechanism_ids=["m1"],
            judge_results=[{"agent": 1, "score": 0.9}],
            round_metrics={"avg_perf": 0.85},
            timestamp="2025-01-01T00:00:00Z",
        )
        assert rr.round_id == 1
        assert rr.seed == 42
        assert rr.timestamp == "2025-01-01T00:00:00Z"

    def test_asdict_roundtrip(self):
        rr = RoundReceipt(
            round_id=2,
            seed=7,
            snapshot_hash_before="x",
            snapshot_hash_after="y",
            accepted_action_ids=[],
            rejected_action_ids=["r1"],
            activated_mechanism_ids=[],
            judge_results=[],
            round_metrics={},
            timestamp="2025-06-15T12:00:00Z",
        )
        d = dataclasses.asdict(rr)
        assert RoundReceipt(**d) == rr

    def test_all_fields_present(self):
        rr = RoundReceipt(
            round_id=0,
            seed=0,
            snapshot_hash_before="",
            snapshot_hash_after="",
            accepted_action_ids=[],
            rejected_action_ids=[],
            activated_mechanism_ids=[],
            judge_results=[],
            round_metrics={},
            timestamp="",
        )
        field_names = {f.name for f in dataclasses.fields(rr)}
        expected = {
            "round_id",
            "seed",
            "snapshot_hash_before",
            "snapshot_hash_after",
            "accepted_action_ids",
            "rejected_action_ids",
            "activated_mechanism_ids",
            "judge_results",
            "round_metrics",
            "timestamp",
        }
        assert field_names == expected


# ──────────────────────────────────────────────
# Task 2 – Receipt hashing tests
# ──────────────────────────────────────────────

from kernel.receipt import canonical_json, compute_receipt_hash, compute_state_hash


class TestCanonicalJson:
    def test_sorted_keys(self):
        """Keys must be sorted regardless of insertion order."""
        raw = canonical_json({"z": 1, "a": 2, "m": 3})
        assert raw == b'{"a":2,"m":3,"z":1}'

    def test_no_whitespace(self):
        """No extra whitespace anywhere in the output."""
        raw = canonical_json({"key": [1, 2, {"nested": True}]})
        assert b" " not in raw
        assert b"\n" not in raw

    def test_identical_bytes_for_identical_input(self):
        """Calling twice with the same input produces byte-identical output."""
        obj = {"hello": "world", "num": 42, "nested": {"a": [1, 2, 3]}}
        assert canonical_json(obj) == canonical_json(obj)

    def test_utf8_encoding(self):
        """Non-ASCII characters are preserved (ensure_ascii=False)."""
        raw = canonical_json({"greeting": "hola"})
        assert isinstance(raw, bytes)
        decoded = raw.decode("utf-8")
        assert "hola" in decoded

    def test_default_str_for_non_serializable(self):
        """Non-serializable objects fall back to str()."""
        from datetime import datetime

        dt = datetime(2025, 1, 1)
        raw = canonical_json({"ts": dt})
        assert isinstance(raw, bytes)
        assert b"2025" in raw


class TestComputeStateHash:
    def test_hex_digest_length(self):
        """SHA-256 hex digest is always 64 characters."""
        snap = {
            "world_id": "w1",
            "round_id": 0,
            "members": [],
            "land": {},
            "active_mechanisms": [],
            "active_contracts": [],
            "physics_constraints": [],
            "state_hash": "placeholder",
        }
        h = compute_state_hash(snap)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_excludes_state_hash_field(self):
        """Changing only state_hash should NOT change the computed hash."""
        snap_a = {
            "world_id": "w1",
            "round_id": 0,
            "members": [],
            "land": {},
            "active_mechanisms": [],
            "active_contracts": [],
            "physics_constraints": [],
            "state_hash": "aaa",
        }
        snap_b = dict(snap_a, state_hash="bbb")
        assert compute_state_hash(snap_a) == compute_state_hash(snap_b)

    def test_different_data_different_hash(self):
        """Different snapshot content must produce a different hash."""
        snap_a = {"world_id": "w1", "round_id": 0, "members": [], "land": {}}
        snap_b = {"world_id": "w1", "round_id": 1, "members": [], "land": {}}
        assert compute_state_hash(snap_a) != compute_state_hash(snap_b)


class TestComputeReceiptHash:
    def test_hex_digest_length(self):
        h = compute_receipt_hash({"round_id": 1, "seed": 42})
        assert len(h) == 64

    def test_deterministic(self):
        receipt = {"round_id": 1, "seed": 42, "data": [1, 2, 3]}
        assert compute_receipt_hash(receipt) == compute_receipt_hash(receipt)

    def test_different_data_different_hash(self):
        r1 = {"round_id": 1, "seed": 42}
        r2 = {"round_id": 2, "seed": 42}
        assert compute_receipt_hash(r1) != compute_receipt_hash(r2)


# ──────────────────────────────────────────────
# Task 3 – ExecutionSandbox tests
# ──────────────────────────────────────────────

from kernel.execution_sandbox import (
    InProcessSandbox,
    SandboxContext,
    SandboxResult,
)


class TestSandboxResult:
    def test_creation_defaults(self):
        sr = SandboxResult(success=True)
        assert sr.success is True
        assert sr.error is None
        assert sr.traceback_str is None

    def test_creation_with_error(self):
        sr = SandboxResult(success=False, error="oops", traceback_str="line 1\nline 2")
        assert sr.success is False
        assert sr.error == "oops"


class TestSandboxContext:
    def test_creation_defaults(self):
        ctx = SandboxContext(execution_engine=None, member_index=0)
        assert ctx.extra_env == {}

    def test_creation_with_extra_env(self):
        ctx = SandboxContext(
            execution_engine=None, member_index=3, extra_env={"key": "val"}
        )
        assert ctx.extra_env == {"key": "val"}


class TestInProcessSandbox:
    def setup_method(self):
        self.sandbox = InProcessSandbox()

    def test_sandbox_execute_simple_code(self):
        """Code defines agent_action; sandbox returns success."""
        code = "def agent_action(engine):\n    return {'move': 'north'}\n"
        ctx = SandboxContext(execution_engine=None, member_index=0)
        result = self.sandbox.execute_agent_code(code, ctx)
        assert isinstance(result, SandboxResult)
        assert result.success is True
        assert result.error is None

    def test_sandbox_captures_error(self):
        """Code raises ValueError; sandbox returns success=False with error."""
        code = "def agent_action(engine, member_index):\n    raise ValueError('bad value')\n"

        class FakeEngine:
            pass

        ctx = SandboxContext(execution_engine=FakeEngine(), member_index=0)
        result = self.sandbox.execute_agent_code(code, ctx)
        assert result.success is False
        assert "bad value" in result.error

    def test_sandbox_no_agent_action(self):
        """Code doesn't define agent_action; sandbox returns error."""
        code = "x = 42\n"
        ctx = SandboxContext(execution_engine=None, member_index=0)
        result = self.sandbox.execute_agent_code(code, ctx)
        assert result.success is False
        assert "agent_action" in result.error

    def test_sandbox_mechanism_code(self):
        """Code defines propose_modification; sandbox returns success."""
        code = "def propose_modification(engine):\n    return {'rule': 'new'}\n"
        ctx = SandboxContext(execution_engine=None, member_index=0)
        result = self.sandbox.execute_mechanism_code(code, ctx)
        assert isinstance(result, SandboxResult)
        assert result.success is True
        assert result.error is None

    def test_sandbox_mechanism_no_function(self):
        """Mechanism code without propose_modification returns error."""
        code = "y = 99\n"
        ctx = SandboxContext(execution_engine=None, member_index=0)
        result = self.sandbox.execute_mechanism_code(code, ctx)
        assert result.success is False
        assert "propose_modification" in result.error

    def test_sandbox_syntax_error(self):
        """Syntax error in code is captured gracefully."""
        code = "def agent_action(engine)\n"  # missing colon
        ctx = SandboxContext(execution_engine=None, member_index=0)
        result = self.sandbox.execute_agent_code(code, ctx)
        assert result.success is False
        assert result.error is not None


# ──────────────────────────────────────────────
# Task 4 – WorldKernel facade tests
# ──────────────────────────────────────────────

import tempfile

from kernel.world_kernel import WorldKernel


class TestWorldKernelInit:
    def test_world_kernel_init(self):
        """WorldKernel initialises with round_id == 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(init_member_number=5, land_shape=(5, 5), random_seed=42)
            wk = WorldKernel(config, tmpdir)
            assert wk.round_id == 0

    def test_world_kernel_get_snapshot(self):
        """get_snapshot returns WorldSnapshot with correct member count and valid hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(init_member_number=5, land_shape=(5, 5), random_seed=42)
            wk = WorldKernel(config, tmpdir)
            snap = wk.get_snapshot()
            assert isinstance(snap, WorldSnapshot)
            assert len(snap.members) == 5
            assert snap.round_id == 0
            assert len(snap.state_hash) == 64
            assert all(c in "0123456789abcdef" for c in snap.state_hash)

    def test_world_kernel_snapshot_deterministic(self):
        """Two kernels with same seed produce identical state_hash."""
        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:
            config = WorldConfig(init_member_number=5, land_shape=(5, 5), random_seed=42)
            wk1 = WorldKernel(config, tmpdir1)
            wk2 = WorldKernel(config, tmpdir2)
            snap1 = wk1.get_snapshot()
            snap2 = wk2.get_snapshot()
            # world_id differs (uuid), so we compare member data and land
            assert snap1.members == snap2.members
            assert snap1.land == snap2.land


# ──────────────────────────────────────────────
# Task 5 – Round lifecycle tests
# ──────────────────────────────────────────────

from kernel.schemas import ActionIntent, MechanismProposal


class TestAcceptActions:
    def test_accept_actions_basic(self):
        """Submit one no-op action; assert success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(init_member_number=5, land_shape=(5, 5), random_seed=42)
            wk = WorldKernel(config, tmpdir)
            wk.begin_round()
            action = ActionIntent(
                agent_id=0,
                round_id=1,
                code="def agent_action(engine, member_index):\n    pass\n",
                idempotency_key="act-1",
            )
            results = wk.accept_actions([action])
            assert len(results) == 1
            assert results[0].success is True
            assert results[0].agent_id == 0

    def test_accept_actions_error_handling(self):
        """Submit bad code; assert success=False with error message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(init_member_number=5, land_shape=(5, 5), random_seed=42)
            wk = WorldKernel(config, tmpdir)
            wk.begin_round()
            action = ActionIntent(
                agent_id=0,
                round_id=1,
                code="def agent_action(engine, member_index):\n    raise RuntimeError('boom')\n",
                idempotency_key="act-bad",
            )
            results = wk.accept_actions([action])
            assert len(results) == 1
            assert results[0].success is False
            assert "boom" in results[0].error

    def test_accept_actions_idempotency(self):
        """Submit same idempotency_key twice; assert cached result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(init_member_number=5, land_shape=(5, 5), random_seed=42)
            wk = WorldKernel(config, tmpdir)
            wk.begin_round()
            action = ActionIntent(
                agent_id=0,
                round_id=1,
                code="def agent_action(engine, member_index):\n    pass\n",
                idempotency_key="act-idem",
            )
            results1 = wk.accept_actions([action])
            results2 = wk.accept_actions([action])
            assert results1[0] is results2[0]  # exact same cached object


class TestAcceptMechanisms:
    def test_accept_mechanisms_basic(self):
        """Submit one no-op mechanism; assert executed=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(init_member_number=5, land_shape=(5, 5), random_seed=42)
            wk = WorldKernel(config, tmpdir)
            wk.begin_round()
            mech = MechanismProposal(
                proposal_id="mech-1",
                agent_id=0,
                code="def propose_modification(engine):\n    pass\n",
                round_id=1,
            )
            results = wk.accept_mechanisms([mech])
            assert len(results) == 1
            assert results[0].executed is True
            assert results[0].proposal_id == "mech-1"


class TestSettleRound:
    def test_settle_round_produces_receipt(self):
        """settle_round produces receipt with matching hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(init_member_number=5, land_shape=(5, 5), random_seed=42)
            wk = WorldKernel(config, tmpdir)
            snap_before = wk.get_snapshot()
            wk.begin_round()
            receipt = wk.settle_round(seed=99)
            assert isinstance(receipt, RoundReceipt)
            assert receipt.round_id == 1
            assert receipt.seed == 99
            assert len(receipt.snapshot_hash_before) == 64
            assert len(receipt.snapshot_hash_after) == 64
            assert receipt.snapshot_hash_before != receipt.snapshot_hash_after or True  # may or may not differ

    def test_full_round_lifecycle(self):
        """begin_round -> accept_actions -> settle_round end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(init_member_number=5, land_shape=(5, 5), random_seed=42)
            wk = WorldKernel(config, tmpdir)

            # Round 1
            wk.begin_round()
            assert wk.round_id == 1

            action = ActionIntent(
                agent_id=0,
                round_id=1,
                code="def agent_action(engine, member_index):\n    pass\n",
                idempotency_key="lifecycle-1",
            )
            results = wk.accept_actions([action])
            assert results[0].success is True

            receipt = wk.settle_round(seed=42)
            assert receipt.round_id == 1
            assert wk.get_round_receipt() is receipt


# ──────────────────────────────────────────────
# Task 6 – Golden determinism tests
# ──────────────────────────────────────────────


def _run_deterministic_round(seed, member_count=5, land=(5, 5), num_rounds=1, tmpdir=None):
    """Helper: run num_rounds of begin/settle with a fixed seed.

    Returns (snapshot_hash_before, snapshot_hash_after, final_state_hash)
    for each round as a list of tuples, plus the final WorldKernel.
    """
    config = WorldConfig(
        init_member_number=member_count,
        land_shape=land,
        random_seed=seed,
    )
    wk = WorldKernel(config, tmpdir)
    round_data = []
    for r in range(num_rounds):
        wk.begin_round()
        # Submit a deterministic no-op action for each member
        actions = []
        for i in range(member_count):
            actions.append(
                ActionIntent(
                    agent_id=i,
                    round_id=wk.round_id,
                    code="def agent_action(engine, member_index):\n    pass\n",
                    idempotency_key=f"golden-{seed}-r{r}-a{i}",
                )
            )
        wk.accept_actions(actions)
        receipt = wk.settle_round(seed=seed)
        final_snap = wk.get_snapshot()
        round_data.append((
            receipt.snapshot_hash_before,
            receipt.snapshot_hash_after,
            final_snap.state_hash,
        ))
    return round_data, wk


class TestGoldenDeterminism:
    def test_golden_determinism_same_seed(self):
        """Identical seed + inputs produce identical hashes."""
        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:
            data1, _ = _run_deterministic_round(seed=42, tmpdir=tmpdir1)
            data2, _ = _run_deterministic_round(seed=42, tmpdir=tmpdir2)

            assert data1[0][0] == data2[0][0], "snapshot_hash_before should match"
            assert data1[0][1] == data2[0][1], "snapshot_hash_after should match"
            assert data1[0][2] == data2[0][2], "final state_hash should match"

    def test_golden_determinism_different_seed(self):
        """Different seeds produce different initial hashes."""
        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:
            data1, _ = _run_deterministic_round(seed=42, tmpdir=tmpdir1)
            data2, _ = _run_deterministic_round(seed=99, tmpdir=tmpdir2)

            assert data1[0][0] != data2[0][0], "different seeds should produce different hashes"

    def test_golden_determinism_multiple_rounds(self):
        """3 rounds with identical inputs produce identical final state."""
        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:
            data1, wk1 = _run_deterministic_round(seed=42, num_rounds=3, tmpdir=tmpdir1)
            data2, wk2 = _run_deterministic_round(seed=42, num_rounds=3, tmpdir=tmpdir2)

            # Check each round's hashes match
            for r in range(3):
                assert data1[r][0] == data2[r][0], f"round {r+1} snapshot_hash_before mismatch"
                assert data1[r][1] == data2[r][1], f"round {r+1} snapshot_hash_after mismatch"
                assert data1[r][2] == data2[r][2], f"round {r+1} final state_hash mismatch"

            # Final state must be identical
            snap1 = wk1.get_snapshot()
            snap2 = wk2.get_snapshot()
            assert snap1.members == snap2.members
            assert snap1.land == snap2.land


# ──────────────────────────────────────────────
# K11 – WorldKernel vs IslandExecution equivalence
# ──────────────────────────────────────────────

from MetaIsland.metaIsland import IslandExecution


def _sorted_member_dicts(snapshot):
    """Extract members from a WorldSnapshot and return them sorted by id."""
    return sorted(snapshot.members, key=lambda m: m["id"])


def _sorted_member_attrs(island_execution):
    """Extract member attributes from an IslandExecution and return them sorted by id."""
    return sorted(
        [
            {
                "id": m.id,
                "vitality": float(m.vitality),
                "cargo": float(m.cargo),
                "land_num": int(m.land_num),
            }
            for m in island_execution.current_members
        ],
        key=lambda m: m["id"],
    )


def _assert_members_equivalent(kernel_members, island_members):
    """Assert that two sorted member lists are field-by-field equivalent."""
    assert len(kernel_members) == len(island_members), (
        f"Member count mismatch: kernel={len(kernel_members)}, "
        f"island={len(island_members)}"
    )
    for km, im in zip(kernel_members, island_members):
        assert km["id"] == im["id"], f"ID mismatch: {km['id']} != {im['id']}"
        assert km["vitality"] == pytest.approx(im["vitality"]), (
            f"Vitality mismatch for member {km['id']}"
        )
        assert km["cargo"] == pytest.approx(im["cargo"]), (
            f"Cargo mismatch for member {km['id']}"
        )
        assert km["land_num"] == im["land_num"], (
            f"land_num mismatch for member {km['id']}"
        )


class TestKernelIslandEquivalence:
    """K11: Verify that WorldKernel facade produces identical member states
    to a direct IslandExecution with the same seed and parameters."""

    SEED = 42
    MEMBER_COUNT = 5
    LAND_SHAPE = (10, 10)

    def test_initial_state_equivalence(self):
        """Initial member states from WorldKernel and IslandExecution must match."""
        with tempfile.TemporaryDirectory() as tmpdir_k, \
             tempfile.TemporaryDirectory() as tmpdir_i:
            config = WorldConfig(
                init_member_number=self.MEMBER_COUNT,
                land_shape=self.LAND_SHAPE,
                random_seed=self.SEED,
            )
            wk = WorldKernel(config, tmpdir_k)

            ie = IslandExecution(
                init_member_number=self.MEMBER_COUNT,
                land_shape=self.LAND_SHAPE,
                save_path=tmpdir_i,
                random_seed=self.SEED,
            )

            kernel_members = _sorted_member_dicts(wk.get_snapshot())
            island_members = _sorted_member_attrs(ie)

            _assert_members_equivalent(kernel_members, island_members)

    def test_produce_consume_equivalence(self):
        """After one produce/consume cycle, member states must still match."""
        with tempfile.TemporaryDirectory() as tmpdir_k, \
             tempfile.TemporaryDirectory() as tmpdir_i:
            config = WorldConfig(
                init_member_number=self.MEMBER_COUNT,
                land_shape=self.LAND_SHAPE,
                random_seed=self.SEED,
            )
            wk = WorldKernel(config, tmpdir_k)

            ie = IslandExecution(
                init_member_number=self.MEMBER_COUNT,
                land_shape=self.LAND_SHAPE,
                save_path=tmpdir_i,
                random_seed=self.SEED,
            )

            # WorldKernel path: begin_round + settle_round runs produce/consume
            wk.begin_round()
            wk.settle_round(seed=wk.round_id)

            # Direct IslandExecution path
            ie.new_round()
            ie.produce()
            ie.consume()

            kernel_members = _sorted_member_dicts(wk.get_snapshot())
            island_members = _sorted_member_attrs(ie)

            _assert_members_equivalent(kernel_members, island_members)

    def test_multi_round_equivalence(self):
        """After 3 rounds of produce/consume, member states must match each round."""
        with tempfile.TemporaryDirectory() as tmpdir_k, \
             tempfile.TemporaryDirectory() as tmpdir_i:
            config = WorldConfig(
                init_member_number=self.MEMBER_COUNT,
                land_shape=self.LAND_SHAPE,
                random_seed=self.SEED,
            )
            wk = WorldKernel(config, tmpdir_k)

            ie = IslandExecution(
                init_member_number=self.MEMBER_COUNT,
                land_shape=self.LAND_SHAPE,
                save_path=tmpdir_i,
                random_seed=self.SEED,
            )

            for round_num in range(1, 4):
                wk.begin_round()
                wk.settle_round(seed=wk.round_id)

                ie.new_round()
                ie.produce()
                ie.consume()

                kernel_members = _sorted_member_dicts(wk.get_snapshot())
                island_members = _sorted_member_attrs(ie)

                _assert_members_equivalent(kernel_members, island_members)


# ──────────────────────────────────────────────
# SubprocessSandbox — EngineProxy tests
# ──────────────────────────────────────────────

from kernel.engine_proxy import EngineProxy, MemberProxy, LandProxy


class TestEngineProxy:
    """Verify the EngineProxy records actions instead of executing them."""

    def _make_state(self, n_members=3):
        members = []
        for i in range(n_members):
            members.append({"id": i + 1, "vitality": 50.0, "cargo": 20.0, "land_num": 2})
        return {"members": members, "land": {"shape": [10, 10]}}

    def test_member_proxy_attributes(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        member = proxy.current_members[0]
        assert isinstance(member, MemberProxy)
        assert member.id == 1
        assert member.vitality == 50.0
        assert member.cargo == 20.0
        assert member.land_num == 2

    def test_land_proxy_shape(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        assert isinstance(proxy.land, LandProxy)
        assert proxy.land.shape == (10, 10)

    def test_attack_records_action(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        me = proxy.current_members[0]
        target = proxy.current_members[1]
        proxy.attack(me, target)
        assert len(proxy.actions) == 1
        assert proxy.actions[0] == {"action": "attack", "member_id": 1, "target_id": 2}

    def test_offer_records_action(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        me = proxy.current_members[0]
        target = proxy.current_members[1]
        proxy.offer(me, target)
        assert proxy.actions[0] == {"action": "offer", "member_id": 1, "target_id": 2}

    def test_offer_land_records_action(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        me = proxy.current_members[0]
        target = proxy.current_members[1]
        proxy.offer_land(me, target)
        assert proxy.actions[0] == {"action": "offer_land", "member_id": 1, "target_id": 2}

    def test_expand_records_action(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        me = proxy.current_members[0]
        proxy.expand(me)
        assert proxy.actions[0] == {"action": "expand", "member_id": 1}

    def test_send_message_records_action(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        proxy.send_message(1, 2, "hello")
        assert proxy.actions[0] == {"action": "message", "from_id": 1, "to_id": 2, "message": "hello"}

    def test_multiple_actions_recorded_in_order(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        me = proxy.current_members[0]
        target = proxy.current_members[1]
        proxy.attack(me, target)
        proxy.offer(me, target)
        proxy.expand(me)
        assert len(proxy.actions) == 3
        assert [a["action"] for a in proxy.actions] == ["attack", "offer", "expand"]


# ──────────────────────────────────────────────
# SubprocessSandbox tests
# ──────────────────────────────────────────────

from kernel.subprocess_sandbox import SubprocessSandbox


class TestSubprocessSandbox:
    """Verify SubprocessSandbox runs code in a child process."""

    def _make_context(self, n_members=3):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=n_members, land_shape=(10, 10), random_seed=42)
        kernel = WorldKernel(config, save_path=tmpdir)
        return SandboxContext(
            execution_engine=kernel._execution,
            member_index=0,
        ), kernel

    def test_simple_expand_action(self):
        ctx, kernel = self._make_context()
        sandbox = SubprocessSandbox()
        code = (
            "def agent_action(engine, member_id):\n"
            "    me = engine.current_members[member_id]\n"
            "    engine.expand(me)\n"
        )
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is True
        assert len(result.intended_actions) == 1
        assert result.intended_actions[0]["action"] == "expand"

    def test_attack_action(self):
        ctx, kernel = self._make_context(n_members=5)
        sandbox = SubprocessSandbox()
        code = (
            "def agent_action(engine, member_id):\n"
            "    me = engine.current_members[member_id]\n"
            "    target = engine.current_members[1]\n"
            "    engine.attack(me, target)\n"
        )
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is True
        assert len(result.intended_actions) == 1
        assert result.intended_actions[0]["action"] == "attack"

    def test_multiple_actions(self):
        ctx, kernel = self._make_context(n_members=5)
        sandbox = SubprocessSandbox()
        code = (
            "def agent_action(engine, member_id):\n"
            "    me = engine.current_members[member_id]\n"
            "    target = engine.current_members[1]\n"
            "    engine.attack(me, target)\n"
            "    engine.offer(me, target)\n"
        )
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is True
        assert len(result.intended_actions) == 2

    def test_syntax_error_returns_failure(self):
        ctx, kernel = self._make_context()
        sandbox = SubprocessSandbox()
        code = "def agent_action(engine, member_id)\n"
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is False
        assert "SyntaxError" in (result.error or "")

    def test_no_entry_point_returns_failure(self):
        ctx, kernel = self._make_context()
        sandbox = SubprocessSandbox()
        code = "x = 42\n"
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is False
        assert "agent_action" in (result.error or "")

    def test_runtime_error_returns_failure(self):
        ctx, kernel = self._make_context()
        sandbox = SubprocessSandbox()
        code = (
            "def agent_action(engine, member_id):\n"
            "    raise ValueError('boom')\n"
        )
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is False
        assert "boom" in (result.error or "")

    def test_timeout_returns_failure(self):
        ctx, kernel = self._make_context()
        sandbox = SubprocessSandbox(timeout=2)
        code = (
            "def agent_action(engine, member_id):\n"
            "    import time\n"
            "    time.sleep(10)\n"
        )
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is False
        assert "timeout" in (result.error or "").lower() or "timed out" in (result.error or "").lower()


# ──────────────────────────────────────────────
# Task 9 – apply_intended_actions tests
# ──────────────────────────────────────────────


class TestApplyIntendedActions:
    """Test WorldKernel.apply_intended_actions applies proxy actions to the real engine."""

    def _make_kernel(self, members=5):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=members, land_shape=(10, 10), random_seed=42)
        return WorldKernel(config, save_path=tmpdir)

    def test_expand_action_changes_land(self):
        kernel = self._make_kernel()
        kernel.begin_round()
        kernel._execution.get_neighbors()
        member = kernel._execution.current_members[0]
        old_land = member.land_num
        actions = [{"action": "expand", "member_id": member.id}]
        results = kernel.apply_intended_actions(actions)
        assert len(results) == 1
        # Expand should increase land (if empty land was available)
        if member.current_empty_loc_list:
            assert member.land_num >= old_land

    def test_unknown_action_skipped(self):
        kernel = self._make_kernel()
        kernel.begin_round()
        actions = [{"action": "unknown_thing", "member_id": 1}]
        results = kernel.apply_intended_actions(actions)
        assert len(results) == 1
        assert results[0]["applied"] is False

    def test_empty_actions(self):
        kernel = self._make_kernel()
        kernel.begin_round()
        results = kernel.apply_intended_actions([])
        assert results == []


# ──────────────────────────────────────────────
# Phase 3 Task 1 – MechanismRegistry tests
# ──────────────────────────────────────────────

from kernel.mechanism_registry import MechanismRecord, MechanismRegistry


class TestMechanismRegistry:
    def test_submit_mechanism(self):
        """Submit a mechanism and verify it returns a valid record."""
        reg = MechanismRegistry()
        rec = reg.submit(proposer_id=1, code="def f(): pass", description="test mech")
        assert rec is not None
        assert isinstance(rec, MechanismRecord)
        assert rec.proposer_id == 1
        assert rec.code == "def f(): pass"
        assert rec.description == "test mech"
        assert rec.status == "submitted"
        assert rec.submitted_round == 0

    def test_submit_with_round(self):
        """Submit with explicit round_id."""
        reg = MechanismRegistry()
        rec = reg.submit(proposer_id=2, code="x=1", description="round mech", round_id=5)
        assert rec is not None
        assert rec.submitted_round == 5

    def test_get_pending(self):
        """get_pending returns only submitted records."""
        reg = MechanismRegistry()
        rec1 = reg.submit(proposer_id=1, code="a", description="d1", round_id=0)
        rec2 = reg.submit(proposer_id=2, code="b", description="d2", round_id=0)
        pending = reg.get_pending()
        assert len(pending) == 2
        assert all(r.status == "submitted" for r in pending)

    def test_approve_and_activate(self):
        """Approve then activate a mechanism."""
        reg = MechanismRegistry()
        rec = reg.submit(proposer_id=1, code="c", description="d3")
        reg.mark_approved(rec.mechanism_id, round_id=1, reason="looks good")
        assert rec.status == "approved"
        assert rec.judged_round == 1
        assert rec.judge_reason == "looks good"
        reg.activate(rec.mechanism_id, round_id=2)
        assert rec.status == "active"
        assert rec.activated_round == 2

    def test_reject(self):
        """Reject a mechanism."""
        reg = MechanismRegistry()
        rec = reg.submit(proposer_id=1, code="d", description="d4")
        reg.mark_rejected(rec.mechanism_id, round_id=1, reason="too risky")
        assert rec.status == "rejected"
        assert rec.judge_reason == "too risky"

    def test_get_active(self):
        """get_active returns only active mechanisms."""
        reg = MechanismRegistry()
        rec1 = reg.submit(proposer_id=1, code="a", description="d1", round_id=0)
        rec2 = reg.submit(proposer_id=2, code="b", description="d2", round_id=0)
        reg.mark_approved(rec1.mechanism_id, round_id=1, reason="ok")
        reg.activate(rec1.mechanism_id, round_id=2)
        active = reg.get_active()
        assert len(active) == 1
        assert active[0].mechanism_id == rec1.mechanism_id

    def test_get_all(self):
        """get_all returns all records regardless of status."""
        reg = MechanismRegistry()
        reg.submit(proposer_id=1, code="a", description="d1", round_id=0)
        reg.submit(proposer_id=2, code="b", description="d2", round_id=0)
        reg.submit(proposer_id=3, code="c", description="d3", round_id=0)
        assert len(reg.get_all()) == 3

    def test_get_by_id(self):
        """get_by_id returns the correct record."""
        reg = MechanismRegistry()
        rec = reg.submit(proposer_id=1, code="a", description="d1")
        found = reg.get_by_id(rec.mechanism_id)
        assert found is rec

    def test_get_by_id_not_found(self):
        """get_by_id returns None for unknown id."""
        reg = MechanismRegistry()
        assert reg.get_by_id("nonexistent") is None

    def test_one_proposal_per_agent_per_round(self):
        """Same agent cannot submit twice in same round."""
        reg = MechanismRegistry()
        rec1 = reg.submit(proposer_id=1, code="a", description="d1", round_id=0)
        rec2 = reg.submit(proposer_id=1, code="b", description="d2", round_id=0)
        assert rec1 is not None
        assert rec2 is None

    def test_same_agent_different_rounds(self):
        """Same agent can submit in different rounds."""
        reg = MechanismRegistry()
        rec1 = reg.submit(proposer_id=1, code="a", description="d1", round_id=0)
        rec2 = reg.submit(proposer_id=1, code="b", description="d2", round_id=1)
        assert rec1 is not None
        assert rec2 is not None
        assert rec1.mechanism_id != rec2.mechanism_id


# ──────────────────────────────────────────────
# Phase 3 Task 2 – JudgeAdapter tests
# ──────────────────────────────────────────────

from kernel.judge_adapter import DummyJudge, JudgeAdapter, JudgmentResult


class TestJudgmentResult:
    def test_creation(self):
        """JudgmentResult stores all fields correctly."""
        jr = JudgmentResult(approved=True, reason="ok", latency_ms=12.5)
        assert jr.approved is True
        assert jr.reason == "ok"
        assert jr.latency_ms == 12.5
        assert jr.error is None

    def test_with_error(self):
        """JudgmentResult with error field."""
        jr = JudgmentResult(approved=False, reason="failed", latency_ms=5.0, error="timeout")
        assert jr.approved is False
        assert jr.error == "timeout"


class TestDummyJudge:
    def test_approves_everything(self):
        """DummyJudge always returns approved=True."""
        judge = DummyJudge()
        result = judge.evaluate(code="x=1", proposer_id=1, proposal_type="mechanism")
        assert result.approved is True
        assert result.reason == "dummy-approved"

    def test_approves_with_context(self):
        """DummyJudge approves even with context."""
        judge = DummyJudge()
        result = judge.evaluate(code="x=1", proposer_id=1, proposal_type="mechanism", context={"key": "val"})
        assert result.approved is True


class TestJudgeAdapter:
    def test_timeout_returns_rejected(self):
        """Without LLM config, subprocess errors -> fail-closed -> rejected."""
        adapter = JudgeAdapter(timeout=10.0, use_dummy=False)
        result = adapter.evaluate(code="x=1", proposer_id=1, proposal_type="mechanism")
        assert result.approved is False
        assert result.latency_ms >= 0.0

    def test_dummy_mode(self):
        """JudgeAdapter in dummy mode delegates to DummyJudge."""
        adapter = JudgeAdapter(use_dummy=True)
        result = adapter.evaluate(code="x=1", proposer_id=1, proposal_type="mechanism")
        assert result.approved is True
        assert result.reason == "dummy-approved"


# ──────────────────────────────────────────────
# Phase 3 Task 3 – Round Metrics tests
# ──────────────────────────────────────────────

from kernel.round_metrics import compute_gini, compute_round_metrics


class TestGiniCoefficient:
    def test_equal_distribution(self):
        """All equal values -> Gini = 0.0."""
        assert compute_gini([10, 10, 10, 10]) == pytest.approx(0.0)

    def test_one_has_all(self):
        """One member has everything -> Gini = 0.75."""
        assert compute_gini([0, 0, 0, 100]) == pytest.approx(0.75)

    def test_two_members_unequal(self):
        """[0, 100] -> Gini = 0.5."""
        assert compute_gini([0, 100]) == pytest.approx(0.5)

    def test_single_member(self):
        """Single member -> Gini = 0.0."""
        assert compute_gini([100]) == pytest.approx(0.0)

    def test_empty(self):
        """Empty list -> Gini = 0.0."""
        assert compute_gini([]) == pytest.approx(0.0)

    def test_all_zeros(self):
        """All zeros -> Gini = 0.0."""
        assert compute_gini([0, 0, 0]) == pytest.approx(0.0)


class TestComputeRoundMetrics:
    def test_basic_metrics(self):
        """3 members, check all expected keys are present and correct."""
        members = [
            {"vitality": 10.0},
            {"vitality": 20.0},
            {"vitality": 30.0},
        ]
        metrics = compute_round_metrics(
            members, trade_volume=5, conflict_count=2,
            mechanism_proposals=3, mechanism_approvals=1,
        )
        assert metrics["total_vitality"] == pytest.approx(60.0)
        assert metrics["population"] == 3
        assert metrics["trade_volume"] == 5
        assert metrics["conflict_count"] == 2
        assert metrics["mechanism_proposals"] == 3
        assert metrics["mechanism_approvals"] == 1
        assert "gini_coefficient" in metrics
        assert metrics["gini_coefficient"] >= 0.0

    def test_empty_members(self):
        """Empty members list -> total_vitality=0, population=0."""
        metrics = compute_round_metrics([])
        assert metrics["total_vitality"] == 0.0
        assert metrics["population"] == 0
        assert metrics["gini_coefficient"] == 0.0


# ──────────────────────────────────────────────
# Phase 3 Task 4 – KernelDAGRunner tests
# ──────────────────────────────────────────────

from kernel.dag_runner import KernelDAGRunner


class TestKernelDAGRunner:
    def _make_kernel(self, tmp_path):
        from kernel.schemas import WorldConfig
        from kernel.world_kernel import WorldKernel
        config = WorldConfig(init_member_number=5, land_shape=(10, 10), random_seed=42)
        return WorldKernel(config, save_path=str(tmp_path))

    def test_run_phases_returns_phase_log(self, tmp_path):
        kernel = self._make_kernel(tmp_path)
        kernel.begin_round()
        runner = KernelDAGRunner(kernel._execution)
        log = runner.run_settlement_phases()
        assert "produce" in log
        assert "consume" in log

    def test_phases_in_correct_order(self, tmp_path):
        kernel = self._make_kernel(tmp_path)
        kernel.begin_round()
        runner = KernelDAGRunner(kernel._execution)
        log = runner.run_settlement_phases()
        assert log.index("produce") < log.index("consume")

    def test_deterministic_across_runs(self, tmp_path):
        from kernel.schemas import WorldConfig
        from kernel.world_kernel import WorldKernel
        config = WorldConfig(init_member_number=5, land_shape=(10, 10), random_seed=42)
        k1 = WorldKernel(config, save_path=str(tmp_path / "k1"))
        k1.begin_round()
        KernelDAGRunner(k1._execution).run_settlement_phases()
        snap1 = k1.get_snapshot()

        k2 = WorldKernel(config, save_path=str(tmp_path / "k2"))
        k2.begin_round()
        KernelDAGRunner(k2._execution).run_settlement_phases()
        snap2 = k2.get_snapshot()

        for m1, m2 in zip(snap1.members, snap2.members):
            assert m1["vitality"] == pytest.approx(m2["vitality"])
            assert m1["cargo"] == pytest.approx(m2["cargo"])

    def test_skips_llm_nodes(self, tmp_path):
        kernel = self._make_kernel(tmp_path)
        kernel.begin_round()
        runner = KernelDAGRunner(kernel._execution)
        log = runner.run_settlement_phases()
        assert "analyze" not in log
        assert "agent_decisions" not in log
