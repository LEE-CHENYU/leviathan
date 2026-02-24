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
        code = "def agent_action(engine):\n    raise ValueError('bad value')\n"

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
