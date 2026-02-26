"""Tests for tech debt fixes (Phase 5: Production Hardening)."""

import platform
import sys
import threading
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from kernel.schemas import ActionIntent, WorldConfig
from kernel.world_kernel import WorldKernel
from kernel.subprocess_sandbox import SubprocessSandbox
from kernel.event_log import EventLog
from kernel.store import Store
from api.models import EventEnvelope
from scripts.run_server import build_app


# ── Helpers ──────────────────────────────────────


def _make_app(**kwargs):
    defaults = dict(members=5, land_w=10, land_h=10, seed=42)
    defaults.update(kwargs)
    return build_app(**defaults)


def _register_agent(client, name="Bot"):
    resp = client.post("/v1/agents/register", json={"name": name})
    assert resp.status_code == 200
    return resp.json()


def _open_submissions(app, pace=60.0):
    kernel = app.state.leviathan["kernel"]
    rs = app.state.leviathan["round_state"]
    if kernel.round_id == 0:
        kernel.begin_round()
    rs.open_submissions(round_id=kernel.round_id, pace=pace)


# ── Task 1: Idempotency key scoping ─────────────


class TestIdempotencyKeyScoping:
    def test_idempotency_key_scoped_to_agent(self, tmp_path):
        """Two agents using the same idempotency_key should get independent results."""
        config = WorldConfig(init_member_number=5, land_shape=(5, 5), random_seed=42)
        kernel = WorldKernel(config, save_path=str(tmp_path))
        kernel.begin_round()

        # Both agents use the same key but different agent_ids
        noop_code = "def agent_action(engine, member_idx): pass"
        action_a = ActionIntent(agent_id=0, round_id=1, code=noop_code, idempotency_key="same-key")
        action_b = ActionIntent(agent_id=1, round_id=1, code=noop_code, idempotency_key="same-key")

        results = kernel.accept_actions([action_a, action_b])
        assert len(results) == 2
        # Both should succeed independently (not return cached from first)
        assert results[0].agent_id == 0
        assert results[1].agent_id == 1
        assert results[0].success is True
        assert results[1].success is True


# ── Task 2: Health check ────────────────────────


class TestHealthCheck:
    def test_health_check_degraded(self):
        """Health returns degraded when sim thread is dead."""
        app = _make_app()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        app.state.sim_thread = mock_thread

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert "simulation thread stopped" in data["reason"]

    def test_health_check_ok_default(self):
        """Health returns ok when no sim_thread is set."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── Task 5: Sandbox builtin restrictions ────────


class TestSandboxBuiltinRestrictions:
    def test_sandbox_blocks_open(self):
        """Code using open() should fail."""
        sandbox = SubprocessSandbox(timeout=10)
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=42)
        kernel = WorldKernel(config, save_path="/tmp/test_sandbox_open")
        kernel.begin_round()
        from kernel.execution_sandbox import SandboxContext
        ctx = SandboxContext(execution_engine=kernel._execution, member_index=0)
        code = "def agent_action(engine, idx):\n    open('/tmp/test_sandbox_probe', 'w')"
        result = sandbox.execute_agent_code(code, ctx)
        assert not result.success
        assert "open" in (result.error or "").lower() or "name" in (result.error or "").lower()

    def test_sandbox_blocks_import(self):
        """Code using __import__ should fail."""
        sandbox = SubprocessSandbox(timeout=10)
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=42)
        kernel = WorldKernel(config, save_path="/tmp/test_sandbox_import")
        kernel.begin_round()
        from kernel.execution_sandbox import SandboxContext
        ctx = SandboxContext(execution_engine=kernel._execution, member_index=0)
        code = "def agent_action(engine, idx):\n    __import__('os')"
        result = sandbox.execute_agent_code(code, ctx)
        assert not result.success

    def test_sandbox_allows_safe_builtins(self):
        """Safe builtins (abs, min, max, round, len, range, isinstance) should work."""
        sandbox = SubprocessSandbox(timeout=10)
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=42)
        kernel = WorldKernel(config, save_path="/tmp/test_sandbox_safe")
        kernel.begin_round()
        from kernel.execution_sandbox import SandboxContext
        ctx = SandboxContext(execution_engine=kernel._execution, member_index=0)
        code = (
            "def agent_action(engine, idx):\n"
            "    assert abs(-1) == 1\n"
            "    assert min(1, 2) == 1\n"
            "    assert max(1, 2) == 2\n"
            "    assert round(1.5) == 2\n"
            "    assert len([1, 2]) == 2\n"
            "    assert list(range(3)) == [0, 1, 2]\n"
            "    assert isinstance(1, int)\n"
        )
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success, f"Safe builtins failed: {result.error}"


# ── Task 6: Resource limits ─────────────────────


class TestResourceLimits:
    def test_resource_limits_callable(self):
        """_get_preexec_fn() returns a callable on Unix."""
        sandbox = SubprocessSandbox()
        fn = sandbox._get_preexec_fn()
        if platform.system() != "Windows":
            assert fn is not None
            assert callable(fn)
        else:
            # On Windows, resource module isn't available
            pass


# ── Task 7: Code size limits ────────────────────


class TestCodeSizeLimits:
    def test_code_size_limit_action(self):
        """Action code exceeding 10KB should be rejected with 400."""
        app = _make_app()
        client = TestClient(app)
        agent = _register_agent(client, "BigCodeBot")
        _open_submissions(app)

        big_code = "x = 1\n" * 5000  # > 10KB
        resp = client.post(
            "/v1/world/actions",
            json={"code": big_code, "idempotency_key": "big-1"},
            headers={"X-API-Key": agent["api_key"]},
        )
        assert resp.status_code == 400
        assert "10000" in resp.json()["detail"] or "limit" in resp.json()["detail"].lower()

    def test_code_size_limit_mechanism(self):
        """Mechanism code exceeding 10KB should be rejected with 400."""
        app = _make_app()
        client = TestClient(app)
        agent = _register_agent(client, "BigMechBot")
        _open_submissions(app)

        big_code = "x = 1\n" * 5000  # > 10KB
        resp = client.post(
            "/v1/world/mechanisms/propose",
            json={"code": big_code, "description": "test", "idempotency_key": "big-m-1"},
            headers={"X-API-Key": agent["api_key"]},
        )
        assert resp.status_code == 400
        assert "10000" in resp.json()["detail"] or "limit" in resp.json()["detail"].lower()


# ── Task 8: Event log eviction ──────────────────


class TestEventLogEviction:
    def test_event_log_eviction(self):
        """After appending 600 events with a store, len=600 but _mem <= 500."""
        store = Store(":memory:")
        log = EventLog(store=store)

        for i in range(600):
            log.append(EventEnvelope(
                event_id=0,
                event_type="round_settled",
                round_id=i + 1,
                timestamp=f"2026-01-01T00:00:{i:02d}Z",
                payload={"round": i + 1},
            ))

        assert len(log) == 600
        assert len(log._mem) <= EventLog._MAX_CACHE


# ── Task 4: Better rejection messages ───────────


class TestBetterRejectionMessages:
    def test_action_rejection_includes_state(self):
        """Rejected action response includes round state info."""
        app = _make_app()
        client = TestClient(app)
        agent = _register_agent(client, "RejectBot")

        # Don't open submissions — round stays in "settled" state
        kernel = app.state.leviathan["kernel"]
        if kernel.round_id == 0:
            kernel.begin_round()
        # Round state is "settled" (not "accepting")

        resp = client.post(
            "/v1/world/actions",
            json={"code": "x = 1", "idempotency_key": "rej-1"},
            headers={"X-API-Key": agent["api_key"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rejected"
        assert "state=" in data["reason"]
