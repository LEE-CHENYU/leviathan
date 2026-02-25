"""Tests for the Phase 1 Read API (api package)."""

import dataclasses
import tempfile
from typing import Any, Dict, List

import pytest

# ──────────────────────────────────────────────
# Task 1 – Pydantic response model tests
# ──────────────────────────────────────────────

from api.models import (
    AgentDiscovery,
    EventEnvelope,
    RoundInfo,
    RoundReceiptResponse,
    WorldInfo,
)


class TestModels:
    def test_world_info_creation(self):
        info = WorldInfo(
            world_id="w-123",
            round_id=5,
            member_count=10,
            state_hash="ab" * 32,
        )
        assert info.world_id == "w-123"
        assert info.round_id == 5
        assert info.member_count == 10
        assert info.state_hash == "ab" * 32

    def test_round_receipt_response(self):
        receipt = RoundReceiptResponse(
            round_id=1,
            seed=42,
            snapshot_hash_before="aa" * 32,
            snapshot_hash_after="bb" * 32,
            accepted_action_ids=["a1", "a2"],
            rejected_action_ids=["r1"],
            activated_mechanism_ids=["m1"],
            judge_results=[{"agent": 1, "score": 0.9}],
            round_metrics={"avg_perf": 0.85},
            timestamp="2025-01-01T00:00:00Z",
        )
        assert receipt.round_id == 1
        assert receipt.seed == 42
        assert len(receipt.accepted_action_ids) == 2
        assert receipt.round_metrics["avg_perf"] == 0.85

    def test_round_info_no_receipt(self):
        info = RoundInfo(round_id=3)
        assert info.round_id == 3
        assert info.last_receipt is None

    def test_event_envelope(self):
        event = EventEnvelope(
            event_id=1,
            event_type="round_settled",
            round_id=5,
            timestamp="2025-06-15T12:00:00Z",
            payload={"detail": "something happened"},
        )
        assert event.event_id == 1
        assert event.event_type == "round_settled"
        assert event.payload["detail"] == "something happened"

    def test_agent_discovery(self):
        discovery = AgentDiscovery(
            name="leviathan",
            version="0.1.0",
            api_version="v1",
            capabilities=["read_world", "read_rounds"],
            endpoints={"/health": "GET", "/v1/world": "GET"},
        )
        assert discovery.name == "leviathan"
        assert discovery.version == "0.1.0"
        assert len(discovery.capabilities) == 2
        assert discovery.endpoints["/health"] == "GET"


# ──────────────────────────────────────────────
# Task 2 – Dependency injection tests
# ──────────────────────────────────────────────

from api.deps import create_app_state, get_event_log, get_kernel
from kernel.schemas import WorldConfig
from kernel.world_kernel import WorldKernel


class TestDeps:
    def test_create_app_state(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        state = create_app_state(kernel)
        assert "kernel" in state
        assert "event_log" in state
        assert state["kernel"] is kernel
        assert state["event_log"] == []

    def test_get_kernel_returns_kernel(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        state = create_app_state(kernel)
        assert get_kernel(state) is kernel

    def test_get_event_log_returns_list(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        state = create_app_state(kernel)
        log = get_event_log(state)
        assert isinstance(log, list)
        assert len(log) == 0


# ──────────────────────────────────────────────
# Task 3 – FastAPI app factory and /health tests
# ──────────────────────────────────────────────

from starlette.testclient import TestClient

from api.app import create_app


def _make_test_client(members=5, seed=42):
    """Create a TestClient backed by a fresh WorldKernel.

    Reusable helper for all endpoint tests in later tasks.
    """
    tmpdir = tempfile.mkdtemp()
    config = WorldConfig(init_member_number=members, land_shape=(10, 10), random_seed=seed)
    kernel = WorldKernel(config, save_path=tmpdir)
    app = create_app(kernel)
    return TestClient(app), kernel


class TestHealthEndpoint:
    def test_health(self):
        client, _ = _make_test_client()
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ──────────────────────────────────────────────
# Task 4 – World info and snapshot endpoints
# ──────────────────────────────────────────────


class TestWorldEndpoints:
    def test_world_info(self):
        client, kernel = _make_test_client()
        resp = client.get("/v1/world")
        assert resp.status_code == 200
        data = resp.json()
        assert data["round_id"] == 0
        assert data["member_count"] == 5
        assert len(data["state_hash"]) == 64

    def test_snapshot(self):
        client, kernel = _make_test_client()
        resp = client.get("/v1/world/snapshot")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["members"]) == 5
        assert "land" in data
        assert len(data["state_hash"]) == 64


# ──────────────────────────────────────────────
# Task 5 – Round info and round-by-id endpoints
# ──────────────────────────────────────────────


def _settle_rounds(kernel, app, n=1):
    """Run n rounds and append events to the app's event log."""
    event_log = app.state.leviathan["event_log"]
    for i in range(n):
        kernel.begin_round()
        receipt = kernel.settle_round(seed=kernel.round_id)
        event_log.append(EventEnvelope(
            event_id=len(event_log) + 1,
            event_type="round_settled",
            round_id=receipt.round_id,
            timestamp=receipt.timestamp,
            payload=dataclasses.asdict(receipt),
        ))


class TestRoundEndpoints:
    def test_rounds_current_before_any_round(self):
        client, kernel = _make_test_client()
        resp = client.get("/v1/world/rounds/current")
        assert resp.status_code == 200
        data = resp.json()
        assert data["round_id"] == 0
        assert data["last_receipt"] is None

    def test_rounds_current_after_settle(self):
        client, kernel = _make_test_client()
        _settle_rounds(kernel, client.app, n=1)
        resp = client.get("/v1/world/rounds/current")
        assert resp.status_code == 200
        data = resp.json()
        assert data["round_id"] == 1
        assert data["last_receipt"] is not None
        assert data["last_receipt"]["round_id"] == 1

    def test_round_by_id(self):
        client, kernel = _make_test_client()
        _settle_rounds(kernel, client.app, n=2)
        resp = client.get("/v1/world/rounds/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["round_id"] == 1

    def test_round_not_found(self):
        client, kernel = _make_test_client()
        resp = client.get("/v1/world/rounds/999")
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# Task 6 – Events polling endpoint
# ──────────────────────────────────────────────


class TestEventsEndpoint:
    def test_events_polling(self):
        client, kernel = _make_test_client()
        _settle_rounds(kernel, client.app, n=3)
        resp = client.get("/v1/world/events", params={"since_round": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["round_id"] == 2
        assert data[1]["round_id"] == 3

    def test_events_all(self):
        client, kernel = _make_test_client()
        _settle_rounds(kernel, client.app, n=2)
        resp = client.get("/v1/world/events")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_events_empty(self):
        client, kernel = _make_test_client()
        _settle_rounds(kernel, client.app, n=1)
        resp = client.get("/v1/world/events", params={"since_round": 999})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 0

    def test_events_monotonic_ids(self):
        client, kernel = _make_test_client()
        _settle_rounds(kernel, client.app, n=3)
        resp = client.get("/v1/world/events")
        data = resp.json()
        event_ids = [e["event_id"] for e in data]
        assert len(event_ids) == 3
        assert event_ids == sorted(event_ids)
        assert len(set(event_ids)) == 3


# ──────────────────────────────────────────────
# Task 7 – Agent discovery endpoint
# ──────────────────────────────────────────────


class TestDiscovery:
    def test_discovery_manifest(self):
        client, kernel = _make_test_client()
        resp = client.get("/.well-known/leviathan-agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Leviathan"
        assert data["version"] == "0.1.0"
        assert data["api_version"] == "v1"
        assert "read_snapshot" in data["capabilities"]
        assert "read_events" in data["capabilities"]
        assert "read_receipts" in data["capabilities"]
        assert data["endpoints"]["base"] == "/v1/world"


# ──────────────────────────────────────────────
# Task 8 – Server runner integration test
# ──────────────────────────────────────────────


class TestServerIntegration:
    def test_build_app_creates_working_api(self):
        from scripts.run_server import build_app

        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        resp = client.get("/v1/world")
        assert resp.status_code == 200
        assert resp.json()["member_count"] == 5
        resp = client.get("/.well-known/leviathan-agent.json")
        assert resp.status_code == 200


# ──────────────────────────────────────────────
# A6 – API key auth and rate limiting tests
# ──────────────────────────────────────────────

from fastapi import Depends, Request

from api.auth import APIKeyAuth
from api.deps import get_auth


def _add_protected_route(app):
    """Add a protected test route that invokes the APIKeyAuth dependency."""

    @app.get("/v1/world/protected-test")
    def protected_test(request: Request, auth: APIKeyAuth = Depends(get_auth)):
        # Invoke the auth check manually since get_auth only retrieves
        # the instance — actual validation happens when we call it.
        auth(request)
        return {"protected": True}


class TestAuth:
    """Auth and rate-limiting tests with default (open) configuration."""

    def test_no_auth_by_default(self):
        """When no API keys configured, all endpoints are open."""
        client, _ = _make_test_client()
        assert client.get("/v1/world").status_code == 200

    def test_rate_limiter_allows_normal_traffic(self):
        """Rate limiter doesn't block under normal load."""
        client, _ = _make_test_client()
        for _ in range(10):
            assert client.get("/health").status_code == 200

    def test_rate_limiter_blocks_when_exhausted(self):
        """Rate limiter returns 429 after exceeding the limit."""
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(10, 10), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        app = create_app(kernel, rate_limit=3)
        client = TestClient(app)

        statuses = []
        for _ in range(6):
            statuses.append(client.get("/health").status_code)

        assert 429 in statuses
        assert statuses.count(200) >= 3


class TestAuthEnabled:
    """Auth tests with API key validation enabled."""

    def _make_app_with_auth(self):
        """Create an app with auth enabled and a protected route."""
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(10, 10), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        app = create_app(kernel, api_keys={"test-key-123", "backup-key"})
        _add_protected_route(app)
        return TestClient(app)

    def test_unprotected_routes_remain_open(self):
        """Read-only routes work without auth even when keys are configured."""
        client = self._make_app_with_auth()
        assert client.get("/health").status_code == 200
        assert client.get("/v1/world").status_code == 200

    def test_auth_rejects_missing_key(self):
        """When auth is enabled, missing key returns 401."""
        client = self._make_app_with_auth()
        resp = client.get("/v1/world/protected-test")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Missing API key"

    def test_auth_rejects_invalid_key(self):
        """When auth is enabled, invalid key returns 403."""
        client = self._make_app_with_auth()
        resp = client.get(
            "/v1/world/protected-test",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Invalid API key"

    def test_auth_accepts_valid_header_key(self):
        """Valid X-API-Key header passes auth."""
        client = self._make_app_with_auth()
        resp = client.get(
            "/v1/world/protected-test",
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200
        assert resp.json()["protected"] is True

    def test_auth_accepts_valid_query_key(self):
        """Valid api_key query param passes auth."""
        client = self._make_app_with_auth()
        resp = client.get(
            "/v1/world/protected-test",
            params={"api_key": "backup-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["protected"] is True


# ──────────────────────────────────────────────
# Phase 2 — Agent registry tests
# ──────────────────────────────────────────────

from api.registry import AgentRecord, AgentRegistry


class TestAgentRegistry:
    def test_register_agent(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        registry = AgentRegistry()
        record = registry.register("TestBot", "A test agent", kernel)
        assert isinstance(record, AgentRecord)
        assert record.name == "TestBot"
        assert record.api_key.startswith("lev_")
        assert record.member_id in [m.id for m in kernel._execution.current_members]

    def test_register_assigns_different_members(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        registry = AgentRegistry()
        r1 = registry.register("Bot1", "", kernel)
        r2 = registry.register("Bot2", "", kernel)
        assert r1.member_id != r2.member_id
        assert r1.api_key != r2.api_key
        assert r1.agent_id != r2.agent_id

    def test_register_too_many_returns_none(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=2, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        registry = AgentRegistry()
        registry.register("Bot1", "", kernel)
        registry.register("Bot2", "", kernel)
        result = registry.register("Bot3", "", kernel)
        assert result is None

    def test_get_by_api_key(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        registry = AgentRegistry()
        record = registry.register("TestBot", "", kernel)
        found = registry.get_by_api_key(record.api_key)
        assert found is not None
        assert found.agent_id == record.agent_id

    def test_get_by_api_key_missing(self):
        registry = AgentRegistry()
        assert registry.get_by_api_key("lev_nonexistent") is None

    def test_get_by_agent_id(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        registry = AgentRegistry()
        record = registry.register("TestBot", "", kernel)
        found = registry.get_by_agent_id(record.agent_id)
        assert found is not None
        assert found.name == "TestBot"


# ──────────────────────────────────────────────
# Phase 2 — RoundState tests
# ──────────────────────────────────────────────

from api.round_state import PendingAction, RoundState


class TestRoundState:
    def test_initial_state(self):
        rs = RoundState()
        assert rs.state == "settled"
        assert rs.round_id == 0
        assert rs.deadline is None
        assert rs.get_pending_actions() == []

    def test_open_submissions(self):
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=5.0)
        assert rs.state == "accepting"
        assert rs.round_id == 1
        assert rs.deadline is not None
        assert rs.seconds_remaining() > 0

    def test_submit_action_during_accepting(self):
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=5.0)
        pa = PendingAction(agent_id=1, member_id=10, code="x=1", idempotency_key="k1")
        accepted = rs.submit_action(pa)
        assert accepted is True
        assert len(rs.get_pending_actions()) == 1

    def test_submit_action_rejected_when_not_accepting(self):
        rs = RoundState()
        pa = PendingAction(agent_id=1, member_id=10, code="x=1", idempotency_key="k1")
        accepted = rs.submit_action(pa)
        assert accepted is False

    def test_close_and_drain(self):
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=5.0)
        pa = PendingAction(agent_id=1, member_id=10, code="x=1", idempotency_key="k1")
        rs.submit_action(pa)
        rs.close_submissions()
        assert rs.state == "executing"
        drained = rs.drain_actions()
        assert len(drained) == 1
        assert rs.get_pending_actions() == []

    def test_mark_settled(self):
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=5.0)
        rs.close_submissions()
        rs.mark_settled()
        assert rs.state == "settled"

    def test_idempotency_duplicate_key(self):
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=5.0)
        pa1 = PendingAction(agent_id=1, member_id=10, code="x=1", idempotency_key="k1")
        pa2 = PendingAction(agent_id=1, member_id=10, code="x=2", idempotency_key="k1")
        rs.submit_action(pa1)
        accepted = rs.submit_action(pa2)
        assert accepted is True
        assert len(rs.get_pending_actions()) == 1


# ──────────────────────────────────────────────
# Phase 2 — New models tests
# ──────────────────────────────────────────────

from api.models import (
    AgentRegisterRequest,
    AgentRegisterResponse,
    AgentProfileResponse,
    ActionSubmitRequest,
    ActionSubmitResponse,
    DeadlineResponse,
)


class TestPhase2Models:
    def test_register_request(self):
        req = AgentRegisterRequest(name="Bot1", description="test bot")
        assert req.name == "Bot1"

    def test_register_response(self):
        resp = AgentRegisterResponse(agent_id=1, api_key="lev_abc", member_id=5)
        assert resp.agent_id == 1
        assert resp.api_key == "lev_abc"

    def test_agent_profile(self):
        resp = AgentProfileResponse(
            agent_id=1, name="Bot1", member_id=5, registered_at="2025-01-01T00:00:00Z"
        )
        assert resp.agent_id == 1

    def test_action_submit_request(self):
        req = ActionSubmitRequest(
            code="def agent_action(e, m): pass", idempotency_key="r1-a1"
        )
        assert req.code.startswith("def")

    def test_action_submit_response_accepted(self):
        resp = ActionSubmitResponse(status="accepted", round_id=5)
        assert resp.status == "accepted"

    def test_deadline_response(self):
        resp = DeadlineResponse(
            round_id=3, state="accepting", deadline_utc="2025-01-01T00:00:00Z",
            seconds_remaining=4.5,
        )
        assert resp.seconds_remaining == 4.5


# ──────────────────────────────────────────────
# Phase 2 — App wiring tests
# ──────────────────────────────────────────────

from api.registry import AgentRegistry as AgentRegistryWiring
from api.round_state import RoundState as RoundStateWiring


class TestPhase2AppWiring:
    def test_app_state_has_registry(self):
        client, kernel = _make_test_client()
        state = client.app.state.leviathan
        assert "registry" in state
        assert isinstance(state["registry"], AgentRegistryWiring)

    def test_app_state_has_round_state(self):
        client, kernel = _make_test_client()
        state = client.app.state.leviathan
        assert "round_state" in state
        assert isinstance(state["round_state"], RoundStateWiring)


# ──────────────────────────────────────────────
# Phase 2 — Agent endpoints tests
# ──────────────────────────────────────────────


class TestAgentEndpoints:
    def test_register_agent(self):
        client, kernel = _make_test_client(members=5)
        resp = client.post(
            "/v1/agents/register",
            json={"name": "TestBot", "description": "A test agent"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_id" in data
        assert data["api_key"].startswith("lev_")
        assert "member_id" in data

    def test_register_too_many(self):
        client, kernel = _make_test_client(members=2)
        client.post("/v1/agents/register", json={"name": "Bot1"})
        client.post("/v1/agents/register", json={"name": "Bot2"})
        resp = client.post("/v1/agents/register", json={"name": "Bot3"})
        assert resp.status_code == 409

    def test_agent_me(self):
        client, kernel = _make_test_client(members=5)
        reg_resp = client.post("/v1/agents/register", json={"name": "TestBot"})
        api_key = reg_resp.json()["api_key"]
        resp = client.get("/v1/agents/me", headers={"X-API-Key": api_key})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "TestBot"
        assert "registered_at" in data

    def test_agent_me_unauthorized(self):
        client, kernel = _make_test_client(members=5)
        resp = client.get("/v1/agents/me")
        assert resp.status_code == 401

    def test_agent_me_invalid_key(self):
        client, kernel = _make_test_client(members=5)
        resp = client.get("/v1/agents/me", headers={"X-API-Key": "lev_invalid"})
        assert resp.status_code == 403


# ──────────────────────────────────────────────
# Phase 2 — Action submission + deadline tests
# ──────────────────────────────────────────────


class TestActionEndpoints:
    def _register_and_open(self, client, members=5):
        """Register an agent and open submissions."""
        reg = client.post("/v1/agents/register", json={"name": "Bot"}).json()
        rs = client.app.state.leviathan["round_state"]
        rs.open_submissions(round_id=1, pace=30.0)
        return reg["api_key"], reg["member_id"]

    def test_submit_action_accepted(self):
        client, kernel = _make_test_client(members=5)
        api_key, member_id = self._register_and_open(client)
        resp = client.post(
            "/v1/world/actions",
            json={"code": "def agent_action(e, m): pass", "idempotency_key": "r1-a1"},
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["round_id"] == 1

    def test_submit_action_rejected_when_closed(self):
        client, kernel = _make_test_client(members=5)
        reg = client.post("/v1/agents/register", json={"name": "Bot"}).json()
        resp = client.post(
            "/v1/world/actions",
            json={"code": "def agent_action(e, m): pass", "idempotency_key": "k1"},
            headers={"X-API-Key": reg["api_key"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rejected"

    def test_submit_action_unauthorized(self):
        client, kernel = _make_test_client(members=5)
        resp = client.post(
            "/v1/world/actions",
            json={"code": "pass", "idempotency_key": "k1"},
        )
        assert resp.status_code == 401

    def test_submit_action_idempotency(self):
        client, kernel = _make_test_client(members=5)
        api_key, _ = self._register_and_open(client)
        body = {"code": "def agent_action(e, m): pass", "idempotency_key": "same-key"}
        resp1 = client.post("/v1/world/actions", json=body, headers={"X-API-Key": api_key})
        resp2 = client.post("/v1/world/actions", json=body, headers={"X-API-Key": api_key})
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        rs = client.app.state.leviathan["round_state"]
        assert len(rs.get_pending_actions()) == 1


class TestDeadlineEndpoint:
    def test_deadline_settled(self):
        client, kernel = _make_test_client()
        resp = client.get("/v1/world/rounds/current/deadline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "settled"
        assert data["seconds_remaining"] == 0.0

    def test_deadline_accepting(self):
        client, kernel = _make_test_client()
        rs = client.app.state.leviathan["round_state"]
        rs.open_submissions(round_id=1, pace=30.0)
        resp = client.get("/v1/world/rounds/current/deadline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "accepting"
        assert data["round_id"] == 1
        assert data["seconds_remaining"] > 0


# ──────────────────────────────────────────────
# Phase 2 — Simulation loop integration test
# ──────────────────────────────────────────────


class TestSimLoopIntegration:
    def test_full_external_agent_round(self):
        """End-to-end: register, submit action during accepting window, verify receipt."""
        from scripts.run_server import build_app

        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        client = TestClient(app)

        # Register agent
        reg = client.post("/v1/agents/register", json={"name": "E2EBot"}).json()
        api_key = reg["api_key"]

        # Manually run one round with submission window
        kernel = app.state.leviathan["kernel"]
        round_state = app.state.leviathan["round_state"]

        kernel.begin_round()
        round_state.open_submissions(round_id=kernel.round_id, pace=5.0)

        # Submit action
        resp = client.post(
            "/v1/world/actions",
            json={
                "code": "def agent_action(engine, member_id):\n    me = engine.current_members[member_id]\n    engine.expand(me)\n",
                "idempotency_key": "e2e-r1",
            },
            headers={"X-API-Key": api_key},
        )
        assert resp.json()["status"] == "accepted"

        # Close, execute, settle
        round_state.close_submissions()
        pending = round_state.drain_actions()
        assert len(pending) == 1

        # Execute through subprocess sandbox
        from kernel.subprocess_sandbox import SubprocessSandbox
        from kernel.execution_sandbox import SandboxContext

        sandbox = SubprocessSandbox()
        for pa in pending:
            member_index = kernel._resolve_agent_index(pa.member_id)
            if member_index is not None:
                ctx = SandboxContext(
                    execution_engine=kernel._execution,
                    member_index=member_index,
                )
                result = sandbox.execute_agent_code(pa.code, ctx)
                if result.success:
                    kernel.apply_intended_actions(result.intended_actions)

        receipt = kernel.settle_round(seed=kernel.round_id)
        round_state.mark_settled()

        assert receipt.round_id == 1
        assert round_state.state == "settled"


# ──────────────────────────────────────────────
# Phase 3 — Mechanism models + round state proposals
# ──────────────────────────────────────────────

from api.models import (
    MechanismProposeRequest,
    MechanismProposeResponse,
    MechanismResponse,
    MetricsResponse,
    JudgeStatsResponse,
)
from api.round_state import PendingProposal


class TestPhase3Models:
    def test_mechanism_propose_request(self):
        req = MechanismProposeRequest(
            code="def propose_modification(e): pass",
            description="test mechanism",
            idempotency_key="mech-1",
        )
        assert req.code.startswith("def ")
        assert req.idempotency_key == "mech-1"

    def test_mechanism_propose_response(self):
        resp = MechanismProposeResponse(mechanism_id="abc123", status="submitted")
        assert resp.mechanism_id == "abc123"

    def test_mechanism_response(self):
        resp = MechanismResponse(
            mechanism_id="abc", proposer_id=1, code="code", description="desc",
            status="active", submitted_round=1, judged_round=2,
            judge_reason="ok", activated_round=2,
        )
        assert resp.status == "active"

    def test_metrics_response(self):
        resp = MetricsResponse(
            round_id=1, total_vitality=100.0, gini_coefficient=0.3,
            trade_volume=5, conflict_count=2, mechanism_proposals=1,
            mechanism_approvals=1, population=10,
        )
        assert resp.population == 10

    def test_judge_stats_response(self):
        resp = JudgeStatsResponse(
            total_judgments=10, approved=7, rejected=3,
            approval_rate=0.7, recent_rejections=[],
        )
        assert resp.approval_rate == pytest.approx(0.7)


class TestPendingProposal:
    def test_creation(self):
        pp = PendingProposal(
            agent_id=1, member_id=5, code="def propose_modification(e): pass",
            description="test", idempotency_key="pk-1",
        )
        assert pp.agent_id == 1
        assert pp.member_id == 5

    def test_round_state_submit_proposal(self):
        from api.round_state import RoundState
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=10.0)
        pp = PendingProposal(
            agent_id=1, member_id=5, code="code", description="d", idempotency_key="pk-1"
        )
        assert rs.submit_proposal(pp) is True

    def test_round_state_drain_proposals(self):
        from api.round_state import RoundState
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=10.0)
        pp = PendingProposal(
            agent_id=1, member_id=5, code="code", description="d", idempotency_key="pk-1"
        )
        rs.submit_proposal(pp)
        proposals = rs.drain_proposals()
        assert len(proposals) == 1
        assert rs.drain_proposals() == []


# ──────────────────────────────────────────────
# Phase 3 — Mechanism endpoint tests
# ──────────────────────────────────────────────


class TestMechanismEndpoints:
    def _make_client(self):
        from scripts.run_server import build_app
        from starlette.testclient import TestClient
        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        return TestClient(app), app

    def _register_agent(self, client):
        resp = client.post("/v1/agents/register", json={"name": "Bot"})
        return resp.json()

    def test_propose_mechanism_accepted(self):
        client, app = self._make_client()
        agent = self._register_agent(client)
        rs = app.state.leviathan["round_state"]
        rs.open_submissions(round_id=1, pace=60.0)
        resp = client.post(
            "/v1/world/mechanisms/propose",
            json={"code": "def propose_modification(e): pass", "description": "test", "idempotency_key": "m-1"},
            headers={"X-API-Key": agent["api_key"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "submitted"
        assert data["mechanism_id"] != ""

    def test_propose_mechanism_rejected_outside_window(self):
        client, app = self._make_client()
        agent = self._register_agent(client)
        resp = client.post(
            "/v1/world/mechanisms/propose",
            json={"code": "code", "description": "test", "idempotency_key": "m-1"},
            headers={"X-API-Key": agent["api_key"]},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    def test_propose_mechanism_unauthorized(self):
        client, app = self._make_client()
        resp = client.post(
            "/v1/world/mechanisms/propose",
            json={"code": "code", "description": "test", "idempotency_key": "m-1"},
        )
        assert resp.status_code == 401

    def test_list_mechanisms(self):
        client, app = self._make_client()
        agent = self._register_agent(client)
        rs = app.state.leviathan["round_state"]
        rs.open_submissions(round_id=1, pace=60.0)
        client.post(
            "/v1/world/mechanisms/propose",
            json={"code": "code", "description": "d", "idempotency_key": "m-1"},
            headers={"X-API-Key": agent["api_key"]},
        )
        resp = client.get("/v1/world/mechanisms")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_list_mechanisms_with_status_filter(self):
        client, app = self._make_client()
        resp = client.get("/v1/world/mechanisms?status=active")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_mechanism_by_id(self):
        client, app = self._make_client()
        agent = self._register_agent(client)
        rs = app.state.leviathan["round_state"]
        rs.open_submissions(round_id=1, pace=60.0)
        resp = client.post(
            "/v1/world/mechanisms/propose",
            json={"code": "code", "description": "d", "idempotency_key": "m-1"},
            headers={"X-API-Key": agent["api_key"]},
        )
        mech_id = resp.json()["mechanism_id"]
        resp = client.get(f"/v1/world/mechanisms/{mech_id}")
        assert resp.status_code == 200
        assert resp.json()["mechanism_id"] == mech_id

    def test_get_mechanism_not_found(self):
        client, app = self._make_client()
        resp = client.get("/v1/world/mechanisms/nonexistent")
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# Phase 3 — Metrics and judge stats endpoint tests
# ──────────────────────────────────────────────


class TestMetricsEndpoints:
    def _make_client(self):
        from scripts.run_server import build_app
        from starlette.testclient import TestClient
        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        return TestClient(app), app

    def test_metrics_before_any_round(self):
        client, app = self._make_client()
        resp = client.get("/v1/world/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_vitality" in data
        assert "population" in data

    def test_metrics_after_settle(self):
        client, app = self._make_client()
        kernel = app.state.leviathan["kernel"]
        kernel.begin_round()
        kernel.settle_round(seed=1)
        resp = client.get("/v1/world/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["population"] > 0
        assert data["total_vitality"] > 0

    def test_metrics_history(self):
        client, app = self._make_client()
        kernel = app.state.leviathan["kernel"]
        event_log = app.state.leviathan["event_log"]
        import dataclasses
        from api.models import EventEnvelope
        for i in range(3):
            kernel.begin_round()
            receipt = kernel.settle_round(seed=i + 1)
            event_log.append(EventEnvelope(
                event_id=len(event_log) + 1, event_type="round_settled",
                round_id=receipt.round_id, timestamp=receipt.timestamp,
                payload=dataclasses.asdict(receipt),
            ))
        resp = client.get("/v1/world/metrics/history")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_metrics_history_with_limit(self):
        client, app = self._make_client()
        kernel = app.state.leviathan["kernel"]
        event_log = app.state.leviathan["event_log"]
        import dataclasses
        from api.models import EventEnvelope
        for i in range(5):
            kernel.begin_round()
            receipt = kernel.settle_round(seed=i + 1)
            event_log.append(EventEnvelope(
                event_id=len(event_log) + 1, event_type="round_settled",
                round_id=receipt.round_id, timestamp=receipt.timestamp,
                payload=dataclasses.asdict(receipt),
            ))
        resp = client.get("/v1/world/metrics/history?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_judge_stats(self):
        client, app = self._make_client()
        resp = client.get("/v1/world/judge/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_judgments"] == 0
        assert data["approval_rate"] == 0.0


# ──────────────────────────────────────────────
# Phase 3 — Governance integration test
# ──────────────────────────────────────────────

import threading
import time as _time


class TestGovernanceIntegration:
    def test_full_governance_round(self):
        """E2E: register, propose mechanism, judge evaluates, verify receipt has metrics."""
        from scripts.run_server import build_app, _simulation_loop
        from starlette.testclient import TestClient

        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        client = TestClient(app)

        kernel = app.state.leviathan["kernel"]
        event_log = app.state.leviathan["event_log"]
        round_state = app.state.leviathan["round_state"]
        mechanism_registry = app.state.leviathan["mechanism_registry"]
        judge = app.state.leviathan["judge"]

        # Register an agent
        resp = client.post("/v1/agents/register", json={"name": "GovBot"})
        assert resp.status_code == 200
        agent = resp.json()

        # Run sim loop for 1 round with short pace
        stop = threading.Event()
        t = threading.Thread(
            target=_simulation_loop,
            args=(kernel, event_log, round_state, mechanism_registry, judge, 0.5, 1, stop),
        )
        t.start()

        # Wait for accepting state
        for _ in range(20):
            _time.sleep(0.05)
            if round_state.state == "accepting":
                break

        # Submit a mechanism proposal during the window
        if round_state.state == "accepting":
            resp = client.post(
                "/v1/world/mechanisms/propose",
                json={
                    "code": "def propose_modification(e): pass",
                    "description": "no-op mechanism",
                    "idempotency_key": "gov-test-1",
                },
                headers={"X-API-Key": agent["api_key"]},
            )
            assert resp.status_code == 200

        t.join(timeout=5)
        stop.set()

        # Verify the round settled with metrics
        assert len(event_log) >= 1
        last_event = event_log[-1]
        assert last_event.event_type == "round_settled"
        assert "total_vitality" in last_event.payload.get("round_metrics", {})
        assert "population" in last_event.payload.get("round_metrics", {})


# ──────────────────────────────────────────────
# Phase 4 – EventEnvelope enrichment tests
# ──────────────────────────────────────────────

class TestEventEnvelopeEnrichment:
    def test_event_envelope_new_fields_default_none(self):
        event = EventEnvelope(
            event_id=1, event_type="test", round_id=1,
            timestamp="t", payload={},
        )
        assert event.world_id is None
        assert event.phase is None
        assert event.payload_hash is None
        assert event.prev_event_hash is None

    def test_event_envelope_with_enriched_fields(self):
        event = EventEnvelope(
            event_id=1, event_type="round_settled", round_id=1,
            timestamp="t", payload={"key": "val"},
            world_id="w-123", phase="settlement",
            payload_hash="aa" * 32, prev_event_hash="bb" * 32,
        )
        assert event.world_id == "w-123"
        assert event.phase == "settlement"
        assert event.payload_hash == "aa" * 32
        assert event.prev_event_hash == "bb" * 32


class TestPhase4WorldEndpoint:
    @pytest.fixture
    def client(self):
        from scripts.run_server import build_app
        from fastapi.testclient import TestClient
        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        return TestClient(app)

    def test_world_info_has_public_key(self, client):
        resp = client.get("/v1/world")
        assert resp.status_code == 200
        data = resp.json()
        assert "world_public_key" in data
        assert len(data["world_public_key"]) == 64

    def test_receipt_fields_in_api(self, client):
        kernel = client.app.state.leviathan["kernel"]
        kernel.begin_round()
        kernel.settle_round(seed=1)
        resp = client.get("/v1/world/rounds/current")
        assert resp.status_code == 200
        data = resp.json()
        lr = data["last_receipt"]
        assert lr["constitution_hash"] is not None
        assert lr["oracle_signature"] is not None
        assert lr["world_public_key"] is not None


# ── Phase 4: Moderator Auth, Admin Routes, Ban Enforcement Tests ──


from api.auth import APIKeyAuth


class TestModeratorAuth:
    def test_moderator_key_is_valid(self):
        auth = APIKeyAuth(api_keys={"agent1"}, moderator_keys={"mod1"})
        assert auth.is_moderator_key("mod1") is True
        assert auth.is_moderator_key("agent1") is False
        assert auth.is_moderator_key("unknown") is False

    def test_moderator_key_also_valid_as_regular(self):
        auth = APIKeyAuth(api_keys={"agent1"}, moderator_keys={"mod1"})
        assert auth.is_valid_key("mod1") is True
        assert auth.is_valid_key("agent1") is True
        assert auth.is_valid_key("unknown") is False

    def test_auth_disabled_when_no_keys(self):
        auth = APIKeyAuth()
        assert auth.enabled is False


class TestAdminEndpoints:
    @pytest.fixture
    def mod_client(self):
        from scripts.run_server import build_app
        from fastapi.testclient import TestClient
        app = build_app(
            members=5, land_w=10, land_h=10, seed=42,
            api_keys={"agent1"}, moderator_keys={"mod1"},
        )
        return TestClient(app)

    def test_admin_status(self, mod_client):
        resp = mod_client.get("/v1/admin/status", headers={"X-API-Key": "mod1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["paused"] is False
        assert data["banned_agents"] == []
        assert "quotas" in data

    def test_admin_status_rejects_agent_key(self, mod_client):
        resp = mod_client.get("/v1/admin/status", headers={"X-API-Key": "agent1"})
        assert resp.status_code == 403

    def test_pause_and_resume(self, mod_client):
        resp = mod_client.post("/v1/admin/pause", headers={"X-API-Key": "mod1"})
        assert resp.status_code == 200
        status = mod_client.get("/v1/admin/status", headers={"X-API-Key": "mod1"}).json()
        assert status["paused"] is True
        resp = mod_client.post("/v1/admin/resume", headers={"X-API-Key": "mod1"})
        assert resp.status_code == 200
        status = mod_client.get("/v1/admin/status", headers={"X-API-Key": "mod1"}).json()
        assert status["paused"] is False

    def test_ban_and_unban(self, mod_client):
        resp = mod_client.post("/v1/admin/ban/1", headers={"X-API-Key": "mod1"})
        assert resp.status_code == 200
        status = mod_client.get("/v1/admin/status", headers={"X-API-Key": "mod1"}).json()
        assert 1 in status["banned_agents"]
        resp = mod_client.post("/v1/admin/unban/1", headers={"X-API-Key": "mod1"})
        assert resp.status_code == 200
        status = mod_client.get("/v1/admin/status", headers={"X-API-Key": "mod1"}).json()
        assert 1 not in status["banned_agents"]

    def test_update_quotas(self, mod_client):
        resp = mod_client.put(
            "/v1/admin/quotas",
            json={"max_actions_per_round": 3, "max_proposals_per_round": 1},
            headers={"X-API-Key": "mod1"},
        )
        assert resp.status_code == 200
        status = mod_client.get("/v1/admin/status", headers={"X-API-Key": "mod1"}).json()
        assert status["quotas"]["max_actions_per_round"] == 3

    def test_admin_actions_emit_events(self, mod_client):
        mod_client.post("/v1/admin/pause", headers={"X-API-Key": "mod1"})
        mod_client.post("/v1/admin/ban/1", headers={"X-API-Key": "mod1"})
        events = mod_client.get("/v1/world/events").json()
        admin_events = [e for e in events if e["event_type"].startswith("admin_")]
        assert len(admin_events) >= 2
        assert any(e["event_type"] == "admin_pause" for e in admin_events)
        assert any(e["event_type"] == "admin_ban" for e in admin_events)

    def test_admin_event_has_moderator_hash_not_key(self, mod_client):
        mod_client.post("/v1/admin/pause", headers={"X-API-Key": "mod1"})
        events = mod_client.get("/v1/world/events").json()
        admin_events = [e for e in events if e["event_type"] == "admin_pause"]
        assert len(admin_events) == 1
        payload = admin_events[0]["payload"]
        assert "moderator_key_hash" in payload
        assert payload["moderator_key_hash"] != "mod1"


class TestBanEnforcement:
    @pytest.fixture
    def banned_client(self):
        from scripts.run_server import build_app
        from fastapi.testclient import TestClient
        app = build_app(
            members=5, land_w=10, land_h=10, seed=42,
            api_keys=set(), moderator_keys={"mod1"},
        )
        client = TestClient(app)
        resp = client.post("/v1/agents/register", json={"name": "test_agent"})
        assert resp.status_code == 200
        agent_key = resp.json()["api_key"]
        member_id = resp.json()["member_id"]
        kernel = app.state.leviathan["kernel"]
        round_state = app.state.leviathan["round_state"]
        kernel.begin_round()
        round_state.open_submissions(round_id=kernel.round_id, pace=60.0)
        return client, agent_key, member_id

    def test_banned_agent_action_rejected(self, banned_client):
        client, agent_key, member_id = banned_client
        client.post(f"/v1/admin/ban/{member_id}", headers={"X-API-Key": "mod1"})
        resp = client.post(
            "/v1/world/actions",
            json={"code": "pass", "idempotency_key": "k1"},
            headers={"X-API-Key": agent_key},
        )
        assert resp.status_code == 403
        assert "banned" in resp.json()["detail"].lower()

    def test_banned_agent_proposal_rejected(self, banned_client):
        client, agent_key, member_id = banned_client
        client.post(f"/v1/admin/ban/{member_id}", headers={"X-API-Key": "mod1"})
        resp = client.post(
            "/v1/world/mechanisms/propose",
            json={"code": "pass", "description": "test", "idempotency_key": "k2"},
            headers={"X-API-Key": agent_key},
        )
        assert resp.status_code == 403

    def test_unbanned_agent_can_submit(self, banned_client):
        client, agent_key, member_id = banned_client
        client.post(f"/v1/admin/ban/{member_id}", headers={"X-API-Key": "mod1"})
        client.post(f"/v1/admin/unban/{member_id}", headers={"X-API-Key": "mod1"})
        resp = client.post(
            "/v1/world/actions",
            json={"code": "pass", "idempotency_key": "k3"},
            headers={"X-API-Key": agent_key},
        )
        assert resp.status_code == 200
