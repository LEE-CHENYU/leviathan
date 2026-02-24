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
