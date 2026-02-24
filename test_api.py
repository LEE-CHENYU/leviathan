"""Tests for the Phase 1 Read API (api package)."""

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
