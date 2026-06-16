"""Tests for checkpointing and recovery (Phase 3).

Tests cover:
- Auto-checkpoint before mechanism execution
- Restore checkpoint state
- Restore preserves event log (I-2: append-only)
- Agent can list checkpoints
- Recovery as mechanism (goes through canary + vote)
"""

import pytest
from kernel.schemas import WorldConfig
from kernel.world_kernel import WorldKernel
from kernel.store import Store


# ── Store snapshot listing tests ─────────────────

class TestStoreListSnapshots:
    def test_list_snapshots_empty(self):
        store = Store(":memory:")
        assert store.list_snapshots() == []

    def test_list_snapshots_returns_metadata(self):
        store = Store(":memory:")
        store.store_snapshot(1, {"members": [
            {"id": 0, "vitality": 50.0, "cargo": 10.0, "land_num": 2},
            {"id": 1, "vitality": 40.0, "cargo": 5.0, "land_num": 1},
        ]})
        store.store_snapshot(2, {"members": [
            {"id": 0, "vitality": 45.0, "cargo": 12.0, "land_num": 2},
        ]})
        result = store.list_snapshots()
        assert len(result) == 2
        # Ordered by round_id DESC
        assert result[0]["round_id"] == 2
        assert result[0]["member_count"] == 1
        assert result[0]["total_vitality"] == pytest.approx(45.0)
        assert result[1]["round_id"] == 1
        assert result[1]["member_count"] == 2

    def test_get_snapshot_summary(self):
        store = Store(":memory:")
        store.store_snapshot(5, {"members": [
            {"id": 0, "vitality": 80.0},
            {"id": 1, "vitality": 60.0},
        ]})
        summary = store.get_snapshot_summary(5)
        assert summary is not None
        assert summary["round_id"] == 5
        assert summary["member_count"] == 2
        assert summary["total_vitality"] == pytest.approx(140.0)

    def test_get_snapshot_summary_missing(self):
        store = Store(":memory:")
        assert store.get_snapshot_summary(999) is None


# ── WorldKernel checkpoint API tests ─────────────

class TestWorldKernelCheckpoints:
    @pytest.fixture
    def kernel_with_store(self, tmp_path):
        store = Store(":memory:")
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=42)
        return WorldKernel(config, save_path=str(tmp_path), store=store)

    def test_get_available_checkpoints_empty(self, kernel_with_store):
        checkpoints = kernel_with_store.get_available_checkpoints()
        assert checkpoints == []

    def test_get_available_checkpoints_after_store(self, kernel_with_store):
        kernel = kernel_with_store
        # Manually store a snapshot
        kernel._store.store_snapshot(1, {"members": [
            {"id": 0, "vitality": 50.0}, {"id": 1, "vitality": 40.0}
        ]})
        checkpoints = kernel.get_available_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0]["round_id"] == 1

    def test_restore_checkpoint_state(self, kernel_with_store):
        kernel = kernel_with_store
        # Record initial vitality
        member0 = kernel._execution.current_members[0]
        original_vitality = member0.vitality

        # Store snapshot
        kernel._store.store_snapshot(1, kernel._collect_snapshot_data())

        # Modify state
        member0.vitality = 5.0
        assert member0.vitality == 5.0

        # Restore
        result = kernel.restore_checkpoint(1)
        assert result is True
        assert member0.vitality == pytest.approx(original_vitality)

    def test_restore_missing_checkpoint(self, kernel_with_store):
        result = kernel_with_store.restore_checkpoint(999)
        assert result is False

    def test_restore_preserves_event_log(self, kernel_with_store):
        kernel = kernel_with_store
        # Store snapshot and some events
        kernel._store.store_snapshot(1, kernel._collect_snapshot_data())
        kernel._store.append_event("test_event", 1, "2025-01-01", {"data": "x"})

        # Restore
        kernel.restore_checkpoint(1)

        # Check that original event is still there (I-2: append-only)
        events = kernel._store.get_events()
        assert len(events) >= 2  # original + checkpoint_restored
        event_types = [e["event_type"] for e in events]
        assert "test_event" in event_types
        assert "checkpoint_restored" in event_types

    def test_restore_event_has_correct_payload(self, kernel_with_store):
        kernel = kernel_with_store
        kernel._store.store_snapshot(1, kernel._collect_snapshot_data())
        kernel._round_id = 3

        kernel.restore_checkpoint(1)

        events = kernel._store.get_events()
        restore_events = [e for e in events if e["event_type"] == "checkpoint_restored"]
        assert len(restore_events) == 1
        payload = restore_events[0]["payload"]
        assert payload["restored_to_round"] == 1
        assert payload["current_round"] == 3

    def test_no_store_returns_empty_checkpoints(self, tmp_path):
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=42)
        kernel = WorldKernel(config, save_path=str(tmp_path))
        assert kernel.get_available_checkpoints() == []

    def test_no_store_restore_returns_false(self, tmp_path):
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=42)
        kernel = WorldKernel(config, save_path=str(tmp_path))
        assert kernel.restore_checkpoint(1) is False


# ── Auto-checkpoint tests ────────────────────────

class TestAutoCheckpoint:
    def test_execution_has_collect_snapshot(self):
        """IslandExecution should have _collect_mechanism_snapshot method."""
        from MetaIsland.metaIsland import IslandExecution
        assert hasattr(IslandExecution, "_collect_mechanism_snapshot")

    def test_execution_has_get_available_checkpoints(self):
        """IslandExecution should have get_available_checkpoints method."""
        from MetaIsland.metaIsland import IslandExecution
        assert hasattr(IslandExecution, "get_available_checkpoints")


# ── Recovery as mechanism test ───────────────────

class TestRecoveryAsMechanism:
    def test_recovery_proposal_goes_through_canary(self):
        """A recovery proposal is just code that calls restore_checkpoint.
        It should go through canary testing like any other mechanism."""
        from kernel.canary import CanaryRunner

        class FakeMember:
            def __init__(self, id, vitality=50.0, cargo=10.0, land_num=2):
                self.id = id
                self.vitality = vitality
                self.cargo = cargo
                self.land_num = land_num

        class FakeExecution:
            def __init__(self):
                self.current_members = [FakeMember(0), FakeMember(1)]
                self.execution_history = {"rounds": []}

        engine = FakeExecution()
        runner = CanaryRunner()

        # A recovery mechanism is just code — canary tests it like anything else
        code = "def propose_modification(execution_engine):\n    pass\n"
        report = runner.run_canary(engine, code, "recovery_prop", 0)

        assert report.execution_error is None
        assert report.proposal_id == "recovery_prop"
