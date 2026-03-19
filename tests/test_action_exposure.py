"""Tests for per-member action summaries and interaction data exposure."""
import pytest

from kernel.schemas import WorldConfig
from kernel.world_kernel import WorldKernel
from kernel.execution_sandbox import InProcessSandbox, SandboxContext
from MetaIsland.metaIsland import IslandExecution


@pytest.fixture
def island(tmp_path):
    """Create a small island with 3 members for testing."""
    return IslandExecution(
        init_member_number=3,
        land_shape=(5, 5),
        save_path=str(tmp_path),
        random_seed=42,
    )


@pytest.fixture
def world_kernel(tmp_path):
    """Create a WorldKernel with 3 members."""
    config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=42)
    return WorldKernel(config, save_path=str(tmp_path))


class TestLastRoundActions:
    def test_last_round_actions_populated(self, island):
        """After actions, new_round stamps action counts on members."""
        m0, m1, m2 = island.current_members[0], island.current_members[1], island.current_members[2]

        # Simulate: m0 attacks m1, m2 attacks m0
        island.record_action_dict["attack"][(m0.id, m1.id)] = 5.0
        island.record_action_dict["attack"][(m2.id, m0.id)] = 3.0
        # Simulate expand via _expand_counts
        island._expand_counts[m0.id] = 2
        island._expand_counts[m2.id] = 1

        island.new_round()

        assert m0.last_round_actions["attack"] == 1
        assert m0.last_round_actions["expand"] == 2
        assert m0.last_round_actions["offer"] == 0
        assert m1.last_round_actions["attack"] == 0
        assert m2.last_round_actions["expand"] == 1
        assert m2.last_round_actions["attack"] == 1

    def test_offer_details_populated(self, island):
        """Offer actions populate made/received dicts with correct IDs and amounts."""
        m0, m1 = island.current_members[0], island.current_members[1]

        island.record_action_dict["benefit"][(m0.id, m1.id)] = 12.5

        island.new_round()

        assert m0.last_round_offers_made == {m1.id: 12.5}
        assert m1.last_round_offers_received == {m0.id: 12.5}
        assert m0.last_round_offers_received == {}
        assert m1.last_round_offers_made == {}

    def test_attack_details_populated(self, island):
        """Attack actions populate made/received dicts with correct IDs and amounts."""
        m0, m1 = island.current_members[0], island.current_members[1]

        island.record_action_dict["attack"][(m0.id, m1.id)] = 15.3

        island.new_round()

        assert m0.last_round_attacks_made == {m1.id: 15.3}
        assert m1.last_round_attacks_received == {m0.id: 15.3}
        assert m0.last_round_attacks_received == {}
        assert m1.last_round_attacks_made == {}

    def test_action_data_survives_new_round(self, island):
        """Attributes persist after new_round() since stamping happens before reset."""
        m0, m1 = island.current_members[0], island.current_members[1]

        island.record_action_dict["attack"][(m0.id, m1.id)] = 7.0

        # new_round stamps data then clears record_action_dict
        island.new_round()

        assert m0.last_round_actions["attack"] == 1
        assert m0.last_round_attacks_made == {m1.id: 7.0}
        assert m1.last_round_attacks_received == {m0.id: 7.0}

        # Verify records were cleared
        assert island.record_action_dict["attack"] == {}

    def test_expand_tracked_via_expand_method(self, island):
        """Calling expand() on a member tracks the expand count."""
        m0 = island.current_members[0]
        # Try to expand (may or may not succeed depending on land availability)
        island.expand(m0)

        # Check that _expand_counts was updated (if expansion succeeded)
        if island._expand_counts.get(m0.id, 0) > 0:
            island.new_round()
            assert m0.last_round_actions["expand"] >= 1


class TestSnapshotExposure:
    def test_snapshot_includes_action_data(self, world_kernel):
        """WorldKernel.get_snapshot() includes per-member action fields."""
        snap = world_kernel.get_snapshot()

        for member_dict in snap.members:
            assert "last_round_actions" in member_dict
            assert set(member_dict["last_round_actions"].keys()) == {
                "expand", "attack", "offer", "offer_land"
            }
            assert "last_round_attacks_made" in member_dict
            assert "last_round_attacks_received" in member_dict
            assert "last_round_offers_made" in member_dict
            assert "last_round_offers_received" in member_dict

    def test_interaction_memory_in_snapshot(self, world_kernel):
        """interaction_memory appears in snapshot with correct structure."""
        engine = world_kernel._execution
        m0, m1 = engine.current_members[0], engine.current_members[1]

        # Record interactions and stamp via new_round
        engine.record_action_dict["attack"][(m0.id, m1.id)] = 10.0
        engine.new_round()

        snap = world_kernel.get_snapshot()
        m0_snap = next(m for m in snap.members if m["id"] == m0.id)

        assert "interaction_memory" in m0_snap
        im = m0_snap["interaction_memory"]
        assert "attack_made" in im
        assert str(m1.id) in im["attack_made"]

    def test_snapshot_action_data_after_actions(self, world_kernel):
        """Snapshot reflects actual action data after round with actions."""
        engine = world_kernel._execution
        m0, m1 = engine.current_members[0], engine.current_members[1]

        engine.record_action_dict["benefit"][(m0.id, m1.id)] = 8.0
        engine.new_round()

        snap = world_kernel.get_snapshot()
        m0_snap = next(m for m in snap.members if m["id"] == m0.id)
        m1_snap = next(m for m in snap.members if m["id"] == m1.id)

        assert m0_snap["last_round_actions"]["offer"] == 1
        assert m0_snap["last_round_offers_made"] == {m1.id: 8.0}
        assert m1_snap["last_round_offers_received"] == {m0.id: 8.0}


class TestMechanismAccess:
    def test_mechanism_can_read_action_data(self, island):
        """Mechanism code can read last_round_actions and modify vitality."""
        m0, m1 = island.current_members[0], island.current_members[1]

        # Simulate m0 offered last round
        island.record_action_dict["benefit"][(m0.id, m1.id)] = 5.0
        island.new_round()

        assert m0.last_round_actions["offer"] == 1

        # Record m1's vitality before mechanism
        m1_vit_before = m1.vitality

        # Run mechanism code that reads action data
        code = (
            "def propose_modification(execution_engine):\n"
            "    for m in execution_engine.current_members:\n"
            "        if m.last_round_actions.get('offer', 0) > 0:\n"
            "            m.vitality += 5\n"
        )

        vit_before = m0.vitality
        sandbox = InProcessSandbox()
        ctx = SandboxContext(execution_engine=island, member_index=0)
        result = sandbox.execute_mechanism_code(code, ctx)

        assert result.success, f"Mechanism failed: {result.error}"
        assert m0.vitality == pytest.approx(vit_before + 5)
        # m1 didn't offer, so no bonus
        assert m1.vitality == pytest.approx(m1_vit_before)
