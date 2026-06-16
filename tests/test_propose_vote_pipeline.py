"""End-to-end integration tests for the propose → canary → vote → execute pipeline.

Tests cover:
1. Full pipeline: clean proposal → canary → pending_vote → majority vote → execute → state mutated
2. Broken proposal → canary error → auto-reject → state unchanged
3. Dangerous proposal → divergence flagged → stays pending → explicit vote → approve/reject
4. Multiple proposals: one clean, one broken → correct triage
5. Cross-round voting: flagged proposal persists, then gets voted on next round
6. Enriched /v1/world API reflects pending_proposals_count and checkpoints_available
7. New prompt context: canary_reports, pending_proposals, checkpoint_info in prepare_agent_data()
"""

import tempfile
import pytest

from MetaIsland.metaIsland import IslandExecution
from MetaIsland.nodes.canary_node import CanaryNode, AgentReviewNode
from kernel.canary import CanaryRunner, CanaryReport


# ── Helpers ──────────────────────────────────────────

@pytest.fixture
def execution(tmp_path):
    """Create a fresh IslandExecution with 5 members for pipeline testing."""
    engine = IslandExecution(5, (10, 10), str(tmp_path), random_seed=42)
    engine.new_round()  # Initialize round 1
    return engine


def _make_proposal(member_id, code, proposal_id=None):
    """Build a proposal dict matching what ProposeMechanismNode would produce."""
    if proposal_id is None:
        proposal_id = f"prop_{member_id}"
    return {
        "code": code,
        "member_id": member_id,
        "proposal_id": proposal_id,
        "round": 1,
    }


# Safe no-op mechanism
NOOP_CODE = "def propose_modification(execution_engine):\n    pass\n"

# Mechanism that adds a new attribute to the engine (observable state change)
MARKER_CODE = (
    "def propose_modification(execution_engine):\n"
    "    execution_engine._test_marker = 'pipeline_activated'\n"
)

# Mechanism that modifies vitality AND raises (for testing rejection + state restore)
VITALITY_THEN_CRASH_CODE = (
    "def propose_modification(execution_engine):\n"
    "    for m in execution_engine.current_members:\n"
    "        m.vitality = 9999\n"
    "    raise RuntimeError('after modification')\n"
)

# Mechanism that kills an agent (dangerous)
LETHAL_CODE = (
    "def propose_modification(execution_engine):\n"
    "    for m in execution_engine.current_members:\n"
    "        m.vitality = 0\n"
)

# Mechanism with a syntax error
BROKEN_CODE = "def propose_modification(execution_engine)\n    pass\n"

# Mechanism that raises at runtime
CRASH_CODE = (
    "def propose_modification(execution_engine):\n"
    "    raise RuntimeError('boom')\n"
)


# ═══════════════════════════════════════════════════════
#  Test 1: Full clean pipeline — propose → canary → vote → approve → execute
# ═══════════════════════════════════════════════════════

class TestCleanProposalPipeline:
    def test_clean_proposal_approved_and_executed(self, execution):
        """A clean mechanism goes through canary, agents vote, then it executes."""
        proposal = _make_proposal(0, MARKER_CODE, "prop_clean")

        # Snapshot domain state before canary
        vitality_before = [m.vitality for m in execution.current_members]

        # Step 1: Canary
        canary_node = CanaryNode()
        context = {"execution": execution}
        canary_result = canary_node.execute(context, {"proposals": [proposal]})

        assert len(canary_result["canary_reports"]) == 1
        report = canary_result["canary_reports"][0]
        assert report["execution_error"] is None
        assert report["divergence_flags"] == []
        assert "canary_report" in proposal  # annotated

        # Domain state restored by canary (members unchanged)
        vitality_after_canary = [m.vitality for m in execution.current_members]
        assert vitality_before == vitality_after_canary

        # Step 2: Agent Review (round 1) — clean proposal stays pending
        review_node = AgentReviewNode()
        review_result = review_node.execute(
            {"execution": execution, "round": 1},
            {"proposals": canary_result["proposals"]},
        )

        assert len(review_result["approved"]) == 0
        assert len(review_result["rejected"]) == 0
        assert len(execution.pending_proposals) == 1  # still pending, needs votes

        # Step 2b: Agents cast majority yes votes
        living_count = len(execution.current_members)
        majority = (living_count // 2) + 1
        for i in range(majority):
            execution.pending_proposals["prop_clean"]["votes"][i] = True

        # Step 2c: Agent Review (round 2) — votes resolve
        review_result = review_node.execute(
            {"execution": execution, "round": 2},
            {"proposals": []},  # no new proposals
        )

        assert len(review_result["approved"]) == 1
        assert review_result["approved"][0]["proposal_id"] == "prop_clean"
        assert len(execution.pending_proposals) == 0

        # Step 3: Execute
        execution.execute_mechanism_modifications(approved=review_result["approved"])

        # State IS now changed
        assert hasattr(execution, "_test_marker")
        assert execution._test_marker == "pipeline_activated"

        # Execution history records it
        mods = execution.execution_history["rounds"][-1]["mechanism_modifications"]
        assert len(mods["executed"]) == 1

    def test_noop_proposal_clean_canary(self, execution):
        """A no-op mechanism passes canary with 0% vitality change."""
        proposal = _make_proposal(0, NOOP_CODE)
        runner = CanaryRunner()
        report = runner.run_canary(execution, NOOP_CODE, "prop_noop", 0)

        assert report.execution_error is None
        assert report.vitality_change_pct == pytest.approx(0.0)
        assert report.agents_died == []
        assert report.divergence_flags == []


# ═══════════════════════════════════════════════════════
#  Test 2: Broken proposal — auto-rejected by canary error
# ═══════════════════════════════════════════════════════

class TestBrokenProposalRejection:
    def test_syntax_error_auto_rejected(self, execution):
        """A proposal with syntax error is auto-rejected."""
        proposal = _make_proposal(1, BROKEN_CODE, "prop_broken")

        canary_node = CanaryNode()
        canary_result = canary_node.execute(
            {"execution": execution}, {"proposals": [proposal]}
        )
        assert canary_result["canary_reports"][0]["execution_error"] is not None

        review_node = AgentReviewNode()
        review_result = review_node.execute(
            {"execution": execution, "round": 1},
            {"proposals": canary_result["proposals"]},
        )

        assert len(review_result["rejected"]) == 1
        assert len(review_result["approved"]) == 0
        assert "error" in review_result["rejected"][0]["reason"].lower()

    def test_runtime_crash_auto_rejected(self, execution):
        """A proposal that raises at runtime is auto-rejected."""
        proposal = _make_proposal(2, CRASH_CODE, "prop_crash")

        canary_node = CanaryNode()
        canary_result = canary_node.execute(
            {"execution": execution}, {"proposals": [proposal]}
        )

        review_node = AgentReviewNode()
        review_result = review_node.execute(
            {"execution": execution, "round": 1},
            {"proposals": canary_result["proposals"]},
        )

        assert len(review_result["rejected"]) == 1
        assert len(review_result["approved"]) == 0

    def test_rejected_proposal_does_not_execute(self, execution):
        """Rejected proposals should not mutate domain state."""
        proposal = _make_proposal(0, VITALITY_THEN_CRASH_CODE, "prop_fail")
        # This code sets all vitality to 9999 then raises — canary catches the error

        vitality_before = [m.vitality for m in execution.current_members]

        canary_node = CanaryNode()
        canary_result = canary_node.execute(
            {"execution": execution}, {"proposals": [proposal]}
        )

        # Canary restores domain state (member vitalities unchanged)
        vitality_after_canary = [m.vitality for m in execution.current_members]
        assert vitality_before == vitality_after_canary

        review_node = AgentReviewNode()
        review_result = review_node.execute(
            {"execution": execution, "round": 1},
            {"proposals": canary_result["proposals"]},
        )

        assert len(review_result["rejected"]) == 1
        # Execute with empty approved list — domain state stays unchanged
        execution.execute_mechanism_modifications(approved=review_result["approved"])
        vitality_after_exec = [m.vitality for m in execution.current_members]
        assert vitality_before == vitality_after_exec


# ═══════════════════════════════════════════════════════
#  Test 3: Dangerous proposal — flagged, stays pending, needs explicit vote
# ═══════════════════════════════════════════════════════

class TestDangerousProposalVoting:
    def test_lethal_proposal_flagged_stays_pending(self, execution):
        """A mechanism that kills agents gets flagged and stays pending."""
        proposal = _make_proposal(0, LETHAL_CODE, "prop_lethal")

        canary_node = CanaryNode()
        canary_result = canary_node.execute(
            {"execution": execution}, {"proposals": [proposal]}
        )

        report = canary_result["canary_reports"][0]
        assert report["vitality_change_pct"] < -90.0  # nearly all vitality gone
        assert len(report["divergence_flags"]) > 0

        review_node = AgentReviewNode()
        review_result = review_node.execute(
            {"execution": execution, "round": 1},
            {"proposals": canary_result["proposals"]},
        )

        # Should stay pending (divergence flag but no explicit votes)
        assert len(review_result["approved"]) == 0
        assert len(review_result["rejected"]) == 0
        assert "prop_lethal" in execution.pending_proposals

    def test_explicit_majority_yes_approves_flagged(self, execution):
        """Flagged proposal with majority yes votes gets approved."""
        proposal = _make_proposal(0, MARKER_CODE, "prop_flagged")
        # Manually add divergence flag to simulate a flagged proposal
        proposal["canary_report"] = {
            "execution_error": None,
            "divergence_flags": ["custom_flag"],
            "vitality_change_pct": -10.0,
        }

        # Put it in pending queue with majority yes votes (5 members, majority=3)
        execution.pending_proposals = {
            "prop_flagged": {
                "proposal": proposal,
                "votes": {0: True, 1: True, 2: True},  # 3/5 yes = majority
                "round_submitted": 1,
            }
        }

        review_node = AgentReviewNode()
        review_result = review_node.execute(
            {"execution": execution, "round": 2},
            {"proposals": []},  # no new proposals, just processing pending
        )

        assert len(review_result["approved"]) == 1
        assert review_result["approved"][0]["proposal_id"] == "prop_flagged"
        assert "prop_flagged" not in execution.pending_proposals

    def test_explicit_majority_no_rejects_flagged(self, execution):
        """Flagged proposal with majority no votes gets rejected."""
        proposal = _make_proposal(0, MARKER_CODE, "prop_flagged_no")
        proposal["canary_report"] = {
            "execution_error": None,
            "divergence_flags": ["custom_flag"],
        }

        execution.pending_proposals = {
            "prop_flagged_no": {
                "proposal": proposal,
                "votes": {0: False, 1: False, 2: False},  # 3/5 no = majority
                "round_submitted": 1,
            }
        }

        review_node = AgentReviewNode()
        review_result = review_node.execute(
            {"execution": execution, "round": 2},
            {"proposals": []},
        )

        assert len(review_result["rejected"]) == 1
        assert "prop_flagged_no" not in execution.pending_proposals

    def test_insufficient_votes_stays_pending(self, execution):
        """Flagged proposal without enough votes stays pending."""
        proposal = _make_proposal(0, MARKER_CODE, "prop_stuck")
        proposal["canary_report"] = {
            "execution_error": None,
            "divergence_flags": ["custom_flag"],
        }

        execution.pending_proposals = {
            "prop_stuck": {
                "proposal": proposal,
                "votes": {0: True},  # Only 1/5 voted — not majority
                "round_submitted": 1,
            }
        }

        review_node = AgentReviewNode()
        review_result = review_node.execute(
            {"execution": execution, "round": 2},
            {"proposals": []},
        )

        assert len(review_result["approved"]) == 0
        assert len(review_result["rejected"]) == 0
        assert "prop_stuck" in execution.pending_proposals


# ═══════════════════════════════════════════════════════
#  Test 4: Multiple proposals — correct triage
# ═══════════════════════════════════════════════════════

class TestMultipleProposalTriage:
    def test_clean_and_broken_correctly_triaged(self, execution):
        """One clean and one broken proposal: broken rejected, clean pending then approved by vote."""
        clean = _make_proposal(0, MARKER_CODE, "prop_good")
        broken = _make_proposal(1, CRASH_CODE, "prop_bad")

        canary_node = CanaryNode()
        canary_result = canary_node.execute(
            {"execution": execution}, {"proposals": [clean, broken]}
        )

        assert len(canary_result["canary_reports"]) == 2

        review_node = AgentReviewNode()
        review_result = review_node.execute(
            {"execution": execution, "round": 1},
            {"proposals": canary_result["proposals"]},
        )

        # Broken auto-rejected, clean stays pending
        assert len(review_result["approved"]) == 0
        assert len(review_result["rejected"]) == 1
        assert review_result["rejected"][0]["proposal"]["proposal_id"] == "prop_bad"
        assert "prop_good" in execution.pending_proposals

        # Cast majority votes on clean proposal
        living_count = len(execution.current_members)
        majority = (living_count // 2) + 1
        for i in range(majority):
            execution.pending_proposals["prop_good"]["votes"][i] = True

        # Round 2: votes resolve
        review_result = review_node.execute(
            {"execution": execution, "round": 2},
            {"proposals": []},
        )
        assert len(review_result["approved"]) == 1
        assert review_result["approved"][0]["proposal_id"] == "prop_good"

    def test_clean_and_flagged_correctly_triaged(self, execution):
        """One clean and one lethal proposal: both stay pending until voted."""
        clean = _make_proposal(0, NOOP_CODE, "prop_safe")
        lethal = _make_proposal(1, LETHAL_CODE, "prop_danger")

        canary_node = CanaryNode()
        canary_result = canary_node.execute(
            {"execution": execution}, {"proposals": [clean, lethal]}
        )

        review_node = AgentReviewNode()
        review_result = review_node.execute(
            {"execution": execution, "round": 1},
            {"proposals": canary_result["proposals"]},
        )

        # Both stay pending — clean needs votes, lethal needs votes
        assert len(review_result["approved"]) == 0
        assert len(review_result["rejected"]) == 0
        assert "prop_safe" in execution.pending_proposals
        assert "prop_danger" in execution.pending_proposals


# ═══════════════════════════════════════════════════════
#  Test 5: Cross-round voting
# ═══════════════════════════════════════════════════════

class TestCrossRoundVoting:
    def test_flagged_proposal_persists_and_resolves_next_round(self, execution):
        """Proposal flagged in round 1 accumulates votes in round 2 and resolves."""
        # Round 1: proposal enters as flagged → stays pending
        proposal = _make_proposal(0, MARKER_CODE, "prop_cross")

        canary_node = CanaryNode()
        review_node = AgentReviewNode()

        # Manually flag it
        proposal["canary_report"] = {
            "execution_error": None,
            "divergence_flags": ["custom_warning"],
        }

        review_result_r1 = review_node.execute(
            {"execution": execution, "round": 1},
            {"proposals": [proposal]},
        )

        assert "prop_cross" in execution.pending_proposals
        assert len(review_result_r1["approved"]) == 0

        # Round 2: agents submit votes (majority yes)
        execution.pending_proposals["prop_cross"]["votes"] = {
            0: True, 1: True, 2: True,
        }

        # Start new round
        execution.new_round()
        review_result_r2 = review_node.execute(
            {"execution": execution, "round": 2},
            {"proposals": []},  # no new proposals
        )

        assert len(review_result_r2["approved"]) == 1
        assert review_result_r2["approved"][0]["proposal_id"] == "prop_cross"
        assert "prop_cross" not in execution.pending_proposals


# ═══════════════════════════════════════════════════════
#  Test 6: Enriched API reflects pipeline state
# ═══════════════════════════════════════════════════════

class TestEnrichedWorldAPI:
    def test_world_info_reflects_pending_proposals(self):
        """The /v1/world endpoint reports pending_proposals_count."""
        from scripts.run_server import build_app
        from starlette.testclient import TestClient

        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        client = TestClient(app)
        kernel = app.state.leviathan["kernel"]

        # Initially no pending proposals
        data = client.get("/v1/world").json()
        assert data["pending_proposals_count"] == 0

        # Manually add a pending proposal
        kernel._execution.pending_proposals = {
            "prop_test": {
                "proposal": {"code": "x", "member_id": 0},
                "votes": {},
                "round_submitted": 1,
            }
        }

        data = client.get("/v1/world").json()
        assert data["pending_proposals_count"] == 1

    def test_world_info_governance_defaults(self):
        """Governance field reports correct defaults."""
        from scripts.run_server import build_app
        from starlette.testclient import TestClient

        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        client = TestClient(app)

        data = client.get("/v1/world").json()
        gov = data["governance"]
        assert gov["judge_role"] == "advisory"
        assert gov["voting_threshold"] == "majority"


# ═══════════════════════════════════════════════════════
#  Test 7: prepare_agent_data includes new context fields
# ═══════════════════════════════════════════════════════

class TestPrepareAgentDataContext:
    def test_canary_reports_in_agent_data(self, execution):
        """prepare_agent_data returns canary_reports from current round."""
        # Inject a canary report into execution history
        mods = execution.execution_history["rounds"][-1]["mechanism_modifications"]
        mods["canary_reports"] = [
            {"proposal_id": "p1", "vitality_change_pct": -5.0, "divergence_flags": []}
        ]

        data = execution.prepare_agent_data(0, error_context_type="mechanism")
        assert "canary_reports" in data
        assert len(data["canary_reports"]) == 1
        assert data["canary_reports"][0]["proposal_id"] == "p1"

    def test_pending_proposals_in_agent_data(self, execution):
        """prepare_agent_data returns pending_proposals summary."""
        execution.pending_proposals = {
            "prop_test": {
                "proposal": {"canary_report": {"vitality_change_pct": -3.0}},
                "votes": {0: True},
                "round_submitted": 1,
            }
        }

        data = execution.prepare_agent_data(0, error_context_type="mechanism")
        assert "pending_proposals" in data
        assert len(data["pending_proposals"]) == 1
        assert data["pending_proposals"][0]["proposal_id"] == "prop_test"

    def test_checkpoint_info_in_agent_data(self, execution):
        """prepare_agent_data returns checkpoint_info."""
        data = execution.prepare_agent_data(0, error_context_type="mechanism")
        assert "checkpoint_info" in data
        # May be empty if no checkpoints yet, but the key must exist
        assert isinstance(data["checkpoint_info"], list)

    def test_empty_context_when_no_data(self, execution):
        """Without canary/pending/checkpoint data, fields are empty lists."""
        data = execution.prepare_agent_data(0, error_context_type="mechanism")
        assert data["canary_reports"] == []
        assert data["pending_proposals"] == []
        assert isinstance(data["checkpoint_info"], list)


# ═══════════════════════════════════════════════════════
#  Test 8: Canary report appears in execution history
# ═══════════════════════════════════════════════════════

class TestCanaryReportInHistory:
    def test_review_results_in_execution_history(self, execution):
        """Agent review results are recorded in execution history."""
        proposal = _make_proposal(0, NOOP_CODE, "prop_hist")

        canary_node = CanaryNode()
        canary_result = canary_node.execute(
            {"execution": execution}, {"proposals": [proposal]}
        )

        review_node = AgentReviewNode()
        review_node.execute(
            {"execution": execution, "round": 1},
            {"proposals": canary_result["proposals"]},
        )

        mods = execution.execution_history["rounds"][-1]["mechanism_modifications"]
        assert "review_results" in mods
        # Clean proposal stays pending on first round (needs votes)
        assert mods["review_results"]["approved_count"] == 0
        assert mods["review_results"]["rejected_count"] == 0
        assert mods["review_results"]["still_pending_count"] == 1
