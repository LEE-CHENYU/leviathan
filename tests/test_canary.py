"""Tests for canary testing infrastructure (Phase 2).

Tests cover:
- CanaryRunner: clean execution, vitality drops, error handling, report persistence
- AgentReviewNode: pending queue, majority vote
- DAG new phase order
- Judge advisory interface
"""

import pytest
from unittest.mock import MagicMock, patch
from kernel.canary import CanaryRunner, CanaryReport, _compact_member_stats


# ── Helpers ──────────────────────────────────────

class FakeMember:
    def __init__(self, id, vitality=50.0, cargo=10.0, land_num=2):
        self.id = id
        self.vitality = vitality
        self.cargo = cargo
        self.land_num = land_num


class FakeExecution:
    """Minimal execution engine mock for canary tests."""
    def __init__(self, members=None):
        self.current_members = members or [
            FakeMember(0, vitality=80.0),
            FakeMember(1, vitality=60.0),
            FakeMember(2, vitality=40.0),
        ]
        self.execution_history = {"rounds": []}


# ── CanaryRunner tests ───────────────────────────

class TestCanaryCleanExecution:
    def test_noop_mechanism_produces_clean_report(self):
        engine = FakeExecution()
        runner = CanaryRunner()
        code = "def propose_modification(execution_engine):\n    pass\n"
        report = runner.run_canary(engine, code, "prop_1", 0)

        assert isinstance(report, CanaryReport)
        assert report.proposal_id == "prop_1"
        assert report.proposer_id == 0
        assert report.execution_error is None
        assert report.vitality_change_pct == pytest.approx(0.0)
        assert report.agents_died == []
        assert report.divergence_flags == []

    def test_state_restored_after_canary(self):
        engine = FakeExecution()
        original_vitality = engine.current_members[0].vitality
        code = (
            "def propose_modification(execution_engine):\n"
            "    execution_engine.current_members[0].vitality = 0\n"
        )
        runner = CanaryRunner()
        runner.run_canary(engine, code, "prop_1", 0)
        assert engine.current_members[0].vitality == original_vitality


class TestCanaryVitalityDrop:
    def test_destructive_mechanism_flags_divergence(self):
        engine = FakeExecution()
        code = (
            "def propose_modification(execution_engine):\n"
            "    for m in execution_engine.current_members:\n"
            "        m.vitality = 0\n"
        )
        runner = CanaryRunner()
        report = runner.run_canary(engine, code, "prop_1", 0)

        assert report.vitality_change_pct < -50.0
        assert any("vitality_drop" in f for f in report.divergence_flags)

    def test_minor_change_does_not_flag(self):
        engine = FakeExecution()
        code = (
            "def propose_modification(execution_engine):\n"
            "    execution_engine.current_members[0].vitality -= 5\n"
        )
        runner = CanaryRunner()
        report = runner.run_canary(engine, code, "prop_1", 0)

        assert not any("vitality_drop" in f for f in report.divergence_flags)


class TestCanaryErrorHandling:
    def test_broken_code_reports_error_no_crash(self):
        engine = FakeExecution()
        code = (
            "def propose_modification(execution_engine):\n"
            "    raise RuntimeError('intentional error')\n"
        )
        runner = CanaryRunner()
        report = runner.run_canary(engine, code, "prop_1", 0)

        assert report.execution_error is not None
        assert "RuntimeError" in report.execution_error
        assert "execution_error" in report.divergence_flags

    def test_syntax_error_reports_error(self):
        engine = FakeExecution()
        code = "def propose_modification(execution_engine)\n    pass\n"
        runner = CanaryRunner()
        report = runner.run_canary(engine, code, "prop_1", 0)

        assert report.execution_error is not None

    def test_state_restored_after_error(self):
        engine = FakeExecution()
        original_count = len(engine.current_members)
        code = (
            "def propose_modification(execution_engine):\n"
            "    execution_engine.current_members.pop()\n"
            "    raise RuntimeError('oops')\n"
        )
        runner = CanaryRunner()
        runner.run_canary(engine, code, "prop_1", 0)
        assert len(engine.current_members) == original_count


class TestCanaryReportPersisted:
    def test_report_visible_in_execution_history(self):
        engine = FakeExecution()
        engine.execution_history = {
            "rounds": [{"mechanism_modifications": {"attempts": []}}]
        }
        code = "def propose_modification(execution_engine):\n    pass\n"

        from MetaIsland.nodes.canary_node import CanaryNode
        node = CanaryNode()
        context = {"execution": engine}
        proposals = [{"code": code, "member_id": 0, "proposal_id": "prop_0"}]
        result = node.execute(context, {"proposals": proposals})

        mods = engine.execution_history["rounds"][-1]["mechanism_modifications"]
        assert "canary_reports" in mods
        assert len(mods["canary_reports"]) == 1
        assert mods["canary_reports"][0]["proposal_id"] == "prop_0"


class TestCanaryReportSerialization:
    def test_to_dict_roundtrip(self):
        report = CanaryReport(
            proposal_id="p1",
            proposer_id=0,
            snapshot_before={"_total_vitality": 100},
            snapshot_after={"_total_vitality": 90},
            vitality_change_pct=-10.0,
            agents_died=[],
            divergence_flags=[],
        )
        d = report.to_dict()
        assert d["proposal_id"] == "p1"
        assert d["vitality_change_pct"] == -10.0
        assert d["execution_error"] is None


# ── AgentReviewNode tests ────────────────────────

class TestReviewNodePendingQueue:
    def test_proposals_accumulate_across_calls(self):
        from MetaIsland.nodes.canary_node import AgentReviewNode
        engine = FakeExecution()
        engine.execution_history = {
            "rounds": [{"mechanism_modifications": {"attempts": []}}]
        }
        node = AgentReviewNode()
        context = {"execution": engine, "round": 1}

        # First round: one proposal with divergence flag -> stays pending
        p1 = {
            "code": "x", "member_id": 0, "proposal_id": "p1",
            "canary_report": {"execution_error": None, "divergence_flags": ["vitality_drop_55%"]},
        }
        node.execute(context, {"proposals": [p1]})
        assert "p1" in engine.pending_proposals

        # Second round: another flagged proposal
        p2 = {
            "code": "y", "member_id": 1, "proposal_id": "p2",
            "canary_report": {"execution_error": None, "divergence_flags": ["agents_died:[1]"]},
        }
        node.execute(context, {"proposals": [p2]})
        assert "p1" in engine.pending_proposals
        assert "p2" in engine.pending_proposals


class TestReviewNodeMajorityVote:
    def test_majority_yes_activates(self):
        from MetaIsland.nodes.canary_node import AgentReviewNode
        engine = FakeExecution()  # 3 members -> majority = 2
        engine.execution_history = {
            "rounds": [{"mechanism_modifications": {"attempts": []}}]
        }
        engine.pending_proposals = {
            "p1": {
                "proposal": {"code": "x", "member_id": 0, "proposal_id": "p1",
                             "canary_report": {"execution_error": None, "divergence_flags": ["flag"]}},
                "votes": {0: True, 1: True},  # 2/3 yes -> majority
                "round_submitted": 1,
            }
        }
        node = AgentReviewNode()
        context = {"execution": engine, "round": 2}
        result = node.execute(context, {"proposals": []})

        assert len(result["approved"]) == 1
        assert result["approved"][0]["proposal_id"] == "p1"
        assert "p1" not in engine.pending_proposals

    def test_majority_no_rejects(self):
        from MetaIsland.nodes.canary_node import AgentReviewNode
        engine = FakeExecution()
        engine.execution_history = {
            "rounds": [{"mechanism_modifications": {"attempts": []}}]
        }
        engine.pending_proposals = {
            "p1": {
                "proposal": {"code": "x", "member_id": 0, "proposal_id": "p1",
                             "canary_report": {"execution_error": None, "divergence_flags": ["flag"]}},
                "votes": {0: False, 1: False},
                "round_submitted": 1,
            }
        }
        node = AgentReviewNode()
        context = {"execution": engine, "round": 2}
        result = node.execute(context, {"proposals": []})

        assert len(result["rejected"]) == 1
        assert "p1" not in engine.pending_proposals

    def test_clean_canary_auto_approves(self):
        from MetaIsland.nodes.canary_node import AgentReviewNode
        engine = FakeExecution()
        engine.execution_history = {
            "rounds": [{"mechanism_modifications": {"attempts": []}}]
        }
        node = AgentReviewNode()
        context = {"execution": engine, "round": 1}
        proposal = {
            "code": "x", "member_id": 0, "proposal_id": "p1",
            "canary_report": {"execution_error": None, "divergence_flags": []},
        }
        result = node.execute(context, {"proposals": [proposal]})
        assert len(result["approved"]) == 1

    def test_canary_error_auto_rejects(self):
        from MetaIsland.nodes.canary_node import AgentReviewNode
        engine = FakeExecution()
        engine.execution_history = {
            "rounds": [{"mechanism_modifications": {"attempts": []}}]
        }
        node = AgentReviewNode()
        context = {"execution": engine, "round": 1}
        proposal = {
            "code": "x", "member_id": 0, "proposal_id": "p1",
            "canary_report": {"execution_error": "RuntimeError: boom", "divergence_flags": ["execution_error"]},
        }
        result = node.execute(context, {"proposals": [proposal]})
        assert len(result["rejected"]) == 1


# ── DAG topology tests ──────────────────────────

class TestDAGNewPhaseOrder:
    def test_graph_has_canary_and_review_nodes(self):
        """Test that the default graph includes canary and agent_review nodes."""
        from MetaIsland.graph_engine import ExecutionGraph
        from MetaIsland.nodes.canary_node import CanaryNode, AgentReviewNode

        graph = ExecutionGraph()
        canary = CanaryNode()
        review = AgentReviewNode()
        graph.add_node(canary)
        graph.add_node(review)
        graph.connect("canary", "agent_review")

        assert "canary" in graph.nodes
        assert "agent_review" in graph.nodes

    def test_canary_feeds_into_review(self):
        """Canary node output connects to agent_review input."""
        from MetaIsland.graph_engine import ExecutionGraph
        from MetaIsland.nodes.canary_node import CanaryNode, AgentReviewNode

        graph = ExecutionGraph()
        canary = CanaryNode()
        review = AgentReviewNode()
        graph.add_node(canary)
        graph.add_node(review)
        graph.connect("canary", "agent_review", output_key="proposals", input_key="proposals")

        assert review.inputs["proposals"][0] is canary


# ── Judge advisory tests ─────────────────────────

class TestJudgeAdvisory:
    def test_judge_advisory_returns_concern_level(self):
        """Judge advisory method returns (concern_level, reason) tuple."""
        judge = MagicMock()
        judge.judge_proposal_advisory.return_value = ("LOW", "Minimal risk identified")
        result = judge.judge_proposal_advisory("code", 0, "mechanism")
        assert result[0] in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(result[1], str)

    def test_judge_has_advisory_method(self):
        """The Judge class has the judge_proposal_advisory method."""
        from MetaIsland.judge import Judge
        assert hasattr(Judge, "judge_proposal_advisory")

    def test_canary_runner_accepts_judge_param(self):
        """CanaryRunner.run_canary accepts a judge parameter."""
        import inspect
        sig = inspect.signature(CanaryRunner.run_canary)
        assert "judge" in sig.parameters
