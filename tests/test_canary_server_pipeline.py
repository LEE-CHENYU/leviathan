"""Integration tests for the canary testing + agent voting server pipeline.

Tests the full flow: propose → canary test → reject errors / flag divergence /
pending_vote for clean → agents vote → resolve_votes → execute approved → activate.
"""

import threading
import time

import pytest
from starlette.testclient import TestClient

from scripts.run_server import build_app, _simulation_loop


# ── Helpers ──────────────────────────────────────────


def _make_app(**kwargs):
    defaults = dict(members=5, land_w=10, land_h=10, seed=42, moderator_keys={"mod1"})
    defaults.update(kwargs)
    app = build_app(**defaults)
    return app


def _register_agent(client, name="Bot"):
    resp = client.post("/v1/agents/register", json={"name": name})
    assert resp.status_code == 200
    return resp.json()


def _open_submissions(app, round_id=None, pace=60.0):
    kernel = app.state.leviathan["kernel"]
    rs = app.state.leviathan["round_state"]
    if kernel.round_id == 0:
        kernel.begin_round()
    if round_id is None:
        round_id = kernel.round_id
    rs.open_submissions(round_id=round_id, pace=pace)


def _propose(client, api_key, code, description="test", idem_key="m-1"):
    return client.post(
        "/v1/world/mechanisms/propose",
        json={"code": code, "description": description, "idempotency_key": idem_key},
        headers={"X-API-Key": api_key},
    )


def _run_one_sim_round(app):
    """Simulate one round: close submissions, canary test, resolve votes, settle.

    This replicates the sim loop logic without threading, for deterministic tests.
    """
    import dataclasses
    from api.models import EventEnvelope
    from kernel.canary import CanaryRunner
    from kernel.subprocess_sandbox import SubprocessSandbox
    from kernel.execution_sandbox import SandboxContext

    kernel = app.state.leviathan["kernel"]
    event_log = app.state.leviathan["event_log"]
    round_state = app.state.leviathan["round_state"]
    mechanism_registry = app.state.leviathan["mechanism_registry"]
    judge = app.state.leviathan["judge"]
    moderator = app.state.leviathan["moderator"]

    sandbox = SubprocessSandbox()
    canary_runner = CanaryRunner()

    # Begin round if not already started
    if kernel.round_id == 0:
        kernel.begin_round()

    # Close submissions
    round_state.close_submissions()

    # Canary test pending proposals
    proposals = round_state.drain_proposals()
    judge_results = []

    for pp in proposals:
        matching_rec = None
        for rec in mechanism_registry.get_pending():
            if rec.proposer_id == pp.member_id and rec.code == pp.code:
                matching_rec = rec
                break
        if matching_rec is None:
            continue

        report = canary_runner.run_canary(
            execution_engine=kernel._execution,
            mechanism_code=pp.code,
            proposal_id=matching_rec.mechanism_id,
            proposer_id=pp.member_id,
            judge=judge,
        )
        mechanism_registry.mark_canary_result(matching_rec.mechanism_id, report.to_dict())

    # Resolve votes
    living_count = len(kernel._execution.current_members)
    approved_recs, rejected_recs = mechanism_registry.resolve_votes(living_count, kernel.round_id)

    for rec in approved_recs + rejected_recs:
        judge_results.append({
            "proposer_id": rec.proposer_id,
            "approved": rec.status == "approved",
            "reason": rec.judge_reason,
            "latency_ms": 0.0,
            "proposal_id": rec.mechanism_id,
        })

    # Execute approved mechanisms
    for rec in approved_recs:
        try:
            ctx = SandboxContext(execution_engine=kernel._execution, member_index=0)
            sandbox_result = sandbox.execute_mechanism_code(rec.code, ctx)
            if sandbox_result.success:
                mechanism_registry.activate(rec.mechanism_id, kernel.round_id)
        except Exception:
            pass

    # Execute pending actions
    pending = round_state.drain_actions()
    for pa in pending:
        member_index = kernel._resolve_agent_index(pa.member_id)
        if member_index is not None:
            ctx = SandboxContext(execution_engine=kernel._execution, member_index=member_index)
            result = sandbox.execute_agent_code(pa.code, ctx)
            if result.success:
                kernel.apply_intended_actions(result.intended_actions)

    moderator.store_snapshot(kernel.get_snapshot())

    receipt = kernel.settle_round(
        seed=kernel.round_id,
        judge_results=judge_results,
        mechanism_proposals=len(proposals),
        mechanism_approvals=len(approved_recs),
    )
    round_state.mark_settled()


# ── Tests ────────────────────────────────────────────


class TestCanaryPipeline:
    def test_proposal_gets_canary_tested(self):
        """Submit clean mechanism, run one sim round, verify canary_report and pending_vote."""
        app = _make_app()
        client = TestClient(app)
        agent = _register_agent(client)

        _open_submissions(app, round_id=1, pace=0.5)
        code = "def propose_modification(execution_engine):\n    pass\n"
        resp = _propose(client, agent["api_key"], code)
        assert resp.status_code == 200
        mech_id = resp.json()["mechanism_id"]
        assert mech_id != ""

        # Close submissions and run sim round
        _run_one_sim_round(app)

        # Verify canary report populated and status is pending_vote (not auto-approved)
        resp = client.get(f"/v1/world/mechanisms/{mech_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["canary_report"] is not None
        assert data["status"] == "pending_vote"

    def test_broken_proposal_auto_rejected(self):
        """Submit syntax-error mechanism, verify status rejected."""
        app = _make_app()
        client = TestClient(app)
        agent = _register_agent(client)

        _open_submissions(app, round_id=1, pace=0.5)
        code = "def propose_modification(execution_engine):\n    raise RuntimeError('boom')\n"
        resp = _propose(client, agent["api_key"], code)
        mech_id = resp.json()["mechanism_id"]

        _run_one_sim_round(app)

        resp = client.get(f"/v1/world/mechanisms/{mech_id}")
        data = resp.json()
        assert data["status"] == "rejected"
        assert "Canary execution error" in data["judge_reason"]

    def test_flagged_proposal_stays_pending(self):
        """Submit mechanism that kills agents (lethal), verify flagged status."""
        app = _make_app()
        client = TestClient(app)
        agent = _register_agent(client)

        _open_submissions(app, round_id=1, pace=0.5)
        # This mechanism sets all vitality to 0, causing a massive vitality drop
        code = (
            "def propose_modification(execution_engine):\n"
            "    for m in execution_engine.current_members:\n"
            "        m.vitality = 0.0\n"
        )
        resp = _propose(client, agent["api_key"], code)
        mech_id = resp.json()["mechanism_id"]

        _run_one_sim_round(app)

        resp = client.get(f"/v1/world/mechanisms/{mech_id}")
        data = resp.json()
        # Should be flagged by canary, then stay pending (no votes yet to resolve)
        assert data["status"] in ("canary_flagged", "pending_vote")
        assert data["canary_report"] is not None
        assert len(data["canary_report"].get("divergence_flags", [])) > 0


class TestVoteEndpoint:
    def test_vote_endpoint_records_vote(self):
        """Register 2 agents, submit flagged proposal, POST vote, verify response."""
        app = _make_app()
        client = TestClient(app)
        agent1 = _register_agent(client, "Bot1")
        agent2 = _register_agent(client, "Bot2")

        # Submit a flagged proposal (lethal mechanism)
        _open_submissions(app, round_id=1, pace=0.5)
        code = (
            "def propose_modification(execution_engine):\n"
            "    for m in execution_engine.current_members:\n"
            "        m.vitality = 0.0\n"
        )
        resp = _propose(client, agent1["api_key"], code)
        mech_id = resp.json()["mechanism_id"]

        _run_one_sim_round(app)

        # Now vote on the flagged mechanism
        resp = client.post(
            f"/v1/world/mechanisms/{mech_id}/vote",
            json={"vote": True, "idempotency_key": "v1"},
            headers={"X-API-Key": agent2["api_key"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["vote_recorded"] is True
        assert data["current_votes"]["yes"] == 1

    def test_vote_endpoint_auth_required(self):
        """Verify 401 without API key."""
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/world/mechanisms/nonexistent/vote",
            json={"vote": True, "idempotency_key": "v1"},
        )
        assert resp.status_code == 401

    def test_vote_on_non_votable_returns_400(self):
        """Submit a mechanism but don't run canary — voting should fail."""
        app = _make_app()
        client = TestClient(app)
        agent = _register_agent(client)

        _open_submissions(app, round_id=1, pace=60.0)
        code = "def propose_modification(e): pass"
        resp = _propose(client, agent["api_key"], code)
        mech_id = resp.json()["mechanism_id"]

        # Try voting without canary run — mechanism is still "submitted"
        resp = client.post(
            f"/v1/world/mechanisms/{mech_id}/vote",
            json={"vote": True, "idempotency_key": "v1"},
            headers={"X-API-Key": agent["api_key"]},
        )
        assert resp.status_code == 400


class TestVoteResolution:
    def test_majority_vote_approves(self):
        """Flag a proposal, cast majority yes votes, resolve, verify approved."""
        app = _make_app(members=5)
        client = TestClient(app)

        # Register 3 agents (majority of 5 is 3)
        agents = [_register_agent(client, f"Bot{i}") for i in range(3)]

        # Submit lethal proposal
        _open_submissions(app, round_id=1, pace=0.5)
        code = (
            "def propose_modification(execution_engine):\n"
            "    for m in execution_engine.current_members:\n"
            "        m.vitality = 0.0\n"
        )
        resp = _propose(client, agents[0]["api_key"], code)
        mech_id = resp.json()["mechanism_id"]

        _run_one_sim_round(app)

        # Cast majority yes votes
        for ag in agents:
            resp = client.post(
                f"/v1/world/mechanisms/{mech_id}/vote",
                json={"vote": True, "idempotency_key": f"v-{ag['agent_id']}"},
                headers={"X-API-Key": ag["api_key"]},
            )
            assert resp.status_code == 200

        # Resolve votes manually
        mechanism_registry = app.state.leviathan["mechanism_registry"]
        kernel = app.state.leviathan["kernel"]
        living_count = len(kernel._execution.current_members)
        approved, rejected = mechanism_registry.resolve_votes(living_count, kernel.round_id)

        assert len(approved) == 1
        assert approved[0].mechanism_id == mech_id
        assert "Approved by vote" in approved[0].judge_reason

    def test_majority_vote_rejects(self):
        """Cast majority no votes, verify rejected."""
        app = _make_app(members=5)
        client = TestClient(app)

        agents = [_register_agent(client, f"Bot{i}") for i in range(3)]

        _open_submissions(app, round_id=1, pace=0.5)
        code = (
            "def propose_modification(execution_engine):\n"
            "    for m in execution_engine.current_members:\n"
            "        m.vitality = 0.0\n"
        )
        resp = _propose(client, agents[0]["api_key"], code)
        mech_id = resp.json()["mechanism_id"]

        _run_one_sim_round(app)

        # Cast majority no votes
        for ag in agents:
            client.post(
                f"/v1/world/mechanisms/{mech_id}/vote",
                json={"vote": False, "idempotency_key": f"v-{ag['agent_id']}"},
                headers={"X-API-Key": ag["api_key"]},
            )

        mechanism_registry = app.state.leviathan["mechanism_registry"]
        kernel = app.state.leviathan["kernel"]
        living_count = len(kernel._execution.current_members)
        approved, rejected = mechanism_registry.resolve_votes(living_count, kernel.round_id)

        assert len(rejected) == 1
        assert rejected[0].mechanism_id == mech_id
        assert "Rejected by vote" in rejected[0].judge_reason


class TestMechanismDetail:
    def test_mechanism_detail_shows_canary_and_votes(self):
        """GET /{id} returns canary_report, votes_yes/no, vote_threshold."""
        app = _make_app()
        client = TestClient(app)
        agent = _register_agent(client)

        _open_submissions(app, round_id=1, pace=0.5)
        code = "def propose_modification(execution_engine):\n    pass\n"
        resp = _propose(client, agent["api_key"], code)
        mech_id = resp.json()["mechanism_id"]

        _run_one_sim_round(app)

        resp = client.get(f"/v1/world/mechanisms/{mech_id}")
        data = resp.json()
        assert "canary_report" in data
        assert "votes_yes" in data
        assert "votes_no" in data
        assert "vote_threshold" in data
        assert data["vote_threshold"] > 0

    def test_list_mechanisms_pending_vote_filter(self):
        """GET /mechanisms?status=pending_vote returns flagged mechanisms."""
        app = _make_app()
        client = TestClient(app)
        agent = _register_agent(client)

        # Submit a flagged proposal
        _open_submissions(app, round_id=1, pace=0.5)
        code = (
            "def propose_modification(execution_engine):\n"
            "    for m in execution_engine.current_members:\n"
            "        m.vitality = 0.0\n"
        )
        _propose(client, agent["api_key"], code)

        _run_one_sim_round(app)

        resp = client.get("/v1/world/mechanisms?status=pending_vote")
        assert resp.status_code == 200
        data = resp.json()
        # Should have the flagged mechanism
        assert len(data) >= 1
        assert data[0]["status"] in ("canary_flagged", "pending_vote")


class TestExistingTests:
    def test_existing_api_tests_still_pass(self):
        """Verify basic API functionality still works by running a quick sanity check."""
        app = _make_app()
        client = TestClient(app)

        # Health check
        assert client.get("/health").status_code == 200

        # World info
        resp = client.get("/v1/world")
        assert resp.status_code == 200
        assert resp.json()["member_count"] == 5

        # Discovery
        resp = client.get("/.well-known/leviathan-agent.json")
        assert resp.status_code == 200

        # Register and submit action
        agent = _register_agent(client)
        _open_submissions(app)
        resp = client.post(
            "/v1/world/actions",
            json={"code": "def agent_action(e, m): pass", "idempotency_key": "k1"},
            headers={"X-API-Key": agent["api_key"]},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"
