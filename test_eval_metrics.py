import pytest

from utils.eval_metrics import fallback_metrics


def test_fallback_metrics_survival_std_and_error_tags():
    round_record = {
        "round_end_deltas": {
            "member_a": {"survival_chance": 0.1},
            "member_b": {"survival_chance": 0.3},
        },
        "errors": {
            "agent_code_errors": [
                {
                    "code": (
                        "def agent_action(execution_engine, member_id):\n"
                        "    execution_engine.attack(member_id, 1)\n"
                    )
                }
            ]
        },
    }

    metrics, _ = fallback_metrics(round_record)

    assert metrics.get("population_avg_survival_delta") == pytest.approx(0.2)
    assert metrics.get("population_std_survival_delta") == pytest.approx(0.1)
    assert metrics.get("agent_code_error_count") == 1
    assert metrics.get("agent_code_error_rate") == pytest.approx(0.5)
    tag_counts = metrics.get("agent_code_error_tag_counts") or {}
    assert tag_counts.get("attack") == 1


def test_fallback_metrics_llm_connection_error_tags():
    round_record = {
        "errors": {
            "agent_code_errors": [
                {
                    "error": "An error occurred: Connection error.",
                    "traceback": "httpx.ConnectError: [Errno 8] nodename nor servname provided",
                    "code": "",
                }
            ]
        }
    }

    metrics, _ = fallback_metrics(round_record)

    assert metrics.get("agent_code_error_count") == 1
    tag_counts = metrics.get("agent_code_error_tag_counts") or {}
    assert tag_counts.get("llm_connection_error") == 1


def test_fallback_metrics_mechanism_contract_physics():
    round_record = {
        "agent_actions": [
            {"member_id": 0, "code_executed": "execution_engine.attack(0, 1)", "performance_change": 0.1},
            {"member_id": 1, "code_executed": "execution_engine.bear(1)", "performance_change": 0.2},
        ],
        "mechanism_modifications": {
            "attempts": [{}, {}],
            "executed": [{}],
            "approved_ids": [0],
        },
        "errors": {
            "mechanism_errors": [{}, {}],
            "agent_code_errors": [],
        },
        "contract_stats": {
            "total_contracts": 4,
            "pending": 1,
            "active": 2,
            "completed": 1,
            "failed": 0,
        },
        "contract_partner_counts": {"0": 2, "1": 1},
        "contract_partner_top_share": {"0": 0.6, "1": 0.5},
        "physics_stats": {"active_constraints": 3, "domains": ["trade", "resource"]},
    }

    metrics, _ = fallback_metrics(round_record)

    assert metrics.get("plan_alignment_total_actions") == 2
    assert metrics.get("mechanism_attempted_count") == 2
    assert metrics.get("mechanism_approved_count") == 1
    assert metrics.get("mechanism_executed_count") == 1
    assert metrics.get("mechanism_error_count") == 2
    assert metrics.get("contract_total") == 4
    assert metrics.get("contract_pending") == 1
    assert metrics.get("contract_active") == 2
    assert metrics.get("contract_completed") == 1
    assert metrics.get("contract_failed") == 0
    assert metrics.get("contract_partner_unique_avg") == pytest.approx(1.5)
    assert metrics.get("contract_partner_top_share_avg") == pytest.approx(0.55)
    assert metrics.get("physics_active_constraints") == 3
    assert metrics.get("physics_domain_count") == 2


def test_fallback_metrics_mechanism_implicit_approvals():
    round_record = {
        "mechanism_modifications": {
            "attempts": [{}, {}],
            "executed": [{}],
        }
    }

    metrics, sources = fallback_metrics(round_record)

    assert metrics.get("mechanism_attempted_count") == 2
    assert metrics.get("mechanism_approved_count") == 2
    assert metrics.get("mechanism_executed_count") == 1
    assert sources.get("mechanism_approved_count", "").startswith("mechanism_modifications:implicit")
