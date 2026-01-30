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


def test_fallback_metrics_llm_finish_reason_counts():
    round_record = {
        "generated_code": {
            "member_0": {"llm_metadata": {"finish_reason": "length"}},
            "member_1": {"llm_metadata": {"finish_reason": "stop"}},
        },
        "mechanism_modifications": {
            "attempts": [
                {"llm_metadata": {"finish_reason": "stop"}},
                {"llm_metadata_primary": {"finish_reason": "length"}},
                {"llm_metadata": {"model": "test-model"}},
            ]
        },
    }

    metrics, sources = fallback_metrics(round_record)

    counts = metrics.get("llm_finish_reason_counts") or {}
    assert counts.get("length") == 2
    assert counts.get("stop") == 2
    assert metrics.get("llm_finish_reason_total") == 5
    assert metrics.get("llm_finish_reason_length_count") == 2
    assert metrics.get("llm_finish_reason_missing_count") == 1
    assert sources.get("llm_finish_reason_total") == (
        "generated_code/llm_metadata+mechanism_modifications/attempts"
    )


def test_fallback_metrics_llm_token_caps():
    round_record = {
        "generated_code": {
            "member_0": {
                "llm_metadata": {
                    "finish_reason": "length",
                    "request_max_tokens": 1200,
                    "prompt_tokens": 300,
                    "completion_tokens": 1200,
                }
            },
            "member_1": {
                "llm_metadata": {
                    "finish_reason": "stop",
                    "request_max_tokens": 800,
                    "prompt_tokens": 200,
                    "completion_tokens": 500,
                }
            },
        },
        "mechanism_modifications": {
            "attempts": [
                {
                    "llm_metadata": {
                        "finish_reason": "length",
                        "request_max_completion_tokens": 600,
                        "prompt_tokens": 100,
                        "completion_tokens": 600,
                    }
                }
            ]
        },
    }

    metrics, sources = fallback_metrics(round_record)

    assert metrics.get("llm_request_cap_count") == 3
    assert metrics.get("llm_request_cap_min") == 600
    assert metrics.get("llm_request_cap_max") == 1200
    assert metrics.get("llm_request_cap_avg") == pytest.approx((1200 + 800 + 600) / 3)
    assert metrics.get("llm_request_cap_source_counts") == {
        "max_tokens": 2,
        "max_completion_tokens": 1,
    }
    assert metrics.get("llm_prompt_tokens_avg") == pytest.approx((300 + 200 + 100) / 3)
    assert metrics.get("llm_completion_tokens_avg") == pytest.approx((1200 + 500 + 600) / 3)
    assert metrics.get("llm_completion_at_request_cap_count") == 2
    assert metrics.get("llm_completion_at_request_cap_rate") == pytest.approx(2 / 3)
    assert sources.get("llm_request_cap_avg") == (
        "generated_code/llm_metadata+mechanism_modifications/attempts"
    )
