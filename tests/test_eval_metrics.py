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
            "attempts": [
                {"judge": {"approved": True, "reason": "ok"}},
                {"judge": {"approved": False, "reason": "Unrealistic physics"}},
            ],
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
    assert metrics.get("mechanism_judge_approved_count") == 1
    assert metrics.get("mechanism_judge_rejected_count") == 1
    assert metrics.get("mechanism_judge_missing_count") == 0
    reason_counts = metrics.get("mechanism_judge_rejection_reason_counts") or {}
    assert reason_counts.get("Unrealistic physics") == 1
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


def test_fallback_metrics_syntax_error_positions():
    round_record = {
        "errors": {
            "agent_code_errors": [
                {
                    "error_category": "llm_syntax_error",
                    "error_details": {
                        "error_type": "SyntaxError",
                        "error_line": 9,
                        "code_line_count": 10,
                        "error_text": "// trailing note",
                    },
                    "llm_metadata": {
                        "finish_reason": "length",
                        "request_max_tokens": 200,
                        "completion_tokens": 200,
                    },
                },
                {
                    "error_category": "llm_syntax_error",
                    "error_details": {
                        "error_type": "SyntaxError",
                        "error_line": 2,
                        "code_line_count": 10,
                        "error_text": "if member_id not self.inventories:",
                    },
                    "llm_metadata": {
                        "finish_reason": "stop",
                        "request_max_tokens": 200,
                        "completion_tokens": 150,
                    },
                },
                {
                    "error_category": "llm_syntax_error",
                    "llm_metadata": {
                        "finish_reason": "",
                        "request_max_tokens": 300,
                        "completion_tokens": 300,
                    },
                },
            ],
            "mechanism_errors": [
                {
                    "error_category": "llm_syntax_error",
                    "error_details": {
                        "error_type": "SyntaxError",
                        "error_line": 1,
                        "code_line_count": 3,
                        "error_text": "I'll analyze the situation before writing code.",
                    },
                    "llm_metadata": {
                        "finish_reason": "stop",
                        "request_max_completion_tokens": 100,
                        "completion_tokens": 80,
                    },
                }
            ],
        }
    }

    metrics, sources = fallback_metrics(round_record)

    assert metrics.get("llm_syntax_error_count") == 4
    assert metrics.get("llm_syntax_error_near_end_count") == 2
    assert metrics.get("llm_syntax_error_near_end_10pct_count") == 1
    assert metrics.get("llm_syntax_error_mid_count") == 1
    assert metrics.get("llm_syntax_error_unknown_count") == 1
    assert metrics.get("llm_syntax_error_line_ratio_samples") == 3
    assert metrics.get("llm_syntax_error_line_ratio_min") == pytest.approx(0.2)
    assert metrics.get("llm_syntax_error_line_ratio_max") == pytest.approx(0.9)
    assert metrics.get("llm_syntax_error_line_ratio_avg") == pytest.approx((0.9 + 0.2 + (1 / 3)) / 3)
    assert metrics.get("llm_syntax_error_near_end_rate") == pytest.approx(2 / 3)
    assert metrics.get("llm_syntax_error_near_end_10pct_rate") == pytest.approx(1 / 3)
    assert metrics.get("llm_syntax_error_js_comment_count") == 1
    assert metrics.get("llm_syntax_error_non_code_prefix_count") == 1
    assert metrics.get("llm_syntax_error_missing_in_count") == 1
    assert metrics.get("llm_syntax_error_finish_reason_counts") == {"length": 1, "stop": 2}
    assert metrics.get("llm_syntax_error_finish_reason_total") == 4
    assert metrics.get("llm_syntax_error_finish_reason_length_count") == 1
    assert metrics.get("llm_syntax_error_finish_reason_missing_count") == 1
    assert metrics.get("llm_syntax_error_request_cap_count") == 4
    assert metrics.get("llm_syntax_error_completion_at_request_cap_count") == 2
    assert metrics.get("llm_syntax_error_completion_at_request_cap_rate") == pytest.approx(0.5)
    assert metrics.get("llm_syntax_error_agent_count") == 3
    assert metrics.get("llm_syntax_error_agent_near_end_count") == 1
    assert metrics.get("llm_syntax_error_agent_near_end_10pct_count") == 1
    assert metrics.get("llm_syntax_error_agent_js_comment_count") == 1
    assert metrics.get("llm_syntax_error_agent_non_code_prefix_count") == 0
    assert metrics.get("llm_syntax_error_agent_missing_in_count") == 1
    assert metrics.get("llm_syntax_error_mechanism_count") == 1
    assert metrics.get("llm_syntax_error_mechanism_near_end_count") == 1
    assert metrics.get("llm_syntax_error_mechanism_near_end_10pct_count") == 0
    assert metrics.get("llm_syntax_error_mechanism_js_comment_count") == 0
    assert metrics.get("llm_syntax_error_mechanism_non_code_prefix_count") == 1
    assert metrics.get("llm_syntax_error_mechanism_missing_in_count") == 0
    assert sources.get("llm_syntax_error_near_end_rate") == (
        "errors/agent_code_errors+mechanism_errors"
    )
    assert sources.get("llm_syntax_error_near_end_10pct_rate") == (
        "errors/agent_code_errors+mechanism_errors"
    )
    assert sources.get("llm_syntax_error_missing_in_count") == (
        "errors/agent_code_errors+mechanism_errors"
    )
    assert sources.get("llm_syntax_error_agent_count") == "errors/agent_code_errors"
    assert sources.get("llm_syntax_error_mechanism_count") == "errors/mechanism_errors"


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


def test_fallback_metrics_prompt_section_chars():
    round_record = {
        "generated_code": {
            "member_0": {
                "llm_metadata": {
                    "prompt_char_count": 400,
                    "prompt_section_chars": {
                        "base_code": 100,
                        "analysis_memory": 30,
                        "base_code_base_island": 60,
                        "base_code_base_land": 40,
                    },
                }
            },
            "member_1": {
                "llm_metadata": {
                    "prompt_char_count": 300,
                    "prompt_section_chars": {
                        "base_code": 50,
                        "report": 10,
                        "base_code_base_island": 30,
                        "base_code_base_land": 20,
                    },
                }
            },
        },
        "mechanism_modifications": {
            "attempts": [
                {
                    "llm_metadata": {
                        "prompt_char_count": 350,
                        "prompt_section_chars": {
                            "base_code": 70,
                            "report": 20,
                            "base_code_base_island": 50,
                            "base_code_base_land": 20,
                        },
                    }
                }
            ]
        },
    }

    metrics, sources = fallback_metrics(round_record)

    totals = metrics.get("llm_prompt_section_chars_total") or {}
    averages = metrics.get("llm_prompt_section_chars_avg") or {}
    max_vals = metrics.get("llm_prompt_section_chars_max") or {}

    assert totals.get("base_code") == pytest.approx(220)
    assert totals.get("analysis_memory") == pytest.approx(30)
    assert totals.get("report") == pytest.approx(30)
    assert totals.get("base_code_base_island") == pytest.approx(140)
    assert totals.get("base_code_base_land") == pytest.approx(80)
    assert averages.get("base_code") == pytest.approx(220 / 3)
    assert averages.get("analysis_memory") == pytest.approx(30)
    assert averages.get("report") == pytest.approx(15)
    assert averages.get("base_code_base_island") == pytest.approx(140 / 3)
    assert averages.get("base_code_base_land") == pytest.approx(80 / 3)
    assert max_vals.get("base_code") == pytest.approx(100)
    assert max_vals.get("analysis_memory") == pytest.approx(30)
    assert max_vals.get("report") == pytest.approx(20)
    assert max_vals.get("base_code_base_island") == pytest.approx(60)
    assert max_vals.get("base_code_base_land") == pytest.approx(40)
    assert metrics.get("llm_prompt_section_entry_count") == 3
    assert metrics.get("llm_prompt_char_count_avg") == pytest.approx(350)
    assert metrics.get("llm_prompt_char_count_min") == pytest.approx(300)
    assert metrics.get("llm_prompt_char_count_max") == pytest.approx(400)
    assert metrics.get("llm_prompt_section_top_avg_key") == "base_code"
    assert metrics.get("llm_prompt_section_top_avg_chars") == pytest.approx(220 / 3)
    assert metrics.get("llm_prompt_section_top_base_code_key") == "base_code_base_island"
    assert metrics.get("llm_prompt_section_top_base_code_chars") == pytest.approx(140 / 3)
    assert metrics.get("llm_prompt_section_base_code_ratio") == pytest.approx((220 / 3) / 350)
    assert metrics.get("llm_prompt_section_top_avg_ratio") == pytest.approx((220 / 3) / 350)
    assert metrics.get("llm_prompt_section_top_base_code_ratio") == pytest.approx(140 / 220)
    assert sources.get("llm_prompt_section_chars_total") == (
        "generated_code/llm_metadata+mechanism_modifications/attempts"
    )
    assert sources.get("llm_prompt_char_count_avg") == (
        "generated_code/llm_metadata+mechanism_modifications/attempts"
    )
    assert sources.get("llm_prompt_section_top_avg_key") == (
        "generated_code/llm_metadata+mechanism_modifications/attempts"
    )
    assert sources.get("llm_prompt_section_base_code_ratio") == (
        "generated_code/llm_metadata+mechanism_modifications/attempts"
    )
    assert sources.get("llm_prompt_section_top_avg_ratio") == (
        "generated_code/llm_metadata+mechanism_modifications/attempts"
    )
    assert sources.get("llm_prompt_section_top_base_code_ratio") == (
        "generated_code/llm_metadata+mechanism_modifications/attempts"
    )


def test_fallback_metrics_analysis_card_signature_quality():
    round_record = {
        "analysis_cards": {
            "0": {
                "baseline_signature": ["expand", "offer"],
                "variation_signature": ["contracts"],
            },
            "1": {
                "baseline_signature": ["[expand", "resources]"],
                "variation_signature": ["benefit", "message]"],
            },
        }
    }

    metrics, sources = fallback_metrics(round_record)

    assert metrics.get("analysis_card_signature_card_count") == 2
    assert metrics.get("analysis_card_signature_tag_total") == 7
    assert metrics.get("analysis_card_signature_invalid_tag_count") == 1
    assert metrics.get("analysis_card_signature_recoverable_tag_count") == 3
    assert metrics.get("analysis_card_signature_invalid_tag_rate") == pytest.approx(1 / 7)
    assert metrics.get("analysis_card_signature_recoverable_tag_rate") == pytest.approx(3 / 7)
    assert metrics.get("analysis_card_signature_empty_card_count") == 0
    assert metrics.get("analysis_card_signature_empty_card_rate") == pytest.approx(0.0)
    assert sources.get("analysis_card_signature_tag_total") == "analysis_cards"


def test_fallback_metrics_analysis_card_signature_parses_action_dicts():
    round_record = {
        "analysis_cards": {
            "0": {
                "baseline_signature": ["{'actions': ['offer', 'expand'], 'details': 'x'}"],
                "variation_signature": ["attack+offer_land"],
            },
            "1": {
                "baseline_signature": {"actions": ["message", "contracts"]},
                "variation_signature": ["expand|offer"],
            },
        }
    }

    metrics, _ = fallback_metrics(round_record)

    assert metrics.get("analysis_card_signature_card_count") == 2
    assert metrics.get("analysis_card_signature_tag_total") == 8
    assert metrics.get("analysis_card_signature_invalid_tag_count") == 0
    assert metrics.get("analysis_card_signature_recoverable_tag_count") == 0
    assert metrics.get("analysis_card_signature_invalid_tag_rate") == pytest.approx(0.0)
    assert metrics.get("analysis_card_signature_empty_card_count") == 0
