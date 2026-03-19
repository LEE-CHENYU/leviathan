"""Tests for the corporate-strategy demo plugin."""

from __future__ import annotations

import asyncio
import json

from leviathan_core import SimulationRuntime
from plugins.corporate_strategy import SimpleCorporateStrategyPlugin


class PerActorLLMRunner:
    """Returns deterministic JSON decisions for each company."""

    RESPONSES = {
        "AAA": {
            "capital_allocation": {
                "rd_budget": 90,
                "capex_budget": 60,
                "buyback_budget": 20,
            },
            "strategic_moves": ["Expand AI product line"],
            "rationale": "Increase investment while keeping spend inside budget.",
        },
        "BBB": {
            "capital_allocation": {
                "rd_budget": 40,
                "capex_budget": 90,
                "buyback_budget": 90,
            },
            "strategic_moves": ["Defend margins"],
            "rationale": "Aggressive spend that should force judge intervention.",
        },
    }

    async def run(self, prompt, actor_id, phase_id, round_ctx):
        return json.dumps(self.RESPONSES[actor_id])


def test_corporate_strategy_plugin_runs_on_kernel():
    plugin = SimpleCorporateStrategyPlugin()
    runtime = SimulationRuntime(plugin=plugin, llm_runner=PerActorLLMRunner())

    result = asyncio.run(runtime.run(plugin.default_config()))

    assert result["plugin_id"] == "corporate_strategy_demo"
    assert result["rounds_completed"] == 1

    judge_phase = result["history"][0]["phases"]["judge"]
    bbb_report = judge_phase["judge"]["BBB"]
    assert any(v["constraint"] == "budget" for v in bbb_report["violations"])
    bbb_alloc = judge_phase["actions"]["BBB"]["capital_allocation"]
    assert bbb_alloc["buyback_budget"] < 90.0
    assert (
        bbb_alloc["rd_budget"]
        + bbb_alloc["capex_budget"]
        + bbb_alloc["buyback_budget"]
        <= bbb_report["available_budget"]
    )

    state = result["state"]
    assert state["companies"]["AAA"].revenue > 1000.0
    assert state["companies"]["BBB"].cash >= 0.0

    score_phase = result["history"][0]["phases"]["score"]["evaluation"]
    assert set(score_phase) == {"AAA", "BBB"}
    assert score_phase["AAA"]["weighted_score"] >= score_phase["BBB"]["weighted_score"]
