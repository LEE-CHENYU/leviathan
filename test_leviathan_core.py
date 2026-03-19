"""Smoke tests for the new leviathan_core scaffold."""

from __future__ import annotations

import asyncio
import json

from leviathan_core import (
    AgentRole,
    PhaseGraph,
    PhaseGraphSpec,
    PhaseSpec,
    PluginCapabilities,
    SimulationRuntime,
)


class CountingObservationBuilder:
    def build(self, state, actor_id, round_ctx):
        return {
            "actor_id": actor_id,
            "counter": state["counter"],
            "round": round_ctx.round_index,
        }


class JsonActionParser:
    def prompt_schema(self, actor_id, phase_id):
        return (
            f"Actor {actor_id} is in phase {phase_id}. "
            "Return a JSON object with a numeric delta field."
        )

    def parse(self, raw_text, actor_id, phase_id):
        return json.loads(raw_text)

    def fallback(self, actor_id, phase_id):
        return {"delta": 0}


class PassthroughJudge:
    def validate(self, state, actions, round_ctx):
        return actions, {"validated": len(actions)}


class CounterReducer:
    def apply(self, state, actions, round_ctx):
        new_state = dict(state)
        new_state["counter"] += sum(action.get("delta", 0) for action in actions.values())
        return new_state


class CounterEvaluator:
    async def evaluate(self, state, actions, round_ctx):
        return {"counter": state["counter"], "round": round_ctx.round_index}


class StubLLMRunner:
    async def run(self, prompt, actor_id, phase_id, round_ctx):
        return '{"delta": 1}'


class CountingPlugin:
    plugin_id = "counting"
    capabilities = PluginCapabilities(deterministic_judge=True)

    def build_initial_state(self, config):
        return {"counter": 0}

    def build_phase_graph(self, config):
        return PhaseGraphSpec(
            phases=[
                PhaseSpec("decide", "action"),
                PhaseSpec("validate", "judge", depends_on=["decide"]),
                PhaseSpec("apply", "reduce", depends_on=["validate"]),
                PhaseSpec("score", "evaluate", depends_on=["apply"]),
            ]
        )

    def get_roles(self, state):
        return [
            AgentRole("alpha", "Alpha", "tester"),
            AgentRole("beta", "Beta", "tester"),
        ]

    def get_observation_builder(self, phase_id):
        return CountingObservationBuilder() if phase_id == "decide" else None

    def get_action_parser(self, phase_id):
        return JsonActionParser() if phase_id == "decide" else None

    def get_judge(self, phase_id):
        return PassthroughJudge() if phase_id == "validate" else None

    def get_reducer(self, phase_id):
        return CounterReducer() if phase_id == "apply" else None

    def get_evaluator(self, phase_id):
        return CounterEvaluator() if phase_id == "score" else None

    def get_stop_condition(self):
        return None

    def get_finalizer(self):
        return None


def test_phase_graph_layers():
    graph = PhaseGraph(
        PhaseGraphSpec(
            phases=[
                PhaseSpec("start", "action"),
                PhaseSpec("judge", "judge", depends_on=["start"]),
                PhaseSpec("reduce", "reduce", depends_on=["judge"]),
                PhaseSpec("score", "evaluate", depends_on=["reduce"]),
            ]
        )
    )
    assert [[phase.phase_id for phase in layer] for layer in graph.layers] == [
        ["start"],
        ["judge"],
        ["reduce"],
        ["score"],
    ]


def test_runtime_executes_minimal_plugin():
    runtime = SimulationRuntime(
        plugin=CountingPlugin(),
        llm_runner=StubLLMRunner(),
    )
    result = asyncio.run(runtime.run({"rounds": 2}))

    assert result["plugin_id"] == "counting"
    assert result["rounds_completed"] == 2
    assert result["state"]["counter"] == 4
    assert result["history"][0]["phases"]["decide"]["actions"]["alpha"]["delta"] == 1
    assert result["history"][1]["phases"]["score"]["evaluation"]["counter"] == 4
