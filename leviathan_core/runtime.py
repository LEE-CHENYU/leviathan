"""Minimal runtime for executing plugin-defined simulations."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from leviathan_core.graph import PhaseGraph
from leviathan_core.protocols import (
    ArtifactStore,
    EventSink,
    LLMRunner,
    PhaseResult,
    PhaseSpec,
    RoundContext,
    SimulationPlugin,
)


class NullEventSink:
    """Default event sink that drops all events."""

    def emit(self, event: dict[str, Any]) -> None:
        return None


class InMemoryArtifactStore:
    """Simple artifact sink for tests and early integration work."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def save(
        self,
        *,
        round_ctx: RoundContext,
        phase_id: str,
        name: str,
        data: Any,
    ) -> None:
        self.records.append(
            {
                "round_index": round_ctx.round_index,
                "phase_id": phase_id,
                "name": name,
                "data": data,
            }
        )


class SimulationRuntime:
    """Runs a plugin-defined phase graph for a fixed number of rounds."""

    def __init__(
        self,
        plugin: SimulationPlugin,
        llm_runner: LLMRunner | None = None,
        artifact_store: ArtifactStore | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.plugin = plugin
        self.llm_runner = llm_runner
        self.artifact_store = artifact_store or InMemoryArtifactStore()
        self.event_sink = event_sink or NullEventSink()

    async def run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Execute the plugin graph round by round."""

        state = self.plugin.build_initial_state(config)
        graph = PhaseGraph(self.plugin.build_phase_graph(config))
        stop_condition = self.plugin.get_stop_condition()
        finalizer = self.plugin.get_finalizer()
        max_rounds = int(config.get("rounds", 1))
        history: list[dict[str, Any]] = []

        for round_index in range(max_rounds):
            phase_outputs: dict[str, dict[str, Any]] = {}
            latest_actions: dict[str, dict[str, Any]] = {}

            for layer in graph.layers:
                for phase in layer:
                    round_ctx = RoundContext(
                        round_index=round_index,
                        config=config,
                        metadata={"phase_outputs": phase_outputs},
                        phase_id=phase.phase_id,
                    )
                    result = await self._execute_phase(
                        phase=phase,
                        state=state,
                        latest_actions=latest_actions,
                        round_ctx=round_ctx,
                    )
                    phase_outputs[phase.phase_id] = result.outputs
                    if "actions" in result.outputs:
                        latest_actions = result.outputs["actions"]
                    if "state" in result.outputs:
                        state = result.outputs["state"]

            history.append(
                {
                    "round_index": round_index,
                    "phases": phase_outputs,
                }
            )

            if stop_condition:
                stop_ctx = RoundContext(
                    round_index=round_index,
                    config=config,
                    metadata={"history": history},
                )
                if stop_condition.should_stop(state, stop_ctx):
                    break

        final_output = None
        if finalizer is not None:
            final_output = await finalizer.finalize(
                state=state,
                history=history,
                config=config,
                llm_runner=self.llm_runner,
            )

        return {
            "plugin_id": self.plugin.plugin_id,
            "rounds_completed": len(history),
            "state": state,
            "history": history,
            "final": final_output,
        }

    async def _execute_phase(
        self,
        *,
        phase: PhaseSpec,
        state: Any,
        latest_actions: dict[str, dict[str, Any]],
        round_ctx: RoundContext,
    ) -> PhaseResult:
        self.event_sink.emit(
            {
                "type": "phase_started",
                "phase_id": phase.phase_id,
                "kind": phase.kind,
                "round_index": round_ctx.round_index,
            }
        )

        if phase.kind == "action":
            outputs = await self._run_action_phase(phase, state, round_ctx)
        elif phase.kind == "judge":
            outputs = self._run_judge_phase(phase, state, latest_actions, round_ctx)
        elif phase.kind == "reduce":
            outputs = self._run_reduce_phase(phase, state, latest_actions, round_ctx)
        elif phase.kind == "evaluate":
            outputs = await self._run_evaluate_phase(
                phase, state, latest_actions, round_ctx
            )
        else:
            raise ValueError(f"Unsupported phase kind: {phase.kind}")

        self.artifact_store.save(
            round_ctx=round_ctx,
            phase_id=phase.phase_id,
            name="result",
            data=outputs,
        )
        self.event_sink.emit(
            {
                "type": "phase_completed",
                "phase_id": phase.phase_id,
                "kind": phase.kind,
                "round_index": round_ctx.round_index,
            }
        )
        return PhaseResult(phase_id=phase.phase_id, kind=phase.kind, outputs=outputs)

    async def _run_action_phase(
        self,
        phase: PhaseSpec,
        state: Any,
        round_ctx: RoundContext,
    ) -> dict[str, Any]:
        if self.llm_runner is None:
            raise ValueError("Action phases require an llm_runner")

        observation_builder = self.plugin.get_observation_builder(phase.phase_id)
        parser = self.plugin.get_action_parser(phase.phase_id)
        if observation_builder is None or parser is None:
            raise ValueError(f"Action phase '{phase.phase_id}' is missing a builder or parser")

        roles = self.plugin.get_roles(state)
        action_concurrency = self._phase_concurrency_limit(round_ctx, phase.kind)
        actor_results = await self._run_actor_batch(
            roles=roles,
            phase=phase,
            state=state,
            round_ctx=round_ctx,
            observation_builder=observation_builder,
            parser=parser,
            concurrency_limit=action_concurrency,
        )

        actions = {actor_id: action for actor_id, _, action in actor_results}
        raw_outputs = {actor_id: raw for actor_id, raw, _ in actor_results}
        return {"actions": actions, "raw_outputs": raw_outputs}

    async def _run_actor_batch(
        self,
        *,
        roles,
        phase: PhaseSpec,
        state: Any,
        round_ctx: RoundContext,
        observation_builder,
        parser,
        concurrency_limit: int | None,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        if concurrency_limit is None:
            tasks = [
                self._run_actor(
                    actor_id=role.role_id,
                    phase=phase,
                    state=state,
                    round_ctx=round_ctx,
                    observation_builder=observation_builder,
                    parser=parser,
                )
                for role in roles
            ]
            return await asyncio.gather(*tasks)

        semaphore = asyncio.Semaphore(concurrency_limit)

        async def _bounded_run(role):
            async with semaphore:
                return await self._run_actor(
                    actor_id=role.role_id,
                    phase=phase,
                    state=state,
                    round_ctx=round_ctx,
                    observation_builder=observation_builder,
                    parser=parser,
                )

        tasks = [_bounded_run(role) for role in roles]
        return await asyncio.gather(*tasks)

    async def _run_actor(
        self,
        *,
        actor_id: str,
        phase: PhaseSpec,
        state: Any,
        round_ctx: RoundContext,
        observation_builder,
        parser,
    ) -> tuple[str, str, dict[str, Any]]:
        observation = observation_builder.build(state, actor_id, round_ctx)
        prompt = self._render_prompt(
            parser.prompt_schema(actor_id, phase.phase_id),
            observation,
        )
        raw_output = await self.llm_runner.run(
            prompt,
            actor_id=actor_id,
            phase_id=phase.phase_id,
            round_ctx=round_ctx,
        )
        try:
            parsed = parser.parse(raw_output, actor_id, phase.phase_id)
        except Exception:
            parsed = parser.fallback(actor_id, phase.phase_id)
        return actor_id, raw_output, parsed

    def _run_judge_phase(
        self,
        phase: PhaseSpec,
        state: Any,
        latest_actions: dict[str, dict[str, Any]],
        round_ctx: RoundContext,
    ) -> dict[str, Any]:
        judge = self.plugin.get_judge(phase.phase_id)
        if judge is None:
            raise ValueError(f"Judge phase '{phase.phase_id}' is missing a judge")
        adjusted_actions, info = judge.validate(state, latest_actions, round_ctx)
        return {"actions": adjusted_actions, "judge": info}

    def _run_reduce_phase(
        self,
        phase: PhaseSpec,
        state: Any,
        latest_actions: dict[str, dict[str, Any]],
        round_ctx: RoundContext,
    ) -> dict[str, Any]:
        reducer = self.plugin.get_reducer(phase.phase_id)
        if reducer is None:
            raise ValueError(f"Reduce phase '{phase.phase_id}' is missing a reducer")
        new_state = reducer.apply(state, latest_actions, round_ctx)
        return {"state": new_state}

    async def _run_evaluate_phase(
        self,
        phase: PhaseSpec,
        state: Any,
        latest_actions: dict[str, dict[str, Any]],
        round_ctx: RoundContext,
    ) -> dict[str, Any]:
        evaluator = self.plugin.get_evaluator(phase.phase_id)
        if evaluator is None:
            raise ValueError(
                f"Evaluate phase '{phase.phase_id}' is missing an evaluator"
            )
        evaluation = await evaluator.evaluate(state, latest_actions, round_ctx)
        return {"evaluation": evaluation}

    @staticmethod
    def _render_prompt(schema_text: str, observation: dict[str, Any]) -> str:
        return (
            f"{schema_text}\n\n"
            "Observation:\n"
            f"{json.dumps(observation, indent=2, default=str)}"
        )

    @staticmethod
    def _phase_concurrency_limit(round_ctx: RoundContext, phase_kind: str) -> int | None:
        config = round_ctx.config
        specific_key = f"{phase_kind}_concurrency"
        value = config.get(specific_key, config.get("max_concurrency"))
        if value is None:
            return None
        limit = int(value)
        return None if limit <= 0 else limit
