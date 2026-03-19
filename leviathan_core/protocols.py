"""Core protocols and dataclasses for domain plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class AgentRole:
    """Declares an actor that can participate in one or more phases."""

    role_id: str
    label: str
    kind: str
    allow_hidden_actions: bool = True


@dataclass(frozen=True)
class PluginCapabilities:
    """Optional engine capabilities used by a plugin."""

    contracts: bool = False
    mechanism_proposals: bool = False
    llm_judge: bool = False
    deterministic_judge: bool = False
    scoring_panel: bool = False


@dataclass(frozen=True)
class PhaseSpec:
    """A single node in the simulation phase graph."""

    phase_id: str
    kind: str
    depends_on: list[str] = field(default_factory=list)
    hidden_actions: bool = False


@dataclass(frozen=True)
class PhaseGraphSpec:
    """A declarative phase DAG."""

    phases: list[PhaseSpec]


@dataclass
class RoundContext:
    """Mutable execution context for a round and phase."""

    round_index: int
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    phase_id: str | None = None


@dataclass
class PhaseResult:
    """Standard container for phase outputs."""

    phase_id: str
    kind: str
    outputs: dict[str, Any]


@runtime_checkable
class LLMRunner(Protocol):
    """Executes model calls for action phases."""

    async def run(
        self,
        prompt: str,
        actor_id: str,
        phase_id: str,
        round_ctx: RoundContext,
    ) -> str: ...


@runtime_checkable
class EventSink(Protocol):
    """Receives structured runtime events."""

    def emit(self, event: dict[str, Any]) -> None: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Persists per-phase artifacts."""

    def save(
        self,
        *,
        round_ctx: RoundContext,
        phase_id: str,
        name: str,
        data: Any,
    ) -> None: ...


@runtime_checkable
class ObservationBuilder(Protocol):
    """Builds the actor-visible input for an action phase."""

    def build(
        self,
        state: Any,
        actor_id: str,
        round_ctx: RoundContext,
    ) -> dict[str, Any]: ...


@runtime_checkable
class ActionParser(Protocol):
    """Defines the response contract for an action phase."""

    def prompt_schema(self, actor_id: str, phase_id: str) -> str: ...

    def parse(
        self,
        raw_text: str,
        actor_id: str,
        phase_id: str,
    ) -> dict[str, Any]: ...

    def fallback(self, actor_id: str, phase_id: str) -> dict[str, Any]: ...


@runtime_checkable
class JudgeAdapter(Protocol):
    """Validates and optionally rewrites actions."""

    def validate(
        self,
        state: Any,
        actions: dict[str, dict[str, Any]],
        round_ctx: RoundContext,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]: ...


@runtime_checkable
class Reducer(Protocol):
    """Applies accepted actions to the simulation state."""

    def apply(
        self,
        state: Any,
        actions: dict[str, dict[str, Any]],
        round_ctx: RoundContext,
    ) -> Any: ...


@runtime_checkable
class Evaluator(Protocol):
    """Computes derived outputs after state transitions or action submission."""

    async def evaluate(
        self,
        state: Any,
        actions: dict[str, dict[str, Any]],
        round_ctx: RoundContext,
    ) -> dict[str, Any]: ...


@runtime_checkable
class StopCondition(Protocol):
    """Allows a plugin to terminate early based on state and history."""

    def should_stop(self, state: Any, round_ctx: RoundContext) -> bool: ...


@runtime_checkable
class Finalizer(Protocol):
    """Builds a terminal artifact after the round loop completes."""

    async def finalize(
        self,
        state: Any,
        history: list[dict[str, Any]],
        config: dict[str, Any],
        llm_runner: LLMRunner | None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class SimulationPlugin(Protocol):
    """Domain-defined extension point for the kernel runtime."""

    plugin_id: str
    capabilities: PluginCapabilities

    def build_initial_state(self, config: dict[str, Any]) -> Any: ...

    def build_phase_graph(self, config: dict[str, Any]) -> PhaseGraphSpec: ...

    def get_roles(self, state: Any) -> list[AgentRole]: ...

    def get_observation_builder(self, phase_id: str) -> ObservationBuilder | None: ...

    def get_action_parser(self, phase_id: str) -> ActionParser | None: ...

    def get_judge(self, phase_id: str) -> JudgeAdapter | None: ...

    def get_reducer(self, phase_id: str) -> Reducer | None: ...

    def get_evaluator(self, phase_id: str) -> Evaluator | None: ...

    def get_stop_condition(self) -> StopCondition | None: ...

    def get_finalizer(self) -> Finalizer | None: ...
