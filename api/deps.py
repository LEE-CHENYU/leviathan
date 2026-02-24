"""Dependency injection helpers for the Leviathan API."""

from typing import Any, Dict, List

from fastapi import Request

from api.auth import APIKeyAuth
from api.models import EventEnvelope
from api.registry import AgentRegistry
from api.round_state import RoundState
from kernel.mechanism_registry import MechanismRegistry
from kernel.judge_adapter import JudgeAdapter
from kernel.world_kernel import WorldKernel


def create_app_state(kernel: WorldKernel) -> Dict[str, Any]:
    """Build the shared application state dictionary from a kernel instance."""
    return {
        "kernel": kernel,
        "event_log": [],
        "registry": AgentRegistry(),
        "round_state": RoundState(),
        "mechanism_registry": MechanismRegistry(),
        "judge": JudgeAdapter(use_dummy=True),
    }


def get_kernel(state: Dict[str, Any]) -> WorldKernel:
    return state["kernel"]


def get_event_log(state: Dict[str, Any]) -> List[EventEnvelope]:
    return state["event_log"]


def get_registry(request: Request) -> AgentRegistry:
    return request.app.state.leviathan["registry"]


def get_round_state(request: Request) -> RoundState:
    return request.app.state.leviathan["round_state"]


def get_auth(request: Request) -> APIKeyAuth:
    """Retrieve the APIKeyAuth instance from application state."""
    return request.app.state.auth


def get_mechanism_registry(request: Request) -> "MechanismRegistry":
    return request.app.state.leviathan["mechanism_registry"]


def get_judge(request: Request) -> "JudgeAdapter":
    return request.app.state.leviathan["judge"]
