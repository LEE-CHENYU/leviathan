"""Dependency injection helpers for the Leviathan API."""

import os
from typing import Any, Dict, List

from fastapi import Request

from api.auth import APIKeyAuth
from api.models import EventEnvelope
from api.registry import AgentRegistry
from api.round_state import RoundState
from kernel.event_log import EventLog
from kernel.mechanism_registry import MechanismRegistry
from kernel.judge_adapter import JudgeAdapter
from kernel.moderator import ModeratorState
from kernel.store import Store
from kernel.world_kernel import WorldKernel


def create_app_state(kernel: WorldKernel, data_dir: str = "") -> Dict[str, Any]:
    """Build the shared application state dictionary from a kernel instance.

    If *data_dir* is provided, a SQLite store is created there and used
    to persist events, mechanisms, and snapshots across restarts.
    """
    store = None
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        store = Store(os.path.join(data_dir, "leviathan.db"))

    return {
        "kernel": kernel,
        "store": store,
        "event_log": EventLog(store=store),
        "registry": AgentRegistry(),
        "round_state": RoundState(),
        "mechanism_registry": MechanismRegistry(store=store),
        "judge": JudgeAdapter(use_dummy=True),
        "moderator": ModeratorState(store=store),
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
