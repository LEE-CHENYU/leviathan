"""Dependency injection helpers for the Leviathan Read API."""

from typing import Any, Dict, List

from api.models import EventEnvelope
from kernel.world_kernel import WorldKernel


def create_app_state(kernel: WorldKernel) -> Dict[str, Any]:
    """Build the shared application state dictionary from a kernel instance."""
    return {"kernel": kernel, "event_log": []}


def get_kernel(state: Dict[str, Any]) -> WorldKernel:
    """Extract the WorldKernel from application state."""
    return state["kernel"]


def get_event_log(state: Dict[str, Any]) -> List[EventEnvelope]:
    """Extract the event log from application state."""
    return state["event_log"]
