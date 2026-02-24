"""World state and round endpoints for the Leviathan Read API."""

import dataclasses
from typing import Any, Dict

from fastapi import APIRouter, Request

from api.models import WorldInfo

router = APIRouter(prefix="/v1/world")


@router.get("", response_model=WorldInfo)
def get_world_info(request: Request) -> WorldInfo:
    """Return summary information about the current world state."""
    kernel = request.app.state.leviathan["kernel"]
    snapshot = kernel.get_snapshot()
    return WorldInfo(
        world_id=snapshot.world_id,
        round_id=snapshot.round_id,
        member_count=len(snapshot.members),
        state_hash=snapshot.state_hash,
    )


@router.get("/snapshot")
def get_snapshot(request: Request) -> Dict[str, Any]:
    """Return the full world snapshot as a plain dictionary."""
    kernel = request.app.state.leviathan["kernel"]
    snapshot = kernel.get_snapshot()
    return dataclasses.asdict(snapshot)
