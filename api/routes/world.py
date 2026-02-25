"""World state and round endpoints for the Leviathan Read API."""

import dataclasses
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from api.models import DeadlineResponse, EventEnvelope, RoundInfo, RoundReceiptResponse, WorldInfo

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
        world_public_key=kernel.oracle.world_public_key,
    )


@router.get("/snapshot")
def get_snapshot(request: Request) -> Dict[str, Any]:
    """Return the full world snapshot as a plain dictionary."""
    kernel = request.app.state.leviathan["kernel"]
    snapshot = kernel.get_snapshot()
    return dataclasses.asdict(snapshot)


@router.get("/rounds/current", response_model=RoundInfo)
def get_current_round(request: Request) -> RoundInfo:
    """Return the current round info with optional last receipt."""
    kernel = request.app.state.leviathan["kernel"]
    receipt = kernel.get_round_receipt()
    last_receipt = None
    if receipt is not None:
        last_receipt = RoundReceiptResponse(**dataclasses.asdict(receipt))
    return RoundInfo(round_id=kernel.round_id, last_receipt=last_receipt)


@router.get("/rounds/current/deadline", response_model=DeadlineResponse)
def get_deadline(request: Request) -> DeadlineResponse:
    """Return the current round's submission deadline."""
    round_state = request.app.state.leviathan["round_state"]
    return DeadlineResponse(
        round_id=round_state.round_id,
        state=round_state.state,
        deadline_utc=round_state.deadline.isoformat() if round_state.deadline else None,
        seconds_remaining=round_state.seconds_remaining(),
    )


@router.get("/rounds/{round_id}", response_model=RoundReceiptResponse)
def get_round_by_id(round_id: int, request: Request) -> RoundReceiptResponse:
    """Return the receipt for a specific round, or 404 if not found."""
    event_log: List[EventEnvelope] = request.app.state.leviathan["event_log"]
    for event in event_log:
        if event.event_type == "round_settled" and event.round_id == round_id:
            return RoundReceiptResponse(**event.payload)
    raise HTTPException(status_code=404, detail=f"Round {round_id} not found")


@router.get("/events", response_model=List[EventEnvelope])
def get_events(
    request: Request, since_round: Optional[int] = None
) -> List[EventEnvelope]:
    """Return event log entries, optionally filtered to rounds after since_round."""
    event_log: List[EventEnvelope] = request.app.state.leviathan["event_log"]
    if since_round is None:
        return list(event_log)
    return [e for e in event_log if e.round_id > since_round]
