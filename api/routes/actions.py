"""Action submission endpoint for external agents."""

from fastapi import APIRouter, HTTPException, Request

from api.models import ActionSubmitRequest, ActionSubmitResponse
from api.round_state import PendingAction

router = APIRouter(prefix="/v1/world")


@router.post("/actions", response_model=ActionSubmitResponse)
def submit_action(body: ActionSubmitRequest, request: Request):
    """Submit an action for the current round."""
    registry = request.app.state.leviathan["registry"]
    round_state = request.app.state.leviathan["round_state"]

    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    record = registry.get_by_api_key(key)
    if record is None:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Check if agent is banned
    mod_state = request.app.state.leviathan["moderator"]
    if mod_state.is_banned(record.member_id):
        raise HTTPException(status_code=403, detail="Agent is banned")

    pa = PendingAction(
        agent_id=record.agent_id,
        member_id=record.member_id,
        code=body.code,
        idempotency_key=body.idempotency_key,
    )
    accepted = round_state.submit_action(pa)

    if not accepted:
        return ActionSubmitResponse(
            status="rejected",
            round_id=round_state.round_id,
            reason="Round not accepting submissions",
        )

    return ActionSubmitResponse(status="accepted", round_id=round_state.round_id)
