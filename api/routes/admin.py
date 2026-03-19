"""Moderator / admin endpoints for world management."""

import hashlib
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, Request

from api.models import AdminStatusResponse, EventEnvelope, QuotaUpdateRequest

router = APIRouter(prefix="/v1/admin")


def _require_moderator(request: Request) -> str:
    """Check moderator key and return the key for audit logging."""
    auth = request.app.state.auth
    if not auth.enabled:
        return "open_access"
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if not auth.is_moderator_key(key):
        raise HTTPException(status_code=403, detail="Moderator access required")
    return key


def _emit_admin_event(request: Request, event_type: str, key: str, payload: dict) -> None:
    """Append an admin event to the event log."""
    from kernel.moderator import ModeratorState
    event_log: List[EventEnvelope] = request.app.state.leviathan["event_log"]
    event_log.append(EventEnvelope(
        event_id=len(event_log) + 1,
        event_type=event_type,
        round_id=request.app.state.leviathan["kernel"].round_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        payload={
            "moderator_key_hash": ModeratorState.hash_key(key),
            **payload,
        },
    ))


@router.get("/status", response_model=AdminStatusResponse)
def admin_status(request: Request):
    key = _require_moderator(request)
    mod_state = request.app.state.leviathan["moderator"]
    d = mod_state.to_dict()
    return AdminStatusResponse(**d)


@router.post("/pause")
def admin_pause(request: Request):
    key = _require_moderator(request)
    mod_state = request.app.state.leviathan["moderator"]
    mod_state.paused = True
    _emit_admin_event(request, "admin_pause", key, {"action": "pause"})
    return {"status": "paused"}


@router.post("/resume")
def admin_resume(request: Request):
    key = _require_moderator(request)
    mod_state = request.app.state.leviathan["moderator"]
    mod_state.paused = False
    _emit_admin_event(request, "admin_resume", key, {"action": "resume"})
    return {"status": "resumed"}


@router.post("/ban/{agent_id}")
def admin_ban(agent_id: int, request: Request):
    key = _require_moderator(request)
    mod_state = request.app.state.leviathan["moderator"]
    mod_state.ban(agent_id)
    _emit_admin_event(request, "admin_ban", key, {"agent_id": agent_id})
    return {"status": "banned", "agent_id": agent_id}


@router.post("/unban/{agent_id}")
def admin_unban(agent_id: int, request: Request):
    key = _require_moderator(request)
    mod_state = request.app.state.leviathan["moderator"]
    mod_state.unban(agent_id)
    _emit_admin_event(request, "admin_unban", key, {"agent_id": agent_id})
    return {"status": "unbanned", "agent_id": agent_id}


@router.put("/quotas")
def admin_update_quotas(body: QuotaUpdateRequest, request: Request):
    key = _require_moderator(request)
    mod_state = request.app.state.leviathan["moderator"]
    if body.max_actions_per_round is not None:
        mod_state.quotas.max_actions_per_round = body.max_actions_per_round
    if body.max_proposals_per_round is not None:
        mod_state.quotas.max_proposals_per_round = body.max_proposals_per_round
    _emit_admin_event(request, "admin_quotas", key, {
        "max_actions_per_round": mod_state.quotas.max_actions_per_round,
        "max_proposals_per_round": mod_state.quotas.max_proposals_per_round,
    })
    return {"status": "updated", "quotas": mod_state.to_dict()["quotas"]}


@router.post("/rollback")
def admin_rollback(request: Request, target_round: int = 0):
    key = _require_moderator(request)
    mod_state = request.app.state.leviathan["moderator"]
    kernel = request.app.state.leviathan["kernel"]
    if target_round <= 0:
        raise HTTPException(status_code=400, detail="target_round must be positive")
    snapshot = mod_state.get_snapshot_for_round(target_round)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"No snapshot available for round {target_round}")
    old_round = kernel.round_id
    kernel._round_id = target_round
    kernel._idempotency_cache = {}
    kernel._last_receipt = None
    _emit_admin_event(request, "admin_rollback", key, {"from_round": old_round, "to_round": target_round})
    return {"status": "rolled_back", "from_round": old_round, "to_round": target_round}
