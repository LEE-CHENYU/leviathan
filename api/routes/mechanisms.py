"""Mechanism proposal and listing endpoints."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request

from api.models import MechanismProposeRequest, MechanismProposeResponse, MechanismResponse
from api.round_state import PendingProposal

router = APIRouter(prefix="/v1/world/mechanisms")


@router.post("/propose", response_model=MechanismProposeResponse)
def propose_mechanism(body: MechanismProposeRequest, request: Request):
    """Submit a mechanism proposal for the current round."""
    registry = request.app.state.leviathan["registry"]
    round_state = request.app.state.leviathan["round_state"]
    mechanism_registry = request.app.state.leviathan["mechanism_registry"]

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

    pp = PendingProposal(
        agent_id=record.agent_id, member_id=record.member_id,
        code=body.code, description=body.description,
        idempotency_key=body.idempotency_key,
    )
    accepted = round_state.submit_proposal(pp)
    if not accepted:
        return MechanismProposeResponse(mechanism_id="", status="rejected")

    mech = mechanism_registry.submit(
        proposer_id=record.member_id, code=body.code,
        description=body.description, round_id=round_state.round_id,
    )
    if mech is None:
        return MechanismProposeResponse(mechanism_id="", status="rejected")

    return MechanismProposeResponse(mechanism_id=mech.mechanism_id, status="submitted")


@router.get("", response_model=List[MechanismResponse])
def list_mechanisms(request: Request, status: Optional[str] = None):
    """List all mechanisms, optionally filtered by status."""
    mechanism_registry = request.app.state.leviathan["mechanism_registry"]
    if status == "active":
        records = mechanism_registry.get_active()
    elif status == "submitted":
        records = mechanism_registry.get_pending()
    else:
        records = mechanism_registry.get_all()
    return [
        MechanismResponse(
            mechanism_id=r.mechanism_id, proposer_id=r.proposer_id, code=r.code,
            description=r.description, status=r.status, submitted_round=r.submitted_round,
            judged_round=r.judged_round, judge_reason=r.judge_reason,
            activated_round=r.activated_round,
        )
        for r in records
    ]


@router.get("/{mechanism_id}", response_model=MechanismResponse)
def get_mechanism(mechanism_id: str, request: Request):
    """Get a mechanism by ID."""
    mechanism_registry = request.app.state.leviathan["mechanism_registry"]
    rec = mechanism_registry.get_by_id(mechanism_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Mechanism not found")
    return MechanismResponse(
        mechanism_id=rec.mechanism_id, proposer_id=rec.proposer_id, code=rec.code,
        description=rec.description, status=rec.status, submitted_round=rec.submitted_round,
        judged_round=rec.judged_round, judge_reason=rec.judge_reason,
        activated_round=rec.activated_round,
    )
