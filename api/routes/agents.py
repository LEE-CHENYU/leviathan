"""Agent registration and profile endpoints."""

from fastapi import APIRouter, HTTPException, Request

from api.models import (
    AgentProfileResponse,
    AgentRegisterRequest,
    AgentRegisterResponse,
)

router = APIRouter(prefix="/v1/agents")


@router.post("/register", response_model=AgentRegisterResponse)
def register_agent(body: AgentRegisterRequest, request: Request):
    """Register a new external agent and assign it to an in-world member."""
    registry = request.app.state.leviathan["registry"]
    kernel = request.app.state.leviathan["kernel"]
    record = registry.register(body.name, body.description, kernel)
    if record is None:
        raise HTTPException(status_code=409, detail="All members are assigned")
    return AgentRegisterResponse(
        agent_id=record.agent_id,
        api_key=record.api_key,
        member_id=record.member_id,
    )


@router.get("/me", response_model=AgentProfileResponse)
def agent_profile(request: Request):
    """Return the profile of the authenticated agent."""
    registry = request.app.state.leviathan["registry"]
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    record = registry.get_by_api_key(key)
    if record is None:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return AgentProfileResponse(
        agent_id=record.agent_id,
        name=record.name,
        member_id=record.member_id,
        registered_at=record.registered_at,
    )
