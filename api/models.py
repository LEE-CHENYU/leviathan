"""Pydantic response models for the Leviathan Read API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class WorldInfo(BaseModel):
    """Summary information about the current world state."""

    world_id: str
    round_id: int
    member_count: int
    state_hash: str


class RoundReceiptResponse(BaseModel):
    """API response model mirroring the kernel RoundReceipt."""

    round_id: int
    seed: int
    snapshot_hash_before: str
    snapshot_hash_after: str
    accepted_action_ids: List[str]
    rejected_action_ids: List[str]
    activated_mechanism_ids: List[str]
    judge_results: List[Dict[str, Any]]
    round_metrics: Dict[str, float]
    timestamp: str


class RoundInfo(BaseModel):
    """Current round state with optional last receipt."""

    round_id: int
    last_receipt: Optional[RoundReceiptResponse] = None


class EventEnvelope(BaseModel):
    """Wrapper for a single event in the event log."""

    event_id: int
    event_type: str
    round_id: int
    timestamp: str
    payload: Dict[str, Any]


class AgentDiscovery(BaseModel):
    """Discovery manifest describing this server's capabilities."""

    name: str
    version: str
    api_version: str
    capabilities: List[str]
    endpoints: Dict[str, str]


# ── Phase 2: Write endpoint models ──────────────


class AgentRegisterRequest(BaseModel):
    """Request body for POST /v1/agents/register."""
    name: str
    description: str = ""


class AgentRegisterResponse(BaseModel):
    """Response from POST /v1/agents/register."""
    agent_id: int
    api_key: str
    member_id: int


class AgentProfileResponse(BaseModel):
    """Response from GET /v1/agents/me."""
    agent_id: int
    name: str
    member_id: int
    registered_at: str


class ActionSubmitRequest(BaseModel):
    """Request body for POST /v1/world/actions."""
    code: str
    idempotency_key: str


class ActionSubmitResponse(BaseModel):
    """Response from POST /v1/world/actions."""
    status: str
    round_id: int
    reason: Optional[str] = None


class DeadlineResponse(BaseModel):
    """Response from GET /v1/world/rounds/current/deadline."""
    round_id: int
    state: str
    deadline_utc: Optional[str] = None
    seconds_remaining: float
