"""Pydantic response models for the Leviathan Read API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class WorldInfo(BaseModel):
    """Summary information about the current world state."""

    world_id: str
    round_id: int
    member_count: int
    state_hash: str
    world_public_key: Optional[str] = None


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
    # Phase 4 fields
    constitution_hash: Optional[str] = None
    oracle_signature: Optional[str] = None
    world_public_key: Optional[str] = None
    origin_world_id: Optional[str] = None
    origin_receipt_hash: Optional[str] = None
    bridge_channel_id: Optional[str] = None
    bridge_seq: Optional[int] = None
    notary_signature: Optional[str] = None


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
    # Phase 4: Federation-prep fields (A4)
    world_id: Optional[str] = None
    phase: Optional[str] = None
    payload_hash: Optional[str] = None
    prev_event_hash: Optional[str] = None


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


# ── Phase 3: Governance models ────────────────


class MechanismProposeRequest(BaseModel):
    """Request body for POST /v1/world/mechanisms/propose."""
    code: str
    description: str
    idempotency_key: str


class MechanismProposeResponse(BaseModel):
    """Response from POST /v1/world/mechanisms/propose."""
    mechanism_id: str
    status: str


class MechanismResponse(BaseModel):
    """Full mechanism detail response."""
    mechanism_id: str
    proposer_id: int
    code: str
    description: str
    status: str
    submitted_round: int
    judged_round: Optional[int] = None
    judge_reason: Optional[str] = None
    activated_round: Optional[int] = None


class MetricsResponse(BaseModel):
    """Response from GET /v1/world/metrics."""
    round_id: int
    total_vitality: float
    gini_coefficient: float
    trade_volume: int
    conflict_count: int
    mechanism_proposals: int
    mechanism_approvals: int
    population: int


class JudgeStatsResponse(BaseModel):
    """Response from GET /v1/world/judge/stats."""
    total_judgments: int
    approved: int
    rejected: int
    approval_rate: float
    recent_rejections: List[Dict[str, Any]]


# ── Phase 4: Moderator / Admin models ────────────


class AdminStatusResponse(BaseModel):
    """Response from GET /v1/admin/status."""
    paused: bool
    banned_agents: List[int]
    quotas: Dict[str, Any]
    snapshot_history_size: int


class QuotaUpdateRequest(BaseModel):
    """Request body for PUT /v1/admin/quotas."""
    max_actions_per_round: Optional[int] = None
    max_proposals_per_round: Optional[int] = None
