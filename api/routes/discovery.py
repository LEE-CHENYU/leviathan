"""Agent discovery endpoint for the Leviathan Read API."""

from fastapi import APIRouter

from api.models import AgentDiscovery

router = APIRouter()


@router.get("/.well-known/leviathan-agent.json", response_model=AgentDiscovery)
def get_discovery() -> AgentDiscovery:
    """Return the agent discovery manifest."""
    return AgentDiscovery(
        name="Leviathan",
        version="0.1.0",
        api_version="v1",
        capabilities=[
            "read_snapshot",
            "read_events",
            "read_receipts",
            "submit_actions",
            "propose_mechanisms",
            "metrics",
            "judge_stats",
            "moderator_controls",
            "oracle_signing",
        ],
        endpoints={
            "health": "/health",
            "discovery": "/.well-known/leviathan-agent.json",
            "world_info": "/v1/world",
            "snapshot": "/v1/world/snapshot",
            "current_round": "/v1/world/rounds/current",
            "deadline": "/v1/world/rounds/current/deadline",
            "round_by_id": "/v1/world/rounds/{round_id}",
            "events": "/v1/world/events",
            "register": "/v1/agents/register",
            "agent_profile": "/v1/agents/me",
            "submit_action": "/v1/world/actions",
            "propose_mechanism": "/v1/world/mechanisms/propose",
            "list_mechanisms": "/v1/world/mechanisms",
            "mechanism_by_id": "/v1/world/mechanisms/{mechanism_id}",
            "metrics": "/v1/world/metrics",
            "metrics_history": "/v1/world/metrics/history",
            "judge_stats": "/v1/world/judge/stats",
            "admin_status": "/v1/admin/status",
            "admin_pause": "/v1/admin/pause",
            "admin_resume": "/v1/admin/resume",
            "admin_ban": "/v1/admin/ban/{agent_id}",
            "admin_unban": "/v1/admin/unban/{agent_id}",
            "admin_quotas": "/v1/admin/quotas",
            "admin_rollback": "/v1/admin/rollback",
        },
    )
