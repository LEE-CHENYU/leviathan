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
        capabilities=["read_snapshot", "read_events", "read_receipts"],
        endpoints={"base": "/v1/world"},
    )
