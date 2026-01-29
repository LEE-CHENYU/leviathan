import os
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import aisuite as ai

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


def offline_mode_enabled() -> bool:
    """Return True when the repo should avoid real LLM calls."""
    if _is_truthy(os.getenv("LLM_OFFLINE")):
        return True
    if _is_truthy(os.getenv("E2E_OFFLINE")):
        return True
    provider = os.getenv("E2E_PROVIDER", "").strip().lower()
    return provider in {"offline", "mock", "stub"}


def get_llm_client() -> Any:
    """Return the real LLM client unless offline mode is enabled."""
    if offline_mode_enabled():
        return OfflineClient()
    return ai.Client()


class OfflineClient:
    """Minimal offline client that mimics aisuite's chat completion shape."""

    def __init__(self) -> None:
        self.chat = _OfflineChat()


class _OfflineChat:
    def __init__(self) -> None:
        self.completions = _OfflineCompletions()


class _OfflineCompletions:
    def create(self, model: Optional[str] = None, messages: Optional[Iterable[Dict[str, Any]]] = None, **kwargs: Any) -> Any:
        prompt = ""
        if messages:
            for message in reversed(list(messages)):
                if isinstance(message, dict) and "content" in message:
                    prompt = str(message["content"])
                    break
        content = _offline_response(prompt)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _offline_response(prompt: str) -> str:
    lower = prompt.lower() if prompt else ""
    if "reply with only one of" in lower and ("approve" in lower or "reject" in lower):
        return "APPROVE: offline stub approval"
    if "agent_action" in lower:
        return _offline_agent_action_code()
    if "propose_modification" in lower or "mechanism proposal" in lower:
        return _offline_mechanism_code()
    if "output format" in lower and "json" in lower:
        return _offline_analysis_text()
    if "analysis" in lower and "strategy plan" in lower:
        return _offline_analysis_text()
    return "Offline stub response."


def _offline_agent_action_code() -> str:
    return (
        "def agent_action(execution_engine, member_id):\n"
        "    \"\"\"Offline stub action for smoke tests.\"\"\"\n"
        "    members = getattr(execution_engine, 'current_members', None)\n"
        "    if not members:\n"
        "        return\n"
        "    if member_id < 0 or member_id >= len(members):\n"
        "        return\n"
        "    me = members[member_id]\n"
        "    if member_id % 2 == 0:\n"
        "        execution_engine.expand(me)\n"
        "    else:\n"
        "        target_index = (member_id + 1) % len(members)\n"
        "        target = members[target_index].id\n"
        "        execution_engine.send_message(me.id, target, 'offline stub: coordinating')\n"
    )


def _offline_mechanism_code() -> str:
    return (
        "def propose_modification(self):\n"
        "    \"\"\"Offline stub: no-op mechanism change.\"\"\"\n"
        "    return None\n"
    )


def _offline_analysis_text() -> str:
    return (
        "Situation summary:\n"
        "- Offline stub active; skipping detailed analysis.\n"
        "- Use minimal baseline/variation actions to keep experiments interpretable.\n"
        "\n"
        "Risks & opportunities:\n"
        "- Risk: strategy quality degraded without LLM guidance.\n"
        "- Opportunity: validate pipeline without API access.\n"
        "\n"
        "Strategy plan:\n"
        "- Baseline (safe): expand.\n"
        "- Variation (bounded-risk): message.\n"
        "- Guardrails: avoid attack/offer in offline mode.\n"
        "\n"
        "Coordination asks:\n"
        "- Send a brief status ping to a neighbor.\n"
        "\n"
        "Memory note: offline stub.\n"
        "\n"
        "```json\n"
        "{\n"
        "  \"hypothesis\": \"Offline stub run; maintain minimal actions.\",\n"
        "  \"baseline_signature\": [\"expand\"],\n"
        "  \"variation_signature\": [\"message\"],\n"
        "  \"success_metrics\": [\"delta_survival>=0\"],\n"
        "  \"guardrails\": [\"avoid attack in offline mode\"],\n"
        "  \"coordination\": [\"send status ping\"],\n"
        "  \"memory_note\": \"offline stub\",\n"
        "  \"diversity_note\": \"alternate expand/message by member_id\",\n"
        "  \"confidence\": 0.05\n"
        "}\n"
        "```\n"
    )
