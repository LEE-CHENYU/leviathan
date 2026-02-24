import json
import os
import re
from types import SimpleNamespace
from typing import Any, Iterable

import aisuite as ai

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag_enabled(keys: Iterable[str]) -> bool:
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        if value.strip().lower() in _TRUTHY:
            return True
    return False


def _offline_enabled() -> bool:
    return _env_flag_enabled(("LLM_OFFLINE", "E2E_OFFLINE"))


def _extract_member_id(prompt: str) -> int:
    if not prompt:
        return 0
    match = re.search(r"member[_\s]?(\d+)", prompt, flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return 0
    return 0


def _offline_action_code(member_id: int) -> str:
    action = "expand" if member_id % 2 == 0 else "offer"
    lines = [
        "def agent_action(execution_engine, member_id):",
        "    members = execution_engine.current_members",
        "    if not members:",
        "        return",
        "    me = members[member_id]",
        "    target = members[(member_id + 1) % len(members)]",
    ]
    if action == "expand":
        lines.append("    execution_engine.expand(me)")
    else:
        lines.append("    execution_engine.offer(me, target)")
    return "\n".join(lines)


def _offline_mechanism_code() -> str:
    return "\n".join([
        "def propose_modification(execution_engine):",
        "    # Offline stub: no changes to mechanics.",
        "    return None",
    ])


def _offline_analysis_text(member_id: int) -> str:
    baseline = ["expand"] if member_id % 2 == 0 else ["offer"]
    variation = ["offer"] if member_id % 2 == 0 else ["expand"]
    card = {
        "hypothesis": "Offline stub: simple action tags keep pipeline metrics populated.",
        "baseline_signature": baseline,
        "variation_signature": variation,
        "success_metrics": ["delta_survival", "delta_vitality"],
        "guardrails": ["avoid negative survival deltas"],
        "coordination": [],
        "memory_note": f"offline_stub_{member_id}",
        "diversity_note": "Rotate tags across members to avoid monoculture.",
        "confidence": 0.2,
    }
    return "\n".join([
        "Situation summary:",
        "- Offline analysis stub (no external LLM call).",
        "Risks & opportunities:",
        "- Treat results as pipeline validation only.",
        "Strategy plan:",
        f"- Baseline tags: {', '.join(baseline)}",
        f"- Variation tags: {', '.join(variation)}",
        "Coordination asks: none.",
        "Memory note: offline stub.",
        "```json",
        json.dumps(card, indent=2),
        "```",
    ])


def _offline_response_for_prompt(prompt: str) -> str:
    if not prompt:
        return "OK"
    lowered = prompt.lower()
    member_id = _extract_member_id(prompt)

    if "reply with only one of" in lowered and "approve" in lowered:
        return "APPROVE: offline stub"
    if "agent_action" in lowered:
        return _offline_action_code(member_id)
    if "propose_modification" in lowered:
        return _offline_mechanism_code()
    if "output format" in lowered and "json" in lowered and "strategy plan" in lowered:
        return _offline_analysis_text(member_id)
    return "OK"


class _OfflineCompletions:
    def create(self, model: str, messages: list, **kwargs) -> Any:
        prompt = ""
        if messages:
            prompt = messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])
        content = _offline_response_for_prompt(prompt)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _OfflineChat:
    def __init__(self) -> None:
        self.completions = _OfflineCompletions()


class OfflineClient:
    def __init__(self) -> None:
        self.chat = _OfflineChat()


def get_offline_client() -> Any:
    """Return offline stub client regardless of environment flags."""
    return OfflineClient()


def get_llm_client() -> Any:
    """Return offline stub client when requested, otherwise the real LLM client."""
    if _offline_enabled():
        return OfflineClient()
    return ai.Client()
