import asyncio
import json
import os
import re
import threading
from types import SimpleNamespace
from typing import Any, Iterable

from llm_runtime.service import get_default_service

_TRUTHY = {"1", "true", "yes", "on"}
_PROFILE_ENV_KEYS = (
    "METAISLAND_LLM_PROFILE",
    "LEVIATHAN_LLM_PROFILE",
    "LLM_RUNTIME_PROFILE",
)
_DEFAULT_PROFILE = "codex_cli_default"


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
    return "\n".join(
        [
            "def propose_modification(execution_engine):",
            "    # Offline stub: no changes to mechanics.",
            "    return None",
        ]
    )


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
    return "\n".join(
        [
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
        ]
    )


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
        prompt = _messages_to_prompt(messages)
        content = _offline_response_for_prompt(prompt)
        return _build_completion(content=content, model=model)


class _OfflineChat:
    def __init__(self) -> None:
        self.completions = _OfflineCompletions()


class OfflineClient:
    def __init__(self) -> None:
        self.chat = _OfflineChat()


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
            else:
                text = getattr(item, "text", None) or getattr(item, "content", "")
            if text:
                parts.append(str(text))
        return "".join(parts)
    return str(content)


def _messages_to_prompt(messages: Any) -> str:
    if not messages:
        return ""
    formatted: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = _message_content_to_text(message.get("content"))
        else:
            role = getattr(message, "role", "user")
            content = _message_content_to_text(getattr(message, "content", ""))
        content = str(content or "").strip()
        if not content:
            continue
        formatted.append(f"[{str(role).upper()}]\n{content}")
    return "\n\n".join(formatted).strip()


def _resolve_profile_name() -> str:
    for key in _PROFILE_ENV_KEYS:
        value = os.getenv(key)
        if value:
            return value.strip()
    return _DEFAULT_PROFILE


def _normalize_model_for_profile(service: Any, profile_name: str, model: str | None) -> str | None:
    if not model:
        return None
    try:
        profile = service.get_profile(profile_name)
    except Exception:
        return model
    backend_name = profile.get("backend")
    if backend_name != "aisuite_api" and ":" in model:
        return model.split(":", 1)[1]
    return model


def _pick_max_tokens(kwargs: dict[str, Any]) -> int | None:
    value = kwargs.get("max_completion_tokens")
    if value is None:
        value = kwargs.get("max_tokens")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pick_json_output(kwargs: dict[str, Any]) -> bool | None:
    response_format = kwargs.get("response_format")
    if isinstance(response_format, dict):
        return response_format.get("type") == "json_object"
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_completion(content: str, model: str | None, info: dict[str, Any] | None = None) -> Any:
    metadata = dict(info or {})
    prompt_tokens = _coerce_int(metadata.get("prompt_tokens") or metadata.get("input_tokens"))
    completion_tokens = _coerce_int(
        metadata.get("completion_tokens") or metadata.get("output_tokens")
    )
    total_tokens = _coerce_int(metadata.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=metadata.get("finish_reason", "stop"),
    )
    return SimpleNamespace(
        model=model,
        choices=[choice],
        usage=usage,
    )


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - propagated below
            error["value"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result.get("value")


class _RuntimeCompletions:
    def __init__(self, service: Any) -> None:
        self._service = service

    def create(self, model: str, messages: list, **kwargs) -> Any:
        prompt = _messages_to_prompt(messages)
        profile_name = _resolve_profile_name()
        normalized_model = _normalize_model_for_profile(self._service, profile_name, model)
        response = _run_coro_sync(
            self._service.generate(
                prompt=prompt,
                profile=profile_name,
                model=normalized_model,
                temperature=kwargs.get("temperature"),
                max_tokens=_pick_max_tokens(kwargs),
                timeout=kwargs.get("timeout"),
                json_output=_pick_json_output(kwargs),
            )
        )

        status = response.info.get("status")
        if status == "timeout":
            raise TimeoutError(response.info.get("error", "TIMEOUT"))
        if status != "success" or not response.text:
            raise RuntimeError(response.info.get("error", "llm_generation_failed"))

        return _build_completion(
            content=response.text,
            model=normalized_model or model,
            info=response.info,
        )


class _RuntimeChat:
    def __init__(self, service: Any) -> None:
        self.completions = _RuntimeCompletions(service)


class RuntimeClient:
    def __init__(self, service: Any) -> None:
        self.chat = _RuntimeChat(service)


def get_offline_client() -> Any:
    """Return offline stub client regardless of environment flags."""
    return OfflineClient()


def get_llm_client() -> Any:
    """Return a compatibility client backed by the shared llm_runtime service."""
    if _offline_enabled():
        return OfflineClient()
    return RuntimeClient(get_default_service())
