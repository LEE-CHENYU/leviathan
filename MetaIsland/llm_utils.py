import os
from typing import Dict, Any


def _use_max_completion_tokens() -> bool:
    """Return True when OpenAI-style max_completion_tokens is required."""
    if os.getenv("LLM_USE_MAX_COMPLETION_TOKENS"):
        return True
    base_url = os.getenv("OPENAI_BASE_URL", "").lower()
    return "openai.com" in base_url


def build_chat_kwargs() -> Dict[str, Any]:
    """Build shared chat completion kwargs from environment settings."""
    kwargs: Dict[str, Any] = {}

    max_tokens = os.getenv("LLM_MAX_TOKENS")
    max_completion_tokens = os.getenv("LLM_MAX_COMPLETION_TOKENS")
    if _use_max_completion_tokens():
        value = max_completion_tokens or max_tokens
        if value:
            try:
                kwargs["max_completion_tokens"] = int(value)
            except ValueError:
                pass
    elif max_tokens:
        try:
            kwargs["max_tokens"] = int(max_tokens)
        except ValueError:
            pass

    temperature = os.getenv("LLM_TEMPERATURE")
    if temperature is not None:
        try:
            kwargs["temperature"] = float(temperature)
        except ValueError:
            pass

    return kwargs


def classify_llm_error(error: Exception) -> str:
    """Classify LLM client errors into coarse categories for diagnostics."""
    if error is None:
        return "llm_unknown"
    message = str(error)
    type_name = error.__class__.__name__ if hasattr(error, "__class__") else ""
    text = f"{type_name} {message}".lower()

    if not text.strip():
        return "llm_unknown"

    if "rate limit" in text or "429" in text:
        return "llm_rate_limit"
    if "timeout" in text or "timed out" in text:
        return "llm_timeout"
    if "authentication" in text or "api key" in text or "unauthorized" in text or "401" in text:
        return "llm_auth_error"
    if (
        "connect" in text
        or "connection" in text
        or "nodename nor servname" in text
        or "dns" in text
        or "name resolution" in text
    ):
        return "llm_connection_error"
    if "bad request" in text or "invalid request" in text or "400" in text:
        return "llm_invalid_request"
    if "not found" in text or "404" in text:
        return "llm_not_found"

    return "llm_unknown"


_OFFLINE_FALLBACK_CATEGORIES = {
    "llm_connection_error",
    "llm_timeout",
    "llm_rate_limit",
}


def should_use_offline_fallback(error: Exception) -> bool:
    """Return True when a transient LLM failure should trigger offline fallback."""
    return classify_llm_error(error) in _OFFLINE_FALLBACK_CATEGORIES
