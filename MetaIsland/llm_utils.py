import os
from typing import Dict, Any


def build_chat_kwargs() -> Dict[str, Any]:
    """Build shared chat completion kwargs from environment settings."""
    kwargs: Dict[str, Any] = {}

    max_tokens = os.getenv("LLM_MAX_TOKENS")
    if max_tokens:
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
