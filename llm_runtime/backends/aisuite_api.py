"""`aisuite` API backend."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from llm_runtime.types import GenerationRequest, GenerationResponse


def _extract_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content or "")


def _normalize_model_name(model: str, provider: str | None) -> str:
    if ":" in model or not provider:
        return model
    return f"{provider}:{model}"


class AisuiteAPIBackend:
    """Simple `aisuite` chat backend."""

    def __init__(self, settings: dict[str, Any]):
        self.settings = dict(settings)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        import aisuite as ai

        model = request.model or self.settings.get("model")
        if not model:
            raise ValueError("Aisuite backend requires a model")
        model = _normalize_model_name(model, self.settings.get("provider"))

        timeout = request.timeout or self.settings.get("timeout")
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.json_output:
            kwargs["response_format"] = {"type": "json_object"}

        client = ai.Client()
        start = time.time()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(client.chat.completions.create, **kwargs),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return GenerationResponse(
                text=None,
                elapsed=time.time() - start,
                info={"status": "timeout", "error": "TIMEOUT"},
            )
        except Exception as exc:
            return GenerationResponse(
                text=None,
                elapsed=time.time() - start,
                info={"status": "error", "error": str(exc)},
            )

        try:
            raw_text = _extract_content(response.choices[0].message).strip()
        except Exception as exc:
            return GenerationResponse(
                text=None,
                elapsed=time.time() - start,
                info={"status": "error", "error": f"response_parse_failed: {exc}"},
            )

        return GenerationResponse(
            text=raw_text,
            elapsed=time.time() - start,
            info={"status": "success"},
        )
