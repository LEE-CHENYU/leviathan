"""Runtime request/response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    """Normalized request passed to a backend adapter."""

    prompt: str
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: int | None = None
    reasoning: str | None = None
    json_output: bool | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResponse:
    """Normalized backend response."""

    text: str | None
    elapsed: float
    info: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[str | None, float, dict[str, Any]]:
        return self.text, self.elapsed, dict(self.info)


class LLMBackend(Protocol):
    """Async backend protocol used by the runtime service."""

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion for the provided request."""
