"""Shared LLM runtime with swappable backend/profile configs."""

from .service import LLMService, get_default_service
from .types import GenerationRequest, GenerationResponse

__all__ = [
    "GenerationRequest",
    "GenerationResponse",
    "LLMService",
    "get_default_service",
]
