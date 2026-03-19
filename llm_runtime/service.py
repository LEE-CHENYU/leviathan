"""Runtime service that resolves YAML profiles to backend adapters."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from llm_runtime.backends import AisuiteAPIBackend, ClaudeCLIBackend, CodexCLIBackend
from llm_runtime.config import BACKENDS_DIR, PROFILES_DIR, load_named_yaml
from llm_runtime.types import GenerationRequest, GenerationResponse


BACKEND_TYPES = {
    "aisuite_api": AisuiteAPIBackend,
    "claude_cli": ClaudeCLIBackend,
    "codex_cli": CodexCLIBackend,
}


class LLMService:
    """Loads backend/profile YAMLs and dispatches normalized requests."""

    def __init__(
        self,
        *,
        backends_dir: Path | None = None,
        profiles_dir: Path | None = None,
    ):
        self.backends_dir = backends_dir or BACKENDS_DIR
        self.profiles_dir = profiles_dir or PROFILES_DIR
        self._backend_configs = load_named_yaml(self.backends_dir)
        self._profile_configs = load_named_yaml(self.profiles_dir)
        self._backend_instances: dict[str, Any] = {}

    def get_profile(self, name: str) -> dict[str, Any]:
        try:
            return dict(self._profile_configs[name])
        except KeyError as exc:
            raise KeyError(f"Unknown LLM profile: {name}") from exc

    def get_backend(self, name: str):
        if name not in self._backend_instances:
            config = self._backend_configs.get(name)
            if config is None:
                raise KeyError(f"Unknown LLM backend: {name}")
            backend_type = config.get("type")
            try:
                backend_cls = BACKEND_TYPES[backend_type]
            except KeyError as exc:
                raise KeyError(f"Unsupported LLM backend type: {backend_type}") from exc
            self._backend_instances[name] = backend_cls(config)
        return self._backend_instances[name]

    async def generate(
        self,
        *,
        prompt: str,
        profile: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        reasoning: str | None = None,
        json_output: bool | None = None,
        options: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        profile_cfg = self.get_profile(profile)
        backend_name = profile_cfg["backend"]
        backend = self.get_backend(backend_name)

        request = GenerationRequest(
            prompt=prompt,
            model=model if model is not None else profile_cfg.get("model"),
            temperature=(
                temperature if temperature is not None else profile_cfg.get("temperature")
            ),
            max_tokens=max_tokens if max_tokens is not None else profile_cfg.get("max_tokens"),
            timeout=timeout if timeout is not None else profile_cfg.get("timeout"),
            reasoning=reasoning if reasoning is not None else profile_cfg.get("reasoning"),
            json_output=(
                json_output if json_output is not None else profile_cfg.get("json_output")
            ),
            options=dict(options or {}),
        )
        return await backend.generate(request)

    async def generate_text(self, **kwargs: Any) -> tuple[str | None, float, dict[str, Any]]:
        return (await self.generate(**kwargs)).as_tuple()


@lru_cache(maxsize=1)
def get_default_service() -> LLMService:
    """Return the process-wide default runtime service."""

    return LLMService()
