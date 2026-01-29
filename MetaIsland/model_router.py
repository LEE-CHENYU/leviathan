import os
from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@lru_cache(maxsize=1)
def _load_model_config():
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _env_override(route_key: str) -> dict:
    normalized = str(route_key).upper().replace("-", "_").replace(".", "_")
    provider = os.getenv(f"MODEL_ROUTE_{normalized}_PROVIDER")
    model_id = os.getenv(f"MODEL_ROUTE_{normalized}_MODEL_ID")
    if provider or model_id:
        return {"provider": provider, "model_id": model_id}
    return {}


def _configure_openrouter_env() -> None:
    key = os.getenv("OPENROUTER_API_KEY")
    if key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key
    if (key or os.getenv("OPENAI_API_KEY")) and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = _OPENROUTER_BASE_URL


def _normalize_provider(provider: object) -> str:
    if not provider:
        return "openai"
    normalized = str(provider).strip().lower()
    if normalized == "openrouter":
        _configure_openrouter_env()
        return "openai"
    return normalized


def model_router(model):
    config = _load_model_config()
    routes = config.get("routes", {})
    default = config.get("default", {})
    benchmark = config.get("benchmark", {})

    env_override = _env_override(model or "default")
    if model in (None, "", "default"):
        entry = default
    elif model in ("benchmark", "eval", "evaluation"):
        entry = benchmark or routes.get("gpt-5.2", {})
    else:
        entry = routes.get(model, {}) if isinstance(routes, dict) else {}

    provider = _normalize_provider(
        env_override.get("provider")
        or entry.get("provider")
        or default.get("provider")
        or "openai"
    )
    model_id = (
        env_override.get("model_id")
        or entry.get("model_id")
        or default.get("model_id")
        or "gpt-5.2"
    )

    return provider, model_id
