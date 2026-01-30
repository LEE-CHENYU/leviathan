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


def _normalize_base_url(value: object) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    return text or None


def _select_base_url(entry: dict, default: dict) -> str | None:
    if isinstance(entry, dict):
        base_url = _normalize_base_url(entry.get("base_url"))
        if base_url:
            return base_url
    if isinstance(default, dict):
        base_url = _normalize_base_url(default.get("base_url"))
        if base_url:
            return base_url
    return None


def _configure_openrouter_env(base_url: str | None = None) -> None:
    key = os.getenv("OPENROUTER_API_KEY")
    if key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key
    if base_url and not os.getenv("OPENROUTER_BASE_URL"):
        os.environ["OPENROUTER_BASE_URL"] = base_url
    if (key or os.getenv("OPENAI_API_KEY")) and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = base_url or _OPENROUTER_BASE_URL


def _normalize_provider(provider: object, base_url: str | None = None) -> str:
    if not provider:
        if base_url and not os.getenv("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = base_url
        return "openai"
    normalized = str(provider).strip().lower()
    if normalized == "openrouter":
        _configure_openrouter_env(base_url)
        return "openai"
    if normalized == "openai":
        if base_url and not os.getenv("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = base_url
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

    raw_provider = (
        env_override.get("provider")
        or entry.get("provider")
        or default.get("provider")
        or "openai"
    )
    base_url = _select_base_url(entry, default)
    provider = _normalize_provider(raw_provider, base_url)
    model_id = (
        env_override.get("model_id")
        or entry.get("model_id")
        or default.get("model_id")
        or "gpt-5.2"
    )

    return provider, model_id
