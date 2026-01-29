from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


@lru_cache(maxsize=1)
def _load_model_config():
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def model_router(model):
    config = _load_model_config()
    routes = config.get("routes", {})
    default = config.get("default", {})
    benchmark = config.get("benchmark", {})

    if model in (None, "", "default"):
        entry = default
    elif model in ("benchmark", "eval", "evaluation"):
        entry = benchmark or routes.get("gpt-5.2", {})
    else:
        entry = routes.get(model, {}) if isinstance(routes, dict) else {}

    provider = entry.get("provider") or default.get("provider") or "openai"
    model_id = entry.get("model_id") or default.get("model_id") or "gpt-5.2"

    return provider, model_id
