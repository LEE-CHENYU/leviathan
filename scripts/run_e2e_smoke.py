#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_metrics import combine_round_metrics, EXPECTED_ROUND_METRICS

IDEALOGY_TEXT = """
[Island Ideology]

Island is a place of abundant resources and opportunity. Agents can:
- Extract resources from land based on land quality
- Build businesses to transform resources into products
- Create contracts with other agents for trade and services
- Propose physical constraints and economic mechanisms
- Form supply chains through interconnected contracts

The economy is entirely agent-driven. Agents write code to:
- Define what resources exist and how they're extracted
- Create markets and pricing mechanisms
- Build production chains and businesses
- Establish labor markets and specialization

Success depends on creating realistic, mutually beneficial economic systems.
""".strip()

CONFIG_PATH = ROOT / "config" / "models.yaml"
_TRUTHY = {"1", "true", "yes", "on"}
_LLM_ERROR_PREFIX = "llm_"
_PREFLIGHT_ENV_KEYS = (
    "E2E_PROVIDER",
    "E2E_MODEL",
    "OPENROUTER_BASE_URL",
    "OPENAI_BASE_URL",
    "LLM_OFFLINE",
    "E2E_OFFLINE",
    "E2E_PREFLIGHT",
    "E2E_FALLBACK_TO_OPENAI",
)
_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def _is_offline() -> bool:
    for key in ("LLM_OFFLINE", "E2E_OFFLINE"):
        value = os.environ.get(key)
        if value and value.strip().lower() in _TRUTHY:
            return True
    return False


def parse_land_shape(value: str) -> Tuple[int, int]:
    value = value.strip().lower().replace("x", ",")
    parts = [p for p in value.split(",") if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("land must be like '6x6' or '6,6'")
    return int(parts[0]), int(parts[1])


def _load_model_config() -> dict:
    try:
        import yaml
    except Exception:
        return {}
    if not CONFIG_PATH.exists():
        return {}
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except Exception:
        return {}
    return config if isinstance(config, dict) else {}


def _load_default_model_id() -> str:
    config = _load_model_config()
    default = config.get("default", {}) if isinstance(config, dict) else {}
    return (default.get("model_id") or "gpt-5.2") if isinstance(default, dict) else "gpt-5.2"


def _load_default_provider() -> str:
    config = _load_model_config()
    default = config.get("default", {}) if isinstance(config, dict) else {}
    if isinstance(default, dict):
        provider = (default.get("provider") or "").strip().lower()
        if provider in {"openrouter", "openai"}:
            return provider
    return "openrouter"


def _coerce_base_url(value: object) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    return text or None


def _load_base_url_for(provider: str) -> Optional[str]:
    provider = (provider or "").strip().lower()
    if not provider:
        return None
    config = _load_model_config()
    if not isinstance(config, dict):
        return None
    default = config.get("default", {}) if isinstance(config.get("default", {}), dict) else {}
    if isinstance(default, dict):
        default_provider = (default.get("provider") or "").strip().lower()
        if default_provider == provider:
            base_url = _coerce_base_url(default.get("base_url"))
            if base_url:
                return base_url
    if provider == "openai":
        benchmark = config.get("benchmark", {}) if isinstance(config.get("benchmark", {}), dict) else {}
        if isinstance(benchmark, dict):
            benchmark_provider = (benchmark.get("provider") or "").strip().lower()
            if benchmark_provider == provider:
                base_url = _coerce_base_url(benchmark.get("base_url"))
                if base_url:
                    return base_url
    return None


def _detect_base_url_source(provider: str, pre_env: dict, cli_overrides: dict) -> str:
    provider = (provider or "").strip().lower()
    if provider == "openrouter":
        if _coerce_base_url(cli_overrides.get("openrouter")):
            return "cli"
        if _coerce_base_url(pre_env.get("OPENROUTER_BASE_URL")):
            return "env"
        if _load_base_url_for("openrouter"):
            return "config"
        return "default"
    if provider == "openai":
        if _coerce_base_url(cli_overrides.get("openai")):
            return "cli"
        if _coerce_base_url(pre_env.get("OPENAI_BASE_URL")):
            return "env"
        if _load_base_url_for("openai"):
            return "config"
        return "default"
    return "unknown"


def _apply_config_base_url(provider: str) -> None:
    base_url = _load_base_url_for(provider)
    if not base_url:
        return
    if provider == "openrouter":
        os.environ.setdefault("OPENROUTER_BASE_URL", base_url)
    elif provider == "openai":
        os.environ.setdefault("OPENAI_BASE_URL", base_url)


def _normalize_base_url(base_url: str, fallback: str) -> str:
    candidate = (base_url or "").strip()
    if not candidate:
        return fallback
    parsed = urlparse(candidate)
    if not parsed.scheme:
        candidate = f"https://{candidate.lstrip('/')}"
        parsed = urlparse(candidate)
    if not parsed.netloc:
        return fallback
    return candidate


def _canonical_base_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    normalized = _normalize_base_url(base_url, base_url)
    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.hostname:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{parsed.hostname.lower()}:{port}{path}"


def _base_urls_match(left: Optional[str], right: Optional[str]) -> bool:
    left_canon = _canonical_base_url(left)
    right_canon = _canonical_base_url(right)
    return bool(left_canon and right_canon and left_canon == right_canon)


def _openrouter_base_url() -> str:
    return _normalize_base_url(
        os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "https://openrouter.ai/api/v1",
    )


def _sanitize_proxy_url(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    parsed = urlparse(candidate)
    if parsed.scheme and parsed.hostname:
        host = parsed.hostname
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{host}{port}"
    parsed = urlparse(f"http://{candidate}")
    if parsed.hostname:
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.hostname}{port}"
    return candidate


def _proxy_env_snapshot() -> dict:
    snapshot = {}
    for key in _PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if not value:
            continue
        if "no_proxy" in key.lower():
            snapshot[key] = value.strip()
        else:
            snapshot[key] = _sanitize_proxy_url(value)
    return snapshot


def _dns_snapshot(base_url: Optional[str]) -> dict:
    if not base_url:
        return {"ok": False, "error": "missing base_url"}
    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        return {"ok": False, "error": f"invalid base_url ({base_url})"}
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port)
        addresses = sorted({info[4][0] for info in infos if info and info[4]})
        return {"ok": True, "host": host, "port": port, "addresses": addresses[:5]}
    except Exception as exc:
        return {"ok": False, "host": host, "port": port, "error": str(exc)}


def _select_provider() -> Tuple[str, str]:
    explicit = os.environ.get("E2E_PROVIDER", "").strip().lower()
    if explicit == "auto":
        explicit = ""
    if explicit:
        if explicit not in {"openrouter", "openai"}:
            raise SystemExit(f"Unsupported E2E_PROVIDER '{explicit}'")
        return explicit, "explicit"

    default_provider = _load_default_provider()
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if default_provider == "openrouter":
        if has_openrouter:
            return "openrouter", "default"
        if has_openai:
            return "openai", "openrouter key missing"
    if default_provider == "openai":
        if has_openai:
            return "openai", "default"
        if has_openrouter:
            return "openrouter", "openai key missing"

    if has_openai and not has_openrouter:
        return "openai", "fallback key"
    if has_openrouter and not has_openai:
        return "openrouter", "fallback key"
    return default_provider or "openrouter", "default"


def _configure_e2e_provider() -> Tuple[str, str, str, Optional[str], Optional[str]]:
    if _is_offline():
        return "", "", "", None, None
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_openai_base_url = os.environ.get("OPENAI_BASE_URL")
    provider, provider_reason = _select_provider()
    e2e_model = os.environ.get("E2E_MODEL") or _load_default_model_id()
    if not os.environ.get("LLM_MAX_TOKENS"):
        os.environ["LLM_MAX_TOKENS"] = "1024"
    _apply_config_base_url(provider)

    if provider == "openrouter":
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise SystemExit("OPENROUTER_API_KEY missing for provider=openrouter")
        os.environ["OPENAI_BASE_URL"] = _openrouter_base_url()
        # Use OpenRouter key just for this process
        os.environ["OPENAI_API_KEY"] = openrouter_key
        os.environ.setdefault("MODEL_ROUTE_DEFAULT_PROVIDER", "openai")
        os.environ.setdefault("MODEL_ROUTE_DEFAULT_MODEL_ID", e2e_model)
        os.environ.setdefault("MODEL_ROUTE_DEEPSEEK_PROVIDER", "openai")
        os.environ.setdefault("MODEL_ROUTE_DEEPSEEK_MODEL_ID", e2e_model)
        return provider, e2e_model, provider_reason, original_openai_key, original_openai_base_url

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing for provider=openai")
    os.environ["OPENAI_BASE_URL"] = _normalize_base_url(
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "https://api.openai.com/v1",
    )
    os.environ.setdefault("LLM_USE_MAX_COMPLETION_TOKENS", "1")
    os.environ.setdefault("MODEL_ROUTE_DEFAULT_PROVIDER", "openai")
    os.environ.setdefault("MODEL_ROUTE_DEFAULT_MODEL_ID", e2e_model or "gpt-5.2")
    os.environ.setdefault("MODEL_ROUTE_DEEPSEEK_PROVIDER", "openai")
    os.environ.setdefault("MODEL_ROUTE_DEEPSEEK_MODEL_ID", e2e_model or "gpt-5.2")
    return provider, e2e_model or "gpt-5.2", provider_reason, original_openai_key, original_openai_base_url


def _configure_openai_fallback(
    model_id: str,
    original_openai_key: Optional[str],
    original_openai_base_url: Optional[str],
    openrouter_base_url: Optional[str] = None,
) -> Tuple[str, str]:
    key = original_openai_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY missing for fallback to openai")
    os.environ["OPENAI_API_KEY"] = key
    base_url, base_url_source = _choose_openai_fallback_base_url(
        original_openai_base_url,
        openrouter_base_url,
    )
    os.environ["OPENAI_BASE_URL"] = _normalize_base_url(base_url, "https://api.openai.com/v1")
    os.environ.setdefault("LLM_USE_MAX_COMPLETION_TOKENS", "1")
    os.environ["MODEL_ROUTE_DEFAULT_PROVIDER"] = "openai"
    os.environ["MODEL_ROUTE_DEFAULT_MODEL_ID"] = model_id
    os.environ["MODEL_ROUTE_DEEPSEEK_PROVIDER"] = "openai"
    os.environ["MODEL_ROUTE_DEEPSEEK_MODEL_ID"] = model_id
    return os.environ["OPENAI_BASE_URL"], base_url_source


def _choose_openai_fallback_base_url(
    original_openai_base_url: Optional[str],
    openrouter_base_url: Optional[str],
) -> Tuple[str, str]:
    candidate = original_openai_base_url or os.environ.get("OPENAI_BASE_URL")
    if candidate and _base_urls_match(candidate, openrouter_base_url):
        config_base = _load_base_url_for("openai")
        if config_base and not _base_urls_match(config_base, openrouter_base_url):
            return _normalize_base_url(config_base, "https://api.openai.com/v1"), "config"
        return "https://api.openai.com/v1", "default"
    if candidate:
        return _normalize_base_url(candidate, "https://api.openai.com/v1"), "env"
    config_base = _load_base_url_for("openai")
    if config_base:
        return _normalize_base_url(config_base, "https://api.openai.com/v1"), "config"
    return "https://api.openai.com/v1", "default"


def _extract_text(body: str) -> Tuple[str, str]:
    try:
        data = json.loads(body)
    except Exception:
        return "", "(unparsed response)"
    choices = data.get("choices") or []
    if not choices:
        return "", "(no choices in response)"
    choice = choices[0]
    msg = choice.get("message", {})
    text = msg.get("content") or choice.get("text") or ""
    return text.strip(), ""


def _http_post(url: str, headers: dict, payload: dict, timeout: int = 20) -> Tuple[bool, Optional[int], str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
            return True, resp.getcode(), body
    except urllib.error.HTTPError as exc:
        return False, exc.code, exc.read(300).decode("utf-8", "ignore")
    except Exception as exc:
        return False, None, str(exc)


def _diagnose_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        return f"Base URL invalid ({base_url})"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        socket.getaddrinfo(host, port)
    except Exception as exc:
        return f"DNS lookup failed for {host}:{port} ({exc})"
    return f"DNS lookup ok for {host}:{port}"


def _preflight_llm(provider: str, model_id: str) -> Tuple[bool, str, str]:
    provider = provider.strip().lower()
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            return False, "OPENROUTER_API_KEY missing", ""
        base_url = _openrouter_base_url().rstrip("/")
        ok, code, body = _http_post(
            f"{base_url}/chat/completions",
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "leviathan-e2e-preflight",
            },
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Return the single word PONG."}],
                "max_tokens": 128,
                "temperature": 0,
            },
        )
        if not ok:
            diag = _diagnose_base_url(base_url) if code is None else ""
            suffix = f" | {diag}" if diag else ""
            return False, f"FAIL ({code}): {body.replace(chr(10), ' ')[:200]}{suffix}", base_url
        text, note = _extract_text(body)
        if not text:
            return False, f"OK but empty response {note}", base_url
        return True, f"OK: {text[:60]}", base_url

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            return False, "OPENAI_API_KEY missing", ""
        base_url = _normalize_base_url(
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "https://api.openai.com/v1",
        ).rstrip("/")
        ok, code, body = _http_post(
            f"{base_url}/chat/completions",
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Return the single word PONG."}],
                "max_completion_tokens": 16,
                "temperature": 0,
            },
        )
        if not ok:
            diag = _diagnose_base_url(base_url) if code is None else ""
            suffix = f" | {diag}" if diag else ""
            return False, f"FAIL ({code}): {body.replace(chr(10), ' ')[:200]}{suffix}", base_url
        text, note = _extract_text(body)
        if not text:
            return False, f"OK but empty response {note}", base_url
        return True, f"OK: {text[:60]}", base_url

    return False, f"Unsupported provider '{provider}'", ""


def _configure_writable_cache(output_dir: Path) -> None:
    cache_root = output_dir / ".cache"
    mpl_cache = cache_root / "matplotlib"
    font_cache = cache_root / "fontconfig"
    cache_root.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    font_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))


def _compute_coverage(expected, metrics):
    if not isinstance(metrics, dict):
        metrics = {}
    missing = [key for key in expected if metrics.get(key) is None]
    coverage = (len(expected) - len(missing)) / len(expected) if expected else None
    return missing, coverage


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_gini(values: list[float]) -> Optional[float]:
    if not values:
        return None
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    sorted_vals = sorted(cleaned)
    total = float(sum(sorted_vals))
    if total <= 0:
        return 0.0
    cumulative = 0.0
    weighted_sum = 0.0
    for idx, value in enumerate(sorted_vals, start=1):
        cumulative += value
        weighted_sum += cumulative
    n = len(sorted_vals)
    gini = (n + 1 - 2 * weighted_sum / total) / n
    return max(0.0, min(1.0, float(gini)))


def _derive_round_context_from_snapshot(snapshot: dict) -> dict:
    if not isinstance(snapshot, dict) or not snapshot:
        return {}
    cargo_vals: list[float] = []
    land_vals: list[float] = []
    wealth_vals: list[float] = []
    for entry in snapshot.values():
        if not isinstance(entry, dict):
            continue
        cargo = _safe_float(entry.get("cargo"))
        land = _safe_float(entry.get("land"))
        if cargo is not None:
            cargo_vals.append(cargo)
        if land is not None:
            land_vals.append(land)
        if cargo is not None and land is not None:
            wealth_vals.append(cargo + land)
    return {
        "gini_cargo": _compute_gini(cargo_vals),
        "gini_land": _compute_gini(land_vals),
        "gini_wealth": _compute_gini(wealth_vals),
    }


def _collect_llm_error_counts(metrics_list) -> Tuple[dict, int]:
    counts: dict = {}
    for metrics in metrics_list:
        if not isinstance(metrics, dict):
            continue
        for key in (
            "agent_code_error_tag_counts",
            "agent_code_error_type_counts",
            "mechanism_error_type_counts",
        ):
            for tag, count in (metrics.get(key) or {}).items():
                if not str(tag).startswith(_LLM_ERROR_PREFIX):
                    continue
                try:
                    count_value = int(count)
                except Exception:
                    count_value = 1
                if tag in counts:
                    counts[tag] = max(counts[tag], count_value)
                else:
                    counts[tag] = count_value
    return counts, sum(counts.values())


def _strict_llm_required() -> bool:
    return _bool_env("E2E_STRICT_LLM", default=True)


def _preflight_env_snapshot() -> dict:
    return {
        key: os.environ.get(key)
        for key in _PREFLIGHT_ENV_KEYS
    }


def _preflight_snapshot(preflight: dict) -> dict:
    base_url = preflight.get("base_url")
    fallback = preflight.get("fallback")
    snapshot = {
        "provider": preflight.get("provider"),
        "model_id": preflight.get("model_id"),
        "provider_reason": preflight.get("provider_reason"),
        "base_url": base_url,
        "base_url_source": preflight.get("base_url_source"),
        "message": preflight.get("message"),
        "fallback": fallback,
        "env": _preflight_env_snapshot(),
        "proxy_env": _proxy_env_snapshot(),
        "dns_check": _dns_snapshot(base_url),
        "secrets_present": {
            "OPENROUTER_API_KEY_SET": bool(os.environ.get("OPENROUTER_API_KEY")),
            "OPENAI_API_KEY_SET": bool(os.environ.get("OPENAI_API_KEY")),
        },
    }
    if isinstance(fallback, dict):
        snapshot["fallback_dns_check"] = _dns_snapshot(fallback.get("base_url"))
    return snapshot


def _preflight_hints(preflight: dict) -> list[str]:
    message = (preflight.get("message") or "").lower()
    fallback_message = ""
    fallback = preflight.get("fallback")
    if isinstance(fallback, dict):
        fallback_message = (fallback.get("message") or "").lower()
    proxy_env = _proxy_env_snapshot()
    hints: list[str] = []
    if "api_key missing" in message or "api_key missing" in fallback_message:
        hints.append("Set the missing API key for the selected provider (OPENROUTER_API_KEY or OPENAI_API_KEY).")
    if "dns lookup failed" in message or "nodename nor servname" in message or "name or service not known" in message:
        hints.append(
            "DNS resolution failed. If you have an OpenAI-compatible gateway, set OPENAI_BASE_URL to a reachable host "
            "(e.g., http://localhost:8000/v1) and E2E_PROVIDER=openai. If relying on OpenAI fallback from OpenRouter, "
            "set OPENAI_BASE_URL before running so the fallback uses it."
        )
        hints.append(
            "You can also pass --openai-base-url/--openrouter-base-url to scripts/run_e2e_smoke.py to avoid env setup."
        )
        hints.append(
            "Alternatively, set base_url in config/models.yaml under default or benchmark to point at your gateway."
        )
        hints.append("If you proxy OpenRouter, set OPENROUTER_BASE_URL to a reachable host.")
        if proxy_env:
            hints.append(
                "Proxy environment detected; ensure HTTPS_PROXY/HTTP_PROXY include a scheme and the proxy host resolves."
            )
        else:
            hints.append(
                "If your network requires a proxy, set HTTPS_PROXY (or HTTP_PROXY) to a reachable gateway."
            )
    if "dns lookup failed" in fallback_message or "nodename nor servname" in fallback_message:
        hints.append("Fallback OpenAI DNS failed as well; verify DNS/network access in this environment.")
    if not hints:
        hints.append("Override provider selection with E2E_PROVIDER=openai or E2E_PROVIDER=openrouter if needed.")
    return hints


def _emit_preflight_diagnostics(preflight: dict) -> None:
    snapshot = _preflight_snapshot(preflight)
    print("Preflight diagnostics:", file=sys.stderr)
    print(json.dumps(snapshot, indent=2), file=sys.stderr)
    hints = _preflight_hints(preflight)
    if hints:
        print("Preflight hints:", file=sys.stderr)
        for hint in hints:
            print(f"- {hint}", file=sys.stderr)


def _persist_preflight_snapshot(output_dir: Path, preflight: dict) -> Optional[Path]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = _preflight_snapshot(preflight)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest = output_dir / "latest_preflight.json"
        stamped = output_dir / f"preflight_{timestamp}.json"
        payload = json.dumps(snapshot, indent=2) + "\n"
        latest.write_text(payload)
        stamped.write_text(payload)
        return latest
    except Exception as exc:
        print(f"Warning: failed to write preflight snapshot: {exc}", file=sys.stderr)
        return None


async def run(
    rounds: int,
    members: int,
    land: Tuple[int, int],
    seed: int,
    output_dir: Path,
    preflight: dict | None = None,
) -> dict:
    from MetaIsland.metaIsland import IslandExecution

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    exec = IslandExecution(members, land, str(run_dir), seed)
    IslandExecution._RECORD_PERIOD = 1
    exec.island_ideology = IDEALOGY_TEXT

    for _ in range(rounds):
        await exec.run_round_with_graph()

    round_record = exec.execution_history['rounds'][-1] if exec.execution_history.get('rounds') else {}
    round_metrics = round_record.get('round_metrics') or {}
    round_end_metrics = round_record.get('round_end_metrics') or {}
    round_context = round_record.get("round_context") or {}
    round_context_source = "round_record" if round_context else None
    if not round_context:
        end_snapshot = round_record.get("round_end_snapshot") or {}
        if end_snapshot:
            round_context = _derive_round_context_from_snapshot(end_snapshot)
            round_context_source = "round_end_snapshot"
        else:
            start_snapshot = round_record.get("round_start_snapshot") or {}
            if start_snapshot:
                round_context = _derive_round_context_from_snapshot(start_snapshot)
                round_context_source = "round_start_snapshot"
    contract_stats = round_record.get('contract_stats')
    physics_stats = round_record.get('physics_stats')
    contract_partner_counts = round_record.get('contract_partner_counts')
    contract_partner_top_share = round_record.get('contract_partner_top_share')
    contract_partner_top_partner = round_record.get('contract_partner_top_partner')
    contract_status_by_party = round_record.get('contract_status_by_party')

    exec_history_file = None
    history_dir = Path(exec.execution_history_path)
    if history_dir.exists():
        files = sorted(history_dir.glob("execution_history_*.json"), key=lambda p: p.stat().st_mtime)
        if files:
            exec_history_file = str(files[-1])

    expected_round_metrics = EXPECTED_ROUND_METRICS
    expected_round_context = [
        "gini_cargo",
        "gini_land",
        "gini_wealth",
    ]
    expected_contract_stats = [
        "total_contracts",
        "pending",
        "active",
        "completed",
        "failed",
    ]
    expected_physics_stats = [
        "active_constraints",
        "domains",
    ]
    expected_contract_partner_stats = [
        "contract_partner_counts",
        "contract_partner_top_share",
        "contract_partner_top_partner",
        "contract_status_by_party",
    ]
    combined_metrics, derived_metrics, derived_sources, missing_metrics, coverage = (
        combine_round_metrics(round_record, expected_round_metrics)
    )
    llm_error_tag_counts, llm_error_total = _collect_llm_error_counts(
        [round_metrics, derived_metrics]
    )
    round_context_missing, round_context_coverage = _compute_coverage(
        expected_round_context,
        round_context,
    )
    contract_stats_missing, contract_stats_coverage = _compute_coverage(
        expected_contract_stats,
        contract_stats,
    )
    physics_stats_missing, physics_stats_coverage = _compute_coverage(
        expected_physics_stats,
        physics_stats,
    )
    contract_partner_snapshot = {
        "contract_partner_counts": contract_partner_counts,
        "contract_partner_top_share": contract_partner_top_share,
        "contract_partner_top_partner": contract_partner_top_partner,
        "contract_status_by_party": contract_status_by_party,
    }
    contract_partner_missing, contract_partner_coverage = _compute_coverage(
        expected_contract_partner_stats,
        contract_partner_snapshot,
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "rounds": rounds,
        "members": members,
        "land_shape": list(land),
        "seed": seed,
        "round_number": round_record.get("round_number"),
        "round_metrics": round_metrics,
        "round_metrics_derived": derived_metrics,
        "round_metrics_derived_sources": derived_sources,
        "round_metrics_combined": combined_metrics,
        "round_metrics_missing": missing_metrics,
        "round_metrics_expected": expected_round_metrics,
        "round_metrics_coverage": coverage,
        "round_context": round_context,
        "round_context_source": round_context_source,
        "round_context_missing": round_context_missing,
        "round_context_expected": expected_round_context,
        "round_context_coverage": round_context_coverage,
        "llm_error_tag_counts": llm_error_tag_counts,
        "llm_error_total": llm_error_total,
        "llm_preflight": preflight or {},
        "round_end_metrics": round_end_metrics,
        "contract_stats": contract_stats or {},
        "contract_stats_missing": contract_stats_missing,
        "contract_stats_expected": expected_contract_stats,
        "contract_stats_coverage": contract_stats_coverage,
        "contract_partner_counts": contract_partner_counts or {},
        "contract_partner_top_share": contract_partner_top_share or {},
        "contract_partner_top_partner": contract_partner_top_partner or {},
        "contract_status_by_party": contract_status_by_party or {},
        "contract_partner_stats_missing": contract_partner_missing,
        "contract_partner_stats_expected": expected_contract_partner_stats,
        "contract_partner_stats_coverage": contract_partner_coverage,
        "physics_stats": physics_stats or {},
        "physics_stats_missing": physics_stats_missing,
        "physics_stats_expected": expected_physics_stats,
        "physics_stats_coverage": physics_stats_coverage,
        "execution_history_file": exec_history_file,
        "save_path": str(run_dir),
    }

    latest_path = output_dir / "latest_summary.json"
    stamped_path = output_dir / f"summary_{run_id}.json"
    latest_path.write_text(json.dumps(summary, indent=2) + "\n")
    stamped_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("E2E smoke run summary:")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal end-to-end MetaIsland smoke test.")
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("E2E_ROUNDS", 1)))
    parser.add_argument("--members", type=int, default=int(os.environ.get("E2E_MEMBERS", 3)))
    parser.add_argument("--land", type=parse_land_shape, default=parse_land_shape(os.environ.get("E2E_LAND", "6x6")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("E2E_SEED", 2023)))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("E2E_OUTPUT_DIR", "execution_histories/e2e_smoke")))
    parser.add_argument("--openai-base-url", type=str, default=None)
    parser.add_argument("--openrouter-base-url", type=str, default=None)
    args = parser.parse_args()

    _configure_writable_cache(args.output_dir)
    load_dotenv()
    pre_env = {
        "OPENROUTER_BASE_URL": os.environ.get("OPENROUTER_BASE_URL"),
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
    }
    cli_base_urls = {
        "openrouter": args.openrouter_base_url,
        "openai": args.openai_base_url,
    }
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    if args.openrouter_base_url:
        os.environ["OPENROUTER_BASE_URL"] = args.openrouter_base_url
    if _is_offline():
        message = "Offline flags are set (LLM_OFFLINE/E2E_OFFLINE), but e2e requires a live LLM."
        if _strict_llm_required():
            raise SystemExit(message)
        print(f"Warning: {message}", file=sys.stderr)

    provider, model_id, provider_reason, original_openai_key, original_openai_base_url = (
        _configure_e2e_provider() if not _is_offline() else ("", "", "", None, None)
    )
    preflight_enabled = _bool_env("E2E_PREFLIGHT", default=not _is_offline())
    preflight_result = {"enabled": preflight_enabled}
    if provider:
        base_url_source = _detect_base_url_source(provider, pre_env, cli_base_urls)
        preflight_result.update(
            {
                "provider": provider,
                "model_id": model_id,
                "provider_reason": provider_reason,
                "base_url_source": base_url_source,
            }
        )
    if preflight_enabled and provider:
        ok, msg, base_url = _preflight_llm(provider, model_id)
        preflight_result.update(
            {
                "ok": ok,
                "message": msg,
                "base_url": base_url,
            }
        )
        if not ok and provider == "openrouter" and _bool_env("E2E_FALLBACK_TO_OPENAI", default=True):
            fallback_model = os.environ.get("E2E_FALLBACK_MODEL", "gpt-5.2")
            fallback_error = None
            fallback_ok = False
            fallback_msg = ""
            fallback_base = ""
            fallback_base_source = ""
            try:
                fallback_base, fallback_base_source = _configure_openai_fallback(
                    fallback_model,
                    original_openai_key,
                    original_openai_base_url,
                    base_url,
                )
                fallback_ok, fallback_msg, fallback_base = _preflight_llm("openai", fallback_model)
            except SystemExit as exc:
                fallback_error = str(exc)
            preflight_result["fallback"] = {
                "provider": "openai",
                "model_id": fallback_model,
                "ok": fallback_ok,
                "message": fallback_msg or fallback_error or "",
                "base_url": fallback_base,
                "base_url_source": fallback_base_source,
            }
            if fallback_ok:
                provider = "openai"
                model_id = fallback_model
                provider_reason = "fallback"
                preflight_result.update(
                    {
                        "provider": provider,
                        "model_id": model_id,
                        "provider_reason": provider_reason,
                        "ok": True,
                        "message": f"fallback ok: {fallback_msg}",
                        "base_url": fallback_base,
                        "base_url_source": fallback_base_source,
                    }
                )
                ok = True
        if not ok and _strict_llm_required():
            _emit_preflight_diagnostics(preflight_result)
            snapshot_path = _persist_preflight_snapshot(args.output_dir, preflight_result)
            if snapshot_path:
                print(f"Preflight snapshot written to {snapshot_path}", file=sys.stderr)
            if isinstance(preflight_result.get("fallback"), dict):
                fb_msg = preflight_result["fallback"].get("message", "")
                if fb_msg:
                    raise SystemExit(
                        f"LLM preflight failed for {provider}: {msg}; fallback openai failed: {fb_msg}"
                    )
            raise SystemExit(f"LLM preflight failed for {provider}: {msg}")

    summary = asyncio.run(
        run(args.rounds, args.members, args.land, args.seed, args.output_dir, preflight_result)
    )
    if _strict_llm_required() and summary.get("llm_error_total", 0):
        print(
            "E2E smoke detected LLM infra errors; results are not valid for evaluation.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
