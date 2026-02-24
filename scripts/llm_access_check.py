#!/usr/bin/env python
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yaml"

load_dotenv()


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


def _coerce_base_url(value):
    if not value:
        return None
    text = str(value).strip()
    return text or None


def _config_base_url_for(provider: str):
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


def _normalize_base_url(base_url, fallback):
    base_url = (base_url or "").strip()
    if not base_url:
        return fallback
    if not base_url.startswith(("http://", "https://")):
        base_url = f"https://{base_url.lstrip('/')}"
    return base_url.rstrip("/")


def _resolve_base_url(provider: str, env_var: str, fallback: str):
    base_url = os.getenv(env_var) or _config_base_url_for(provider)
    return _normalize_base_url(base_url, fallback)


def _post(url, headers, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", "ignore")
            return True, resp.getcode(), body
    except urllib.error.HTTPError as exc:
        return False, exc.code, exc.read(300).decode("utf-8", "ignore")
    except Exception as exc:
        return False, None, str(exc)


def _extract_text(body):
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


def check_openai():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return False, "OPENAI_API_KEY missing"
    base_url = _resolve_base_url("openai", "OPENAI_BASE_URL", "https://api.openai.com/v1")
    ok, code, body = _post(
        f"{base_url}/chat/completions",
        {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "Return the single word PONG."}],
            "max_completion_tokens": 16,
            "temperature": 0,
        },
    )
    if not ok:
        return False, f"FAIL ({code}): {body.replace(chr(10), ' ')[:200]}"
    text, note = _extract_text(body)
    if not text:
        return False, f"OK but empty response {note}"
    return True, f"OK: {text[:60]}"


def check_openrouter():
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return False, "OPENROUTER_API_KEY missing"
    base_url = _resolve_base_url("openrouter", "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    ok, code, body = _post(
        f"{base_url}/chat/completions",
        {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "leviathan-llm-check",
        },
        {
            "model": "minimax/minimax-m2.1",
            "messages": [{"role": "user", "content": "Return the single word PONG."}],
            "max_tokens": 64,
            "temperature": 0,
        },
    )
    if not ok:
        return False, f"FAIL ({code}): {body.replace(chr(10), ' ')[:200]}"
    text, note = _extract_text(body)
    if not text:
        return False, f"OK but empty response {note}"
    return True, f"OK: {text[:60]}"


def main():
    ok_openai, msg_openai = check_openai()
    ok_openrouter, msg_openrouter = check_openrouter()
    print(f"OpenAI: {msg_openai}")
    print(f"OpenRouter: {msg_openrouter}")
    if not (ok_openai and ok_openrouter):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
