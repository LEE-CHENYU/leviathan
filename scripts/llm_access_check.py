#!/usr/bin/env python
import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()


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
    ok, code, body = _post(
        "https://api.openai.com/v1/chat/completions",
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
    ok, code, body = _post(
        "https://openrouter.ai/api/v1/chat/completions",
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
