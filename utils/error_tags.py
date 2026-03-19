from __future__ import annotations

from typing import Optional


def classify_error_tag(error_info: object) -> Optional[str]:
    if not isinstance(error_info, dict):
        return None

    parts = []
    for key in ("error", "traceback", "type"):
        value = error_info.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)

    if not parts:
        return None

    text = " ".join(parts).lower()

    def has_any(patterns: tuple[str, ...]) -> bool:
        return any(pattern in text for pattern in patterns)

    if has_any((
        "invalid api key",
        "authentication",
        "unauthorized",
        "permission denied",
        "401",
    )):
        return "llm_auth_error"

    if has_any(("rate limit", "ratelimit", "429")):
        return "llm_rate_limit"

    if has_any((
        "timeout",
        "timed out",
        "readtimeout",
        "connect timeout",
        "request timed out",
    )):
        return "llm_timeout"

    if has_any((
        "apiconnectionerror",
        "connection error",
        "connecterror",
        "connection refused",
        "connection reset",
        "name or service not known",
        "nodename nor servname",
        "temporary failure in name resolution",
        "name resolution",
        "network is unreachable",
        "dns",
        "getaddrinfo",
    )):
        return "llm_connection_error"

    if has_any((
        "invalid request",
        "bad request",
        "badrequest",
        "invalidrequest",
    )):
        return "llm_invalid_request"

    return None
