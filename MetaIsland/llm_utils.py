import os
from typing import Dict, Any, Optional


class EmptyLLMResponseError(ValueError):
    """Raised when an LLM returns an empty or whitespace-only response."""


def ensure_non_empty_response(text: Optional[str], context: str = "") -> str:
    """Return stripped LLM text or raise when the response is empty."""
    if text is None:
        cleaned = ""
    else:
        cleaned = str(text).strip()
    if not cleaned:
        suffix = f":{context}" if context else ""
        raise EmptyLLMResponseError(f"empty_response{suffix}")
    return cleaned


def _use_max_completion_tokens() -> bool:
    """Return True when OpenAI-style max_completion_tokens is required."""
    if os.getenv("LLM_USE_MAX_COMPLETION_TOKENS"):
        return True
    base_url = os.getenv("OPENAI_BASE_URL", "").lower()
    return "openai.com" in base_url


def build_chat_kwargs() -> Dict[str, Any]:
    """Build shared chat completion kwargs from environment settings."""
    kwargs: Dict[str, Any] = {}

    max_tokens = os.getenv("LLM_MAX_TOKENS")
    max_completion_tokens = os.getenv("LLM_MAX_COMPLETION_TOKENS")
    if _use_max_completion_tokens():
        value = max_completion_tokens or max_tokens
        if value:
            try:
                kwargs["max_completion_tokens"] = int(value)
            except ValueError:
                pass
    elif max_tokens:
        try:
            kwargs["max_tokens"] = int(max_tokens)
        except ValueError:
            pass

    temperature = os.getenv("LLM_TEMPERATURE")
    if temperature is not None:
        try:
            kwargs["temperature"] = float(temperature)
        except ValueError:
            pass

    return kwargs


def extract_request_metadata(request_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract request settings for diagnostics without mutating request kwargs."""
    if not request_kwargs:
        return {}
    metadata: Dict[str, Any] = {}
    if "max_tokens" in request_kwargs:
        metadata["request_max_tokens"] = request_kwargs.get("max_tokens")
    if "max_completion_tokens" in request_kwargs:
        metadata["request_max_completion_tokens"] = request_kwargs.get("max_completion_tokens")
    if "temperature" in request_kwargs:
        metadata["request_temperature"] = request_kwargs.get("temperature")
    return {key: value for key, value in metadata.items() if value is not None}


def _read_attr(obj: Any, attr: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_completion_metadata(completion: Any) -> Dict[str, Any]:
    """Extract minimal metadata from an LLM completion object for diagnostics."""
    if completion is None:
        return {}

    metadata: Dict[str, Any] = {}
    metadata["model"] = _read_attr(completion, "model")

    choices = _read_attr(completion, "choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        metadata["finish_reason"] = _read_attr(first_choice, "finish_reason")

    usage = _read_attr(completion, "usage")
    if usage:
        metadata["prompt_tokens"] = _coerce_int(_read_attr(usage, "prompt_tokens"))
        metadata["completion_tokens"] = _coerce_int(_read_attr(usage, "completion_tokens"))
        metadata["total_tokens"] = _coerce_int(_read_attr(usage, "total_tokens"))

    return {key: value for key, value in metadata.items() if value is not None}


def _trim_context_line(text: str, max_length: int) -> str:
    if max_length and len(text) > max_length:
        return text[:max_length] + "..."
    return text


def describe_syntax_error(
    error: Exception,
    code: Optional[str] = None,
    context_lines: int = 2,
    max_line_length: int = 200,
) -> Dict[str, Any]:
    """Describe syntax errors with line/offset context when available."""
    if not isinstance(error, SyntaxError):
        return {}
    details: Dict[str, Any] = {"error_type": type(error).__name__}
    if error.lineno is not None:
        details["error_line"] = error.lineno
    if error.offset is not None:
        details["error_offset"] = error.offset
    text = getattr(error, "text", None)
    if text:
        details["error_text"] = text.strip()
    msg = getattr(error, "msg", None)
    if msg:
        details["error_msg"] = msg

    if code and error.lineno is not None:
        try:
            lines = str(code).splitlines()
        except Exception:
            lines = []
        if lines:
            line_index = error.lineno - 1
            if line_index < 0 or line_index >= len(lines):
                details["error_line_out_of_range"] = True
                line_index = min(max(line_index, 0), len(lines) - 1)
            span = max(0, context_lines)
            start = max(0, line_index - span)
            end = min(len(lines), line_index + span + 1)
            context_entries = []
            for idx in range(start, end):
                line_text = lines[idx]
                if not isinstance(line_text, str):
                    line_text = str(line_text)
                context_entries.append(
                    {
                        "line": idx + 1,
                        "text": _trim_context_line(line_text, max_line_length),
                    }
                )
            details["error_context"] = {
                "start_line": start + 1,
                "end_line": end,
                "lines": context_entries,
            }
            details["code_line_count"] = len(lines)

    return {key: value for key, value in details.items() if value is not None}


def build_code_stats(raw_code: Optional[str], cleaned_code: Optional[str] = None) -> Dict[str, int]:
    """Build lightweight code length diagnostics for logging."""
    stats: Dict[str, int] = {}
    if raw_code is not None:
        raw_text = str(raw_code)
        stats["raw_len"] = len(raw_text) if raw_text else 0
        stats["raw_lines"] = raw_text.count("\n") + 1 if raw_text else 0
    if cleaned_code is None and raw_code is not None:
        cleaned_code = raw_code
    if cleaned_code is not None:
        cleaned_text = str(cleaned_code)
        stats["cleaned_len"] = len(cleaned_text) if cleaned_text else 0
        stats["cleaned_lines"] = cleaned_text.count("\n") + 1 if cleaned_text else 0
    return stats


def build_prompt_diagnostics(
    prompt: Optional[str],
    sections: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build lightweight prompt size diagnostics for logging."""
    diagnostics: Dict[str, Any] = {}

    if prompt is not None:
        try:
            diagnostics["prompt_char_count"] = len(str(prompt))
        except Exception:
            pass

    if sections:
        section_lengths: Dict[str, int] = {}
        total = 0
        for key, value in sections.items():
            try:
                text = "" if value is None else str(value)
            except Exception:
                text = ""
            length = len(text)
            section_lengths[key] = length
            total += length

        if section_lengths:
            diagnostics["prompt_section_chars"] = section_lengths
            diagnostics["prompt_dynamic_char_total"] = total
            prompt_chars = diagnostics.get("prompt_char_count")
            if isinstance(prompt_chars, int) and prompt_chars > 0:
                try:
                    diagnostics["prompt_dynamic_char_ratio"] = round(
                        total / prompt_chars, 4
                    )
                except Exception:
                    pass

    return diagnostics


def merge_prompt_sections(
    sections: Optional[Dict[str, Any]],
    extra_sections: Optional[Dict[str, Any]],
    prefix: str,
) -> Dict[str, Any]:
    """Merge extra section breakdowns into a flat mapping with a prefix."""
    merged = dict(sections or {})
    if isinstance(extra_sections, dict):
        for key, value in extra_sections.items():
            merged[f"{prefix}{key}"] = value
    return merged


def classify_llm_error(error: Exception) -> str:
    """Classify LLM client errors into coarse categories for diagnostics."""
    if error is None:
        return "llm_unknown"
    if isinstance(error, EmptyLLMResponseError):
        return "llm_empty_response"
    if isinstance(error, SyntaxError):
        return "llm_syntax_error"
    message = str(error)
    type_name = error.__class__.__name__ if hasattr(error, "__class__") else ""
    text = f"{type_name} {message}".lower()

    if not text.strip():
        return "llm_unknown"
    if "empty_response" in text or "empty response" in text:
        return "llm_empty_response"
    if (
        "syntaxerror" in text
        or "invalid syntax" in text
        or "unterminated" in text
        or "unexpected eof" in text
        or "eof while scanning" in text
    ):
        return "llm_syntax_error"

    if "rate limit" in text or "429" in text:
        return "llm_rate_limit"
    if "timeout" in text or "timed out" in text:
        return "llm_timeout"
    if "authentication" in text or "api key" in text or "unauthorized" in text or "401" in text:
        return "llm_auth_error"
    if (
        "connect" in text
        or "connection" in text
        or "nodename nor servname" in text
        or "dns" in text
        or "name resolution" in text
    ):
        return "llm_connection_error"
    if "bad request" in text or "invalid request" in text or "400" in text:
        return "llm_invalid_request"
    if "not found" in text or "404" in text:
        return "llm_not_found"

    return "llm_unknown"


_OFFLINE_FALLBACK_CATEGORIES = {
    "llm_connection_error",
    "llm_timeout",
    "llm_rate_limit",
}


def should_use_offline_fallback(error: Exception) -> bool:
    """Return True when a transient LLM failure should trigger offline fallback."""
    return classify_llm_error(error) in _OFFLINE_FALLBACK_CATEGORIES
