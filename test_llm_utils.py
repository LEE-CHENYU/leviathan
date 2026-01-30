import pytest

from MetaIsland.analyze import _fallback_analysis_text
from MetaIsland.llm_utils import (
    EmptyLLMResponseError,
    build_code_stats,
    build_prompt_diagnostics,
    describe_syntax_error,
    ensure_non_empty_response,
    extract_completion_metadata,
    extract_request_metadata,
    merge_prompt_sections,
)
from MetaIsland.metaIsland import IslandExecution


def test_ensure_non_empty_response_strips_and_raises():
    assert ensure_non_empty_response("  ok  ") == "ok"
    with pytest.raises(EmptyLLMResponseError):
        ensure_non_empty_response("   ")
    with pytest.raises(EmptyLLMResponseError):
        ensure_non_empty_response(None)


def test_fallback_analysis_text_parses_strategy_card():
    text = _fallback_analysis_text(1)
    engine = IslandExecution.__new__(IslandExecution)
    card = IslandExecution._extract_strategy_card(engine, text)
    assert card is not None
    assert card.get("baseline_signature")
    assert card.get("variation_signature")


def test_extract_strategy_card_splits_plus_signatures():
    text = "\n".join([
        "```json",
        "{",
        "  \"hypothesis\": \"test\",",
        "  \"baseline_signature\": \"expand + message + offer\",",
        "  \"variation_signature\": [\"offer_land + expand\"],",
        "  \"success_metrics\": [\"delta_survival\"]",
        "}",
        "```",
    ])
    engine = IslandExecution.__new__(IslandExecution)
    card = IslandExecution._extract_strategy_card(engine, text)
    assert card["baseline_signature"] == ["expand", "message", "offer"]
    assert card["variation_signature"] == ["offer_land", "expand"]


def test_strategy_memory_appendable_wraps_dict():
    engine = IslandExecution.__new__(IslandExecution)

    class DummyMember:
        pass

    member = DummyMember()
    member.strategy_memory = {"auto_notes": ["x"]}
    engine._ensure_strategy_memory_appendable(member)
    member.strategy_memory.append("note")
    assert member.strategy_memory["notes"][-1] == "note"
    assert member.strategy_memory["auto_notes"] == ["x"]


def test_extract_completion_metadata_handles_objects():
    class DummyUsage:
        def __init__(self):
            self.prompt_tokens = 12
            self.completion_tokens = 34
            self.total_tokens = 46

    class DummyChoice:
        def __init__(self, finish_reason):
            self.finish_reason = finish_reason

    class DummyCompletion:
        def __init__(self):
            self.model = "test-model"
            self.choices = [DummyChoice("length")]
            self.usage = DummyUsage()

    metadata = extract_completion_metadata(DummyCompletion())

    assert metadata == {
        "model": "test-model",
        "finish_reason": "length",
        "prompt_tokens": 12,
        "completion_tokens": 34,
        "total_tokens": 46,
    }


def test_extract_request_metadata_filters_empty_values():
    assert extract_request_metadata(None) == {}
    assert extract_request_metadata({}) == {}
    assert extract_request_metadata({"max_tokens": 256, "temperature": 0.7}) == {
        "request_max_tokens": 256,
        "request_temperature": 0.7,
    }
    assert extract_request_metadata({"max_tokens": None, "max_completion_tokens": 128}) == {
        "request_max_completion_tokens": 128,
    }


def test_build_prompt_diagnostics_counts_sections():
    diagnostics = build_prompt_diagnostics(
        "hi123",
        {"alpha": "hi", "beta": None, "gamma": 123},
    )
    assert diagnostics["prompt_char_count"] == 5
    assert diagnostics["prompt_section_chars"]["alpha"] == 2
    assert diagnostics["prompt_section_chars"]["beta"] == 0
    assert diagnostics["prompt_section_chars"]["gamma"] == 3
    assert diagnostics["prompt_dynamic_char_total"] == 5
    assert diagnostics["prompt_dynamic_char_ratio"] == pytest.approx(1.0)


def test_merge_prompt_sections_flattens_breakdowns():
    base = {"alpha": "a"}
    breakdown = {"island": "ii", "member": None}
    merged = merge_prompt_sections(base, breakdown, "base_code_")
    assert merged["alpha"] == "a"
    assert merged["base_code_island"] == "ii"
    assert merged["base_code_member"] is None


def test_describe_syntax_error_extracts_details():
    with pytest.raises(SyntaxError) as exc_info:
        compile("def broken(:\n  pass", "<test>", "exec")
    details = describe_syntax_error(exc_info.value)
    assert details["error_type"] == "SyntaxError"
    assert details["error_line"] == 1
    assert "def broken" in details.get("error_text", "")


def test_describe_syntax_error_includes_context_lines():
    code = "def ok():\n    return 1\n\ndef broken(:\n    return 2\n"
    with pytest.raises(SyntaxError) as exc_info:
        compile(code, "<test>", "exec")
    details = describe_syntax_error(exc_info.value, code, context_lines=1, max_line_length=40)
    context = details.get("error_context")
    assert context is not None
    assert context["start_line"] == 3
    assert context["end_line"] == 5
    assert any(
        entry["line"] == 4 and "def broken" in entry["text"]
        for entry in context["lines"]
    )


def test_build_code_stats_counts_lengths():
    stats = build_code_stats("one\ntwo", "one\ntwo\n")
    assert stats["raw_len"] == 7
    assert stats["raw_lines"] == 2
    assert stats["cleaned_len"] == 8
    assert stats["cleaned_lines"] == 3
