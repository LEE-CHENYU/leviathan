import pytest

from MetaIsland.analyze import _fallback_analysis_text
from MetaIsland.llm_utils import (
    EmptyLLMResponseError,
    build_prompt_diagnostics,
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
