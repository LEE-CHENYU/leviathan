"""Tests for prompt_loader — verifying principles, awareness sections, and safety templates."""

import pytest

from MetaIsland.prompt_loader import get_prompt_loader, PromptLoader


@pytest.fixture
def loader():
    """Fresh PromptLoader instance (no global cache side-effects)."""
    return PromptLoader()


# Shared kwargs for build_mechanism_prompt / build_action_prompt
_COMMON_KWARGS = dict(
    member_id=0,
    island_ideology="",
    error_context="",
    current_mechanisms="",
    report="",
    features="",
    relations="",
    code_memory="",
    past_performance="",
    analysis_memory="",
    analysis_card_summary="",
    experiment_summary="",
    strategy_profile="",
    population_strategy_profile="",
    population_exploration_summary="",
    strategy_recommendations="",
    contextual_strategy_summary="",
    message_context="",
    communication_summary="",
    base_code="",
)


class TestMechanismDesignPrinciples:
    def test_mechanism_design_principles_included(self, loader):
        """The principles section (CONSERVATION etc.) must appear in mechanism prompts."""
        prompt = loader.build_mechanism_prompt(
            modification_attempts="",
            **_COMMON_KWARGS,
        )
        assert "CONSERVATION" in prompt
        assert "Mechanism Design Principles" in prompt

    def test_old_guardrails_key_not_used(self, loader):
        """The old 'mechanism_design_guardrails' key should not appear."""
        base = loader.get_base_prompts()
        assert "mechanism_design_guardrails" not in base


class TestCanaryAwareness:
    def test_canary_awareness_included(self, loader):
        """Canary awareness section must appear in mechanism prompts."""
        prompt = loader.build_mechanism_prompt(
            modification_attempts="",
            **_COMMON_KWARGS,
        )
        assert "Canary Testing Awareness" in prompt


class TestCheckpointAwareness:
    def test_checkpoint_awareness_included(self, loader):
        """Checkpoint awareness section must appear in mechanism prompts."""
        prompt = loader.build_mechanism_prompt(
            modification_attempts="",
            **_COMMON_KWARGS,
        )
        assert "Checkpoint & Recovery Awareness" in prompt


class TestSafetyTemplates:
    def test_safety_templates_formatted(self, loader):
        """Safety templates should appear in the formatted mechanism templates."""
        formatted = loader.format_mechanism_templates(member_id=0)
        assert "SAFETY TEMPLATES" in formatted
        assert "Vitality Watchdog" in formatted
        assert "Insurance Pool" in formatted
        assert "Circuit Breaker" in formatted

    def test_safety_templates_in_mechanism_prompt(self, loader):
        """Safety templates should also appear in the full mechanism prompt."""
        prompt = loader.build_mechanism_prompt(
            modification_attempts="",
            **_COMMON_KWARGS,
        )
        assert "SAFETY TEMPLATES" in prompt
        assert "Vitality Watchdog" in prompt


class TestNewContextSections:
    def test_canary_summary_in_mechanism_prompt(self, loader):
        """When canary_summary is provided, it appears in mechanism prompt."""
        prompt = loader.build_mechanism_prompt(
            modification_attempts="",
            canary_summary="- vitality_change: -5.0%, died: [], flags: []",
            **_COMMON_KWARGS,
        )
        assert "Canary test results (current round):" in prompt
        assert "vitality_change: -5.0%" in prompt

    def test_canary_summary_in_action_prompt(self, loader):
        """When canary_summary is provided, it appears in action prompt."""
        prompt = loader.build_action_prompt(
            canary_summary="- vitality_change: -3.0%",
            **_COMMON_KWARGS,
        )
        assert "Canary test results (current round):" in prompt
        assert "vitality_change: -3.0%" in prompt

    def test_pending_proposals_in_mechanism_prompt(self, loader):
        """When pending_proposals_summary is provided, it appears in mechanism prompt."""
        prompt = loader.build_mechanism_prompt(
            modification_attempts="",
            pending_proposals_summary="- Proposal abc: submitted round 3, votes: {0: True}",
            **_COMMON_KWARGS,
        )
        assert "Pending proposals awaiting votes:" in prompt
        assert "Proposal abc" in prompt

    def test_checkpoint_summary_in_mechanism_prompt(self, loader):
        """When checkpoint_summary is provided, it appears in mechanism prompt."""
        prompt = loader.build_mechanism_prompt(
            modification_attempts="",
            checkpoint_summary="- Round 2: 5 members, vitality 250.0",
            **_COMMON_KWARGS,
        )
        assert "Available checkpoints:" in prompt
        assert "Round 2: 5 members" in prompt

    def test_empty_context_omitted(self, loader):
        """When context summaries are empty strings, their sections are omitted."""
        prompt = loader.build_mechanism_prompt(
            modification_attempts="",
            canary_summary="",
            pending_proposals_summary="",
            checkpoint_summary="",
            **_COMMON_KWARGS,
        )
        assert "Canary test results (current round):" not in prompt
        assert "Pending proposals awaiting votes:" not in prompt
        assert "Available checkpoints:" not in prompt
