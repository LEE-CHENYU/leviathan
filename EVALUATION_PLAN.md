# Evaluation Plan: MetaIsland Strategy Diversity

Goal
- Improve self-improving strategy design while preserving agent/environment diversity.

Source of truth (end-to-end)
- Primary feedback comes from the end-to-end smoke run: `python scripts/run_e2e_smoke.py`.
- The latest summary is written to `execution_histories/e2e_smoke/latest_summary.json`.
- When iteration decisions conflict with unit tests, prefer the e2e summary.
- If the e2e run cannot execute due to missing credentials or provider issues, fix the provider configuration and rerun. Do not use mock or stubbed LLMs for evaluation.
- If `LLM_OFFLINE` / `E2E_OFFLINE` is set or `llm_error_total > 0`, treat the run as invalid and rerun with a working provider.
- The e2e summary now includes `round_metrics_derived` plus `round_metrics_combined`/`round_metrics_coverage` so guardrails can still be inspected when raw `round_metrics` are missing.
- The e2e summary also reports coverage for `contract_stats`, `physics_stats`, and contract partner stats to surface missing instrumentation.
- `llm_preflight` reports whether a live LLM check succeeded before the run; use it to confirm connectivity.

Metrics (tracked per round)
- Population survival (end-of-round): `round_end_population_avg_survival_delta`, `round_end_population_std_survival_delta`.
- Action-phase survival (pre-contract/produce/consume): `population_avg_survival_delta`, `population_std_survival_delta`.
- Diversity health: `population_signature_unique_ratio`, `population_signature_entropy`, `population_signature_dominant_share`.
- Dominant signature alignment: rate of actions matching the population-dominant signature.
- Environment diversity: `round_context.gini_wealth`, `round_context.gini_cargo`, `round_context.gini_land`.
- Agent outcomes: per-agent `delta_survival`, `delta_vitality`, `delta_cargo`.
- Experiment outcomes: baseline vs variation avg `delta_survival` (from experiment summaries).
- Execution reliability: plan/execution alignment rate via `round_metrics.plan_alignment_rate` (planned signature equals executed signature) and plan coverage via `round_metrics.plan_alignment_plan_coverage`.
- Feasibility follow-through: `round_metrics.plan_ineligible_tag_rate` (planned tags infeasible), `round_metrics.plan_only_tag_rate` (feasible planned tags not executed), plus `round_metrics.plan_feasibility_samples`.
- Agent code errors: `round_metrics.agent_code_error_count`, `round_metrics.agent_code_error_rate`, and `round_metrics.agent_code_error_tag_counts` (to spot recurring failure tags; treat `llm_connection_error`, `llm_auth_error`, `llm_rate_limit`, `llm_timeout` as infra signals).
- LLM truncation diagnostics: inspect `generated_code[*].llm_metadata.finish_reason` (e.g., `length`) plus `errors.agent_code_errors[*].code_stats`/`error_details` to confirm syntax errors correlate with truncated outputs.
- LLM finish_reason counts: `round_metrics.llm_finish_reason_counts`, `round_metrics.llm_finish_reason_total`, `round_metrics.llm_finish_reason_length_count`, and `round_metrics.llm_finish_reason_missing_count` to quantify truncation signals across agent + mechanism LLM calls.
- LLM request limits: inspect `generated_code[*].llm_metadata.request_max_tokens` / `request_max_completion_tokens` alongside `completion_tokens` to confirm whether truncation aligns with requested caps.
- LLM cap diagnostics (derived): `round_metrics.llm_request_cap_avg`/`min`/`max`, `llm_request_cap_count`, `llm_completion_at_request_cap_count`, `llm_completion_at_request_cap_rate`, plus `llm_prompt_tokens_avg` and `llm_completion_tokens_avg` to spot prompt pressure and cap saturation in one place.
- Prompt size diagnostics (per attempt): `llm_metadata.prompt_char_count`, `llm_metadata.prompt_section_chars`, `llm_metadata.prompt_dynamic_char_total`, `llm_metadata.prompt_dynamic_char_ratio` to identify which prompt sections dominate when truncation occurs (target those sections for trimming before changing behavior). The base-class section now includes per-class breakdowns (e.g., `base_code_base_island`, `base_code_base_land`, `base_code_base_member`) so the largest code block can be isolated before pruning.
- Empty cleaned outputs: `llm_empty_response` in `agent_code_error_type_counts` or `mechanism_error_type_counts` indicates the LLM returned non-code or code stripped to empty; treat as invalid for behavior evaluation and inspect prompt compliance.
- LLM availability: `llm_error_total` and `llm_error_tag_counts` (should be zero for valid e2e results); `llm_preflight.ok` should be true when enabled.
- Fallback usage: count `errors.agent_code_errors[*].fallback_used` and inspect `fallback_source` to confirm memory reuse vs template default when LLM calls fail.
- Mechanism fallback usage: check `errors.mechanism_errors[*].fallback_used`/`fallback_source`; if non-zero due to infra tags, treat metrics as pipeline-validation only.
- Analysis fallback usage: inspect `errors.analyze_code_errors[*].fallback_used` and confirm plan metrics (`plan_alignment_*`, `plan_ineligible_tag_rate`, `plan_only_tag_rate`) are populated even when LLM calls fail.
- Error categories: `round_metrics.agent_code_error_type_counts` and `round_metrics.mechanism_error_type_counts` to separate LLM connectivity/auth issues from execution faults.
- Contextual fit: average balanced score for entries with context similarity >= 0.35 vs overall average.
- Identity continuity: `memory_active_coverage`, `memory_missing_count`, and `memory_orphan_count` from round metrics (active ids vs code_memory keys).
- Contract activity: `contract_stats.pending`, `contract_stats.active`, `contract_stats.completed`, `contract_stats.failed`.
- Contract diversity: `contract_partner_unique_avg` and `contract_partner_top_share_avg` (round metrics), plus per-agent partner counts in `contract_partner_counts` and `contract_partner_top_share`.
- Physics coverage: `physics_stats.active_constraints`, `physics_stats.domains`.
- Recommendation diversity adjustments: count of strategy recommendations containing "Diversity adjustment" when `population_signature_dominant_share >= 0.60`. Use `python scripts/inspect_execution_history.py` to auto-scan prompts in execution histories (generated_code + mechanism prompts).
- Mechanism gating: `round_metrics.mechanism_attempted_count`, `round_metrics.mechanism_approved_count`, `round_metrics.mechanism_executed_count`, `round_metrics.mechanism_error_count` (cross-check with `mechanism_modifications` and `errors.mechanism_errors`).
- Legacy histories without `round_metrics`: `scripts/inspect_execution_history.py` derives survival deltas (avg/std), agent code error count/rate/tag counts, plan action totals, mechanism/contract/physics counts (when present), and population signature diversity stats from `agent_actions`/`actions` + `errors` and action code as approximate fallbacks.

Baselines
- Use the last 3 completed rounds from `execution_history` as baseline (or first run if none).

Thresholds (guardrails)
- Survival (end-of-round): no worse than baseline by more than 0.02 on `round_end_population_avg_survival_delta`.
- Diversity: `population_signature_dominant_share` must not exceed baseline by >0.05.
- Dominant signature alignment rate should not exceed baseline by >0.05.
- Diversity: `population_signature_unique_ratio` and `population_signature_entropy` should not drop by >0.05.
- Environment: `round_context.gini_wealth` should not drop by >0.05 (avoid homogenizing resources).
- Contract concentration: `contract_partner_top_share_avg` should not exceed baseline by >0.10.
- Mechanism execution must not exceed approved count in any round; mechanism error rate should not exceed baseline by >0.05.
- Experiments: at least 40% of recent actions should match a baseline or variation plan.
- Execution reliability: plan/execution alignment rate should not drop by >0.05 vs baseline.
- Feasibility: `plan_ineligible_tag_rate` and `plan_only_tag_rate` should not exceed baseline by >0.05.
- Agent code errors: `agent_code_error_rate` should not exceed baseline by >0.05; scan `agent_code_error_tag_counts` for recurring high-error tags. If infra tags dominate, rerun with a working provider before judging behavior.
- LLM infra errors (`llm_error_total > 0`, including `llm_empty_response`) invalidate the run; rerun with a working provider before judging behavior.
- LLM truncation: `llm_finish_reason_length_count` should be 0; if >0, treat behavior metrics as invalid and rerun with higher max tokens or trimmed prompts.
- LLM cap saturation: if `llm_completion_at_request_cap_rate` is elevated in the same run as `llm_finish_reason_length_count > 0`, prioritize increasing `LLM_MAX_TOKENS`/`LLM_MAX_COMPLETION_TOKENS` and rerun before judging behavior.
- Fallback usage should be 0 with a healthy LLM provider; if >0, treat metrics as invalid and rerun with a working provider.
- Identity: `memory_active_coverage` should not drop by >0.05 vs baseline; `memory_missing_count` should be 0 after action rounds; track `memory_orphan_count` (expected to rise with deaths).

Expected deltas (small wins)
- `population_signature_unique_ratio` +0.05.
- `population_signature_dominant_share` -0.05.
- Dominant signature alignment rate -0.05.
- `round_context.gini_wealth` stable or +0.02 (preserve heterogeneity).
- Experiment match rate >= 60%.
- Contextual fit: context-matched entries show >= +0.02 balanced score vs overall.
- Feasibility: `plan_ineligible_tag_rate` and `plan_only_tag_rate` -0.05.
- Contract diversity: `contract_partner_unique_avg` +1 without increasing failures.
- Recommendation diversity: when population dominance is high, expect some diversity adjustments in recommendations without reducing survival guardrails.

Stage-gated evaluation
- Small (e2e smoke): 1–3 rounds, 3–5 agents. Use `scripts/run_e2e_smoke.py` and validate guardrails.
- Medium (e2e): 5 rounds, 5–8 agents. Expect diversity metrics to improve without survival drop.
- Large: only if medium shows positive deltas.
