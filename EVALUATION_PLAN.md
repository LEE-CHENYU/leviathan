# Evaluation Plan: MetaIsland Strategy Diversity

Goal
- Improve self-improving strategy design while preserving agent/environment diversity.

Source of truth (end-to-end)
- Primary feedback comes from the end-to-end smoke run: `python scripts/run_e2e_smoke.py`.
- The latest summary is written to `execution_histories/e2e_smoke/latest_summary.json`.
- When iteration decisions conflict with unit tests, prefer the e2e summary.
- If the e2e run cannot execute due to missing credentials or provider issues, fix the provider configuration and rerun. Do not use mock or stubbed LLMs for evaluation.
- If `LLM_OFFLINE` / `E2E_OFFLINE` is set or `llm_error_total` indicates critical infra failures (DNS/auth/rate-limit), treat the run as invalid and rerun with a working provider.
- If `llm_error_total` reflects minor LLM output issues (e.g., syntax/empty/truncation), treat the run as provisional: proceed with behavior iteration but flag the issue and schedule a rerun to confirm.
- The e2e summary now includes `round_metrics_derived` plus `round_metrics_combined`/`round_metrics_coverage` so guardrails can still be inspected when raw `round_metrics` are missing.
- The e2e summary also reports coverage for `contract_stats`, `physics_stats`, and contract partner stats to surface missing instrumentation.
- If `round_context` is absent in the execution history, `scripts/run_e2e_smoke.py` derives `round_context` (gini_cargo/land/wealth) from the latest snapshot and reports `round_context_*` coverage in the summary.
- `llm_preflight` reports whether a live LLM check succeeded before the run; use it to confirm connectivity.

Metrics (tracked per round)
- Population survival (end-of-round): `round_end_population_avg_survival_delta`, `round_end_population_std_survival_delta`.
- Action-phase survival (pre-contract/produce/consume): `population_avg_survival_delta`, `population_std_survival_delta`.
- Diversity health: `population_signature_unique_ratio`, `population_signature_entropy`, `population_signature_dominant_share`.
- Dominant signature alignment: rate of actions matching the population-dominant signature.
- Environment diversity: `round_context.gini_wealth`, `round_context.gini_cargo`, `round_context.gini_land`.
- Resource balance: `round_end_population_avg_cargo_delta` and production/consumption (net production) from `population_state_summary`.
- Round-end pressure cues: when `round_end_population_avg_survival_delta` or `round_end_population_avg_cargo_delta` are negative, expect strategy recommendations to include a "Round-end deltas" line and exploration pressure notes to include "round-end losses" (with severity). When deltas are <= -10, expect an additional "Safety priority" line to bias toward low-cost recovery and baseline-close variations. When resource pressure is active and cargo deltas are available, expect the variation line to include a `resource +/-` adjustment hinting cargo-aware scoring.
- Agent outcomes: per-agent `delta_survival`, `delta_vitality`, `delta_cargo`.
- Experiment outcomes: baseline vs variation avg `delta_survival` (from experiment summaries).
- Execution reliability: plan/execution alignment rate via `round_metrics.plan_alignment_rate` (planned signature equals executed signature) and plan coverage via `round_metrics.plan_alignment_plan_coverage`.
- Feasibility follow-through: `round_metrics.plan_ineligible_tag_rate` (planned tags infeasible), `round_metrics.plan_only_tag_rate` (feasible planned tags not executed), plus `round_metrics.plan_feasibility_samples`.
- Feasibility gaps: `round_metrics.plan_feasibility_missing_reason_counts` to see whether missing feasibility comes from absent experiments, invalid labels, or empty signatures.
- Analysis card tag hygiene: `round_metrics.analysis_card_signature_invalid_tag_rate`, `round_metrics.analysis_card_signature_recoverable_tag_rate`, and `round_metrics.analysis_card_signature_empty_card_rate` to detect malformed baseline/variation tags (e.g., stray punctuation or non-action labels). Baseline/variation signatures now normalize "+"-delimited tag combos into individual tags before storage, so invalid-tag rates should fall when formatting drift is the only issue.
- Analysis card hygiene metrics treat recoverable tags (punctuation/argument noise, dict-like blobs with `actions`) as non-invalid and non-empty so they measure unrecoverable formatting drift. When this logic changes, recompute baselines.
- Agent code errors: `round_metrics.agent_code_error_count`, `round_metrics.agent_code_error_rate`, and `round_metrics.agent_code_error_tag_counts` (to spot recurring failure tags; treat `llm_connection_error`, `llm_auth_error`, `llm_rate_limit`, `llm_timeout` as infra signals; treat `llm_syntax_error` as invalid generated code, likely truncation).
- LLM truncation diagnostics: inspect `generated_code[*].llm_metadata.finish_reason` (e.g., `length`) plus `errors.agent_code_errors[*].code_stats`/`error_details` and `errors.mechanism_errors[*].code_stats`/`error_details` to confirm syntax errors correlate with truncated outputs. When available, use `error_details.error_context` to see the line window around the syntax fault.
- Cleaned-tail trimming check: when syntax errors persist, compare `code_stats.raw_lines` vs `code_stats.cleaned_lines` in errors to confirm tail trimming ran; if `cleaned_lines` is much smaller and errors remain, the truncation likely occurs earlier than the last ~40 lines.
- Missing-comma repair + deep tail trim: `clean_code_string` now attempts to auto-insert a missing comma (when SyntaxError hints at it) and trim deeper tails (beyond 40 lines) when truncation is detected. Expect `llm_syntax_error_mechanism_count` and `mechanism_error_count` to drop when errors were caused by missing commas or long truncated tails.
- LLM finish_reason counts: `round_metrics.llm_finish_reason_counts`, `round_metrics.llm_finish_reason_total`, `round_metrics.llm_finish_reason_length_count`, and `round_metrics.llm_finish_reason_missing_count` to quantify truncation signals across agent + mechanism LLM calls.
- LLM request limits: inspect `generated_code[*].llm_metadata.request_max_tokens` / `request_max_completion_tokens` alongside `completion_tokens` to confirm whether truncation aligns with requested caps.
- LLM cap diagnostics (derived): `round_metrics.llm_request_cap_avg`/`min`/`max`, `llm_request_cap_count`, `llm_completion_at_request_cap_count`, `llm_completion_at_request_cap_rate`, plus `llm_prompt_tokens_avg` and `llm_completion_tokens_avg` to spot prompt pressure and cap saturation in one place.
- Prompt size diagnostics (per attempt): `llm_metadata.prompt_char_count`, `llm_metadata.prompt_section_chars`, `llm_metadata.prompt_dynamic_char_total`, `llm_metadata.prompt_dynamic_char_ratio` to identify which prompt sections dominate when truncation occurs (target those sections for trimming before changing behavior). The base-class section now includes per-class breakdowns (e.g., `base_code_base_island`, `base_code_base_land`, `base_code_base_member`) so the largest code block can be isolated before pruning. Round metrics now aggregate these into `llm_prompt_char_count_avg`/`min`/`max`, `llm_prompt_section_chars_total`/`avg`/`max`, `llm_prompt_section_entry_count`, plus `llm_prompt_section_top_avg_*` and `llm_prompt_section_top_base_code_*` for quick inspection in summaries.
- Prompt dominance ratios (derived): `llm_prompt_section_base_code_ratio` (base_code share of average prompt), `llm_prompt_section_top_avg_ratio` (largest section share), and `llm_prompt_section_top_base_code_ratio` (largest base-code subsection share) to quantify where trimming will yield the biggest prompt savings.
- Empty cleaned outputs: `llm_empty_response` in `agent_code_error_type_counts` or `mechanism_error_type_counts` indicates the LLM returned non-code or code stripped to empty; treat as invalid for behavior evaluation and inspect prompt compliance.
- LLM availability: `llm_error_total` and `llm_error_tag_counts` (unique per tag across raw+derived metrics; should be zero for valid e2e results); `llm_preflight.ok` should be true when enabled.
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
- Mechanism judge feedback: `round_metrics.mechanism_judge_approved_count`, `round_metrics.mechanism_judge_rejected_count`, `round_metrics.mechanism_judge_missing_count`, and `round_metrics.mechanism_judge_rejection_reason_counts` to see why proposals are blocked and whether judge results were recorded.
- Legacy histories without `round_metrics`: `scripts/inspect_execution_history.py` derives survival deltas (avg/std), agent code error count/rate/tag counts, plan action totals, mechanism/contract/physics counts (when present), and population signature diversity stats from `agent_actions`/`actions` + `errors` and action code as approximate fallbacks.

Baselines
- Use the last 3 completed rounds from `execution_history` as baseline (or first run if none).
- Latest e2e smoke (2026-01-30T13:46:42, run `20260130_134037`) is LLM-healthy (`llm_error_total=0`) with `mechanism_judge_approved_count=3`, `mechanism_executed_count=1`, and `agent_code_error_rate=0.0`. End-of-round deltas remain negative but improved (`round_end_population_avg_survival_delta=-10.36`, `round_end_population_avg_cargo_delta=-32.96`). Diversity remains high (`population_signature_unique_ratio=1.0`, `population_signature_dominant_share=0.33`), plan alignment is 1.0, and `plan_ineligible_tag_rate=0.0`. `llm_finish_reason_missing_count=3` persists alongside `llm_request_cap_count=2`; treat truncation diagnostics as provisional and rerun after behavior changes.
- Prior e2e smoke (2026-01-30T13:28:54, run `20260130_131311`) is LLM-healthy (`llm_error_total=0`) with `mechanism_judge_approved_count=3`, `mechanism_executed_count=1`, and `agent_code_error_rate=0.333`. End-of-round deltas are sharply negative (`round_end_population_avg_survival_delta=-14.67`, `round_end_population_avg_cargo_delta=-40.78`). Diversity remains high (`population_signature_unique_ratio=1.0`, `population_signature_dominant_share=0.33`), plan alignment is 1.0, and `plan_ineligible_tag_rate=0.0909`. `llm_finish_reason_missing_count=3` persists alongside `llm_request_cap_count=2`; treat truncation diagnostics as provisional and rerun after behavior changes.
- Latest e2e smoke (2026-01-30T12:38:53, run `20260130_123421`) is LLM-healthy (`llm_error_total=0`) with `mechanism_judge_approved_count=3`, `mechanism_executed_count=1`, and `mechanism_error_count=0`. Analysis card tag hygiene is clean (`analysis_card_signature_invalid_tag_rate=0.0`, `analysis_card_signature_recoverable_tag_rate=0.0`, `analysis_card_signature_empty_card_rate=0.0`). `llm_finish_reason_missing_count=3` persists alongside `llm_request_cap_count=2`; treat truncation diagnostics as provisional and rerun after behavior changes.
- Latest e2e smoke (2026-01-30T11:13:00, run `20260130_110112`) is LLM-healthy (`llm_error_total=0`) but all 3 mechanism attempts were rejected (`mechanism_judge_rejected_count=3`, `mechanism_judge_approved_count=0`). Rejection reasons cite: (1) member-specific unfair advantage, (2) unilateral resource transfer violating autonomy/consent, and (3) currency creation / missing balance checks. Use this as the evidence to tighten mechanism guardrails and templates before scaling.
- Latest e2e smoke (2026-01-30T11:29:32, run `20260130_112116`) is LLM-healthy (`llm_error_total=0`) with `mechanism_judge_approved_count=1`, `mechanism_judge_rejected_count=2`, and `mechanism_executed_count=0` plus `mechanism_error_count=1` (`mechanism_execution`). Rejections cite conservation violations (resource creation + unconstrained credit minting) and an undefined variable bug (`share_id`) in a contract exit path. Use this to reinforce conservation and self-check guardrails and surface judge feedback in prompts.
- Latest e2e smoke (2026-01-30T11:57:44, run `20260130_114524`) is LLM-healthy (`llm_error_total=0`) with `mechanism_judge_approved_count=3`, `mechanism_executed_count=1`, and `mechanism_error_count=0`. Analysis card tag hygiene was noisy at this point (`analysis_card_signature_invalid_tag_rate=0.37`, `analysis_card_signature_recoverable_tag_rate=0.52`), and `llm_finish_reason_missing_count=3` persists alongside `llm_request_cap_count=2` (prompt tokens avg ~21.9k). Use this as the pre-fix baseline for tag hygiene comparisons.
- Latest baseline snapshot (2026-01-30T03:23:35 e2e smoke, run `20260130_024153`, 1 round):
  - `plan_feasibility_missing_reason_counts`: missing_experiment=0, missing_label=1, missing_signature=0.
  - Analysis card tag hygiene: card_count=3, tag_total=14, invalid_tag_rate=0.286, recoverable_tag_rate=0.0, empty_card_rate=0.333.
  - LLM health: `llm_error_total=0`, `llm_syntax_error_count=0`.
- Note: analysis-card hygiene metrics were updated after 2026-01-30; recompute the baseline on the next valid e2e run.
- Latest e2e smoke (2026-01-30T09:55:07, run `20260130_094006`) is LLM-healthy (`llm_error_total=0`) but reports `mechanism_error_count=1` (`mechanism_execution` = `NameError: np not defined`) with `mechanism_executed_count=0`. Rerun after exposing numpy in mechanism execution env; expect `mechanism_error_count=0` and `mechanism_executed_count` to rise with approved mechanisms.
- Latest e2e smoke (2026-01-30T10:42:29, run `20260130_101110`) is LLM-healthy (`llm_error_total=0`) with `mechanism_executed_count=1`, `mechanism_error_count=0`, and `plan_only_tag_rate=0.6`. Prompt size spiked (`llm_prompt_tokens_avg=31090`, `llm_prompt_char_count_avg=114391`) because `current_mechanisms` ballooned (~98k chars from a 49k aggregated mechanism code); `llm_finish_reason_length_count=1` and `llm_finish_reason_missing_count=3` persist. Trim mechanism prompt payloads before the next rerun to reduce truncation pressure.
- Latest e2e smoke (2026-01-30T08:28:19, run `20260130_081430`) is invalid per guardrails (`llm_error_total=1` from `llm_syntax_error=1`). Mechanism errors include `llm_syntax_error` and `mechanism_execution` with `mechanism_executed_count=0`; agent code errors were 0. It also reports `llm_finish_reason_missing_count=3` and `llm_request_cap_count=2` (max_tokens=4096). Inspect syntax-error line context before changing prompts, then rerun e2e.
- Earlier e2e smoke (2026-01-30T08:01:14, run `20260130_074115`) is LLM-healthy (`llm_error_total=0`) and shows `mechanism_judge_missing_count=0`, but `llm_finish_reason_length_count=1` and `llm_completion_at_request_cap_rate=0.5`; treat behavior metrics as provisional and prioritize prompt-length/conciseness fixes before promoting as baseline. It recorded `agent_code_error_count=1` (tag `contracts`) and one mechanism rejection citing conservation and completeness issues.
- Prior e2e smoke (2026-01-30T07:32:39, run `20260130_062642`) is invalid per guardrails (`llm_error_total=1`) and shows `mechanism_judge_missing_count=3`; treat this as a wiring issue to verify after the next rerun.
- Previous e2e smoke summary (2026-01-30T01:18:18) reported `llm_error_total=3` (`llm_syntax_error=2`, `llm_empty_response=1`), so it remains invalid per the LLM-error guardrail and should not replace the baseline.

Thresholds (guardrails)
- Survival (end-of-round): no worse than baseline by more than 0.02 on `round_end_population_avg_survival_delta`.
- Diversity: `population_signature_dominant_share` must not exceed baseline by >0.05.
- Dominant signature alignment rate should not exceed baseline by >0.05.
- Diversity: `population_signature_unique_ratio` and `population_signature_entropy` should not drop by >0.05.
- Environment: `round_context.gini_wealth` should not drop by >0.05 (avoid homogenizing resources).
- Contract concentration: `contract_partner_top_share_avg` should not exceed baseline by >0.10.
- Mechanism execution must not exceed approved count in any round; mechanism error rate should not exceed baseline by >0.05.
- Mechanism judge coverage: `mechanism_judge_missing_count` should be 0 when the judge node is enabled; non-zero indicates missing judge logging or skipped judging.
- Mechanism judge safety: rejection reasons mentioning unfair advantage, unilateral transfers/autonomy violations, or currency creation should be 0 in a healthy run; if present, refine guardrails/templates before scaling.
- Experiments: at least 40% of recent actions should match a baseline or variation plan.
- Execution reliability: plan/execution alignment rate should not drop by >0.05 vs baseline.
- Feasibility: `plan_ineligible_tag_rate` and `plan_only_tag_rate` should not exceed baseline by >0.05.
- Agent code errors: `agent_code_error_rate` should not exceed baseline by >0.05; scan `agent_code_error_tag_counts` for recurring high-error tags. If infra tags dominate, rerun with a working provider before judging behavior.
- LLM infra errors (DNS/auth/rate-limit) invalidate the run; rerun with a working provider before judging behavior.
- Minor LLM output errors (syntax/empty/truncation) do not block iteration, but require a follow-up rerun to confirm improvements.
- LLM truncation: `llm_finish_reason_length_count` should be 0; if >0, treat behavior metrics as invalid and rerun with higher max tokens or trimmed prompts.
- LLM syntax errors: with tail-trimming enabled, expect `llm_error_tag_counts.llm_syntax_error` to drop to 0 in a healthy run; if not, re-examine prompt size and truncation causes.
- Syntax error position: use `round_metrics.llm_syntax_error_near_end_count`, `round_metrics.llm_syntax_error_near_end_10pct_count`, `round_metrics.llm_syntax_error_mid_count`, and the `*_rate` variants to see whether syntax faults cluster in the last few lines or last 10% of lines (truncation) versus mid-body (model quality); if mid-body dominates, prompt trimming alone will not fix it.
- Syntax error line ratio diagnostics: `llm_syntax_error_line_ratio_avg`/`min`/`max` and `llm_syntax_error_line_ratio_samples` to see how early the faults occur; treat ratios as meaningful only when samples > 0.
- Non-code or JS-comment syntax errors (`llm_syntax_error_non_code_prefix_count`, `llm_syntax_error_js_comment_count`) indicate prompt compliance gaps; fix prompts or add cleaning before judging behavior.
- Leading-narrative trimming now strips non-code prefixes before parsing; expect `llm_syntax_error_non_code_prefix_count` (especially mechanism) to drop to 0 on the next valid e2e run.
- Syntax error source splits: use `llm_syntax_error_agent_*` vs `llm_syntax_error_mechanism_*` (count/near_end/non_code_prefix/js_comment/missing_in) to confirm whether prompt compliance or truncation issues are coming from agent vs mechanism LLM calls before changing behavior.
- Syntax error root-cause hints: `llm_syntax_error_js_comment_count` (JS-style `//` comments) and `llm_syntax_error_non_code_prefix_count` (non-code preamble on line 1) to separate prompt compliance issues from truncation.
- Use `python scripts/inspect_execution_history.py --show-syntax-examples --syntax-example-limit 3` to print sample syntax-error lines (e.g., non-code prefixes) so prompt/cleaning fixes can be targeted before rerunning Stage 3.
- Use `python scripts/inspect_execution_history.py --show-analysis-tag-examples --analysis-tag-example-limit 3` to print invalid/recoverable analysis card signature tags so prompt fixes can target formatting drift (arguments, extra punctuation).
- Missing-`in` syntax errors (`llm_syntax_error_missing_in_count`) flag keyword omissions like `if x not y:`; treat as model grammar issues rather than truncation.
- Syntax error truncation linkage: `llm_syntax_error_finish_reason_counts`, `llm_syntax_error_finish_reason_length_count`, `llm_syntax_error_request_cap_count`, and `llm_syntax_error_completion_at_request_cap_rate` to confirm whether syntax errors coincide with length-capped outputs.
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
- Feasibility coverage: `plan_feasibility_missing_reason_counts.missing_label` trends toward 0 as signature parsing recovers action tags.
- Contract diversity: `contract_partner_unique_avg` +1 without increasing failures.
- Recommendation diversity: when population dominance is high, expect some diversity adjustments in recommendations without reducing survival guardrails.
- Mechanism approvals: `mechanism_judge_approved_count` >= 1 with no judge rejections citing unfair advantage, autonomy/consent violations, or currency creation vulnerabilities.
- Resource pressure: when net production is negative or recent cargo drag is high, expect lower exploration pressure and improved `round_end_population_avg_cargo_delta` (less negative vs baseline).

Stage-gated evaluation
- Small (e2e smoke): 1–3 rounds, 3–5 agents. Use `scripts/run_e2e_smoke.py` and validate guardrails.
- Medium (e2e): 5 rounds, 5–8 agents. Expect diversity metrics to improve without survival drop.
- Large: only if medium shows positive deltas.
