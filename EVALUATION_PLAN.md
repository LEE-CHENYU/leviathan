# Evaluation Plan: MetaIsland Strategy Diversity

Goal
- Improve self-improving strategy design while preserving agent/environment diversity.

Metrics (tracked per round)
- Population survival (end-of-round): `round_end_population_avg_survival_delta`, `round_end_population_std_survival_delta`.
- Action-phase survival (pre-contract/produce/consume): `population_avg_survival_delta`, `population_std_survival_delta`.
- Diversity health: `population_signature_unique_ratio`, `population_signature_entropy`, `population_signature_dominant_share`.
- Dominant signature alignment: rate of actions matching the population-dominant signature.
- Environment diversity: `round_context.gini_wealth`, `round_context.gini_cargo`, `round_context.gini_land`.
- Agent outcomes: per-agent `delta_survival`, `delta_vitality`, `delta_cargo`.
- Experiment outcomes: baseline vs variation avg `delta_survival` (from experiment summaries).
- Execution reliability: plan/execution alignment rate via `round_metrics.plan_alignment_rate` (planned signature equals executed signature) and plan coverage via `round_metrics.plan_alignment_plan_coverage`.
- Feasibility follow-through: share of planned actions with ineligible tags, plus planned-only rate.
- Contextual fit: average balanced score for entries with context similarity >= 0.35 vs overall average.
- Identity continuity: `memory_active_coverage`, `memory_missing_count`, and `memory_orphan_count` from round metrics (active ids vs code_memory keys).
- Contract activity: `contract_stats.pending`, `contract_stats.active`, `contract_stats.completed`, `contract_stats.failed`.
- Contract diversity: `contract_partner_unique_avg` and `contract_partner_top_share_avg` (round metrics), plus per-agent partner counts in `contract_partner_counts` and `contract_partner_top_share`.
- Physics coverage: `physics_stats.active_constraints`, `physics_stats.domains`.
- Recommendation diversity adjustments: count of strategy recommendations containing "Diversity adjustment" when `population_signature_dominant_share >= 0.60` (manual log scan).
- Mechanism gating: `round_metrics.mechanism_attempted_count`, `round_metrics.mechanism_approved_count`, `round_metrics.mechanism_executed_count`, `round_metrics.mechanism_error_count` (cross-check with `mechanism_modifications` and `errors.mechanism_errors`).

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
- Feasibility: planned-infeasible tag rate should not exceed baseline by >0.05.
- Identity: `memory_active_coverage` should not drop by >0.05 vs baseline; `memory_missing_count` should be 0 after action rounds; track `memory_orphan_count` (expected to rise with deaths).

Expected deltas (small wins)
- `population_signature_unique_ratio` +0.05.
- `population_signature_dominant_share` -0.05.
- Dominant signature alignment rate -0.05.
- `round_context.gini_wealth` stable or +0.02 (preserve heterogeneity).
- Experiment match rate >= 60%.
- Contextual fit: context-matched entries show >= +0.02 balanced score vs overall.
- Feasibility: planned-infeasible tag rate -0.05.
- Contract diversity: `contract_partner_unique_avg` +1 without increasing failures.
- Recommendation diversity: when population dominance is high, expect some diversity adjustments in recommendations without reducing survival guardrails.

Stage-gated evaluation
- Small: 3 rounds, 3–5 agents. Verify guardrails and that experiment summaries populate.
- Medium: 5 rounds, 5–8 agents. Expect diversity metrics to improve without survival drop.
- Large: only if medium shows positive deltas.
