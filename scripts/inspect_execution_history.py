#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_metrics import (
    combine_round_metrics,
    EXPECTED_ROUND_METRICS,
    fallback_metrics,
    safe_float,
)

METRIC_KEYS = [
    "round_end_population_avg_survival_delta",
    "population_avg_survival_delta",
    "population_std_survival_delta",
    "population_signature_unique_ratio",
    "population_signature_entropy",
    "population_signature_dominant_share",
    "plan_alignment_rate",
    "plan_alignment_plan_coverage",
    "plan_alignment_avg_match_score",
    "plan_ineligible_tag_rate",
    "plan_only_tag_rate",
    "agent_code_error_rate",
    "agent_code_error_count",
    "llm_finish_reason_total",
    "llm_finish_reason_length_count",
    "llm_finish_reason_missing_count",
    "llm_request_cap_count",
    "llm_request_cap_avg",
    "llm_request_cap_min",
    "llm_request_cap_max",
    "llm_completion_at_request_cap_count",
    "llm_completion_at_request_cap_rate",
    "llm_prompt_tokens_avg",
    "llm_completion_tokens_avg",
    "memory_active_coverage",
    "memory_missing_count",
    "memory_orphan_count",
    "mechanism_attempted_count",
    "mechanism_approved_count",
    "mechanism_executed_count",
    "mechanism_error_count",
    "contract_total",
    "contract_pending",
    "contract_active",
    "contract_completed",
    "contract_failed",
    "contract_partner_unique_avg",
    "contract_partner_top_share_avg",
    "physics_active_constraints",
    "physics_domain_count",
]

DOMINANCE_THRESHOLD = 0.60

GUARDRAILS = {
    "round_end_population_avg_survival_delta": ("min", -0.02),
    "population_signature_dominant_share": ("max", 0.05),
    "population_signature_unique_ratio": ("min", -0.05),
    "population_signature_entropy": ("min", -0.05),
    "plan_alignment_rate": ("min", -0.05),
    "plan_ineligible_tag_rate": ("max", 0.05),
    "plan_only_tag_rate": ("max", 0.05),
    "agent_code_error_rate": ("max", 0.05),
    "memory_active_coverage": ("min", -0.05),
    "contract_partner_top_share_avg": ("max", 0.10),
    "mechanism_error_rate": ("max", 0.05),
}


def _collect_metric(
    round_record: dict,
    key: str,
    fallback_metrics: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[float], bool]:
    round_metrics = round_record.get("round_metrics") or {}
    if key in round_metrics:
        return safe_float(round_metrics.get(key)), False
    round_end_metrics = round_record.get("round_end_metrics") or {}
    if key in round_end_metrics:
        return safe_float(round_end_metrics.get(key)), False
    if fallback_metrics and key in fallback_metrics:
        return safe_float(fallback_metrics.get(key)), True
    return None, False


def _extract_prompts(round_record: dict) -> List[str]:
    prompts: List[str] = []
    generated = round_record.get("generated_code") or {}
    if isinstance(generated, dict):
        for entry in generated.values():
            if not isinstance(entry, dict):
                continue
            prompt = entry.get("final_prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt)

    mech_attempts = round_record.get("mechanism_modifications", {}).get("attempts") or []
    if isinstance(mech_attempts, list):
        for attempt in mech_attempts:
            if not isinstance(attempt, dict):
                continue
            prompt = attempt.get("final_prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt)

    return prompts


def _count_diversity_adjustments(prompts: Iterable[str]) -> int:
    count = 0
    for prompt in prompts:
        if "diversity adjustment" in prompt.lower():
            count += 1
    return count


def _mechanism_error_rate(
    round_record: dict,
    fallback_metrics: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    attempted, _ = _collect_metric(round_record, "mechanism_attempted_count", fallback_metrics)
    errors, _ = _collect_metric(round_record, "mechanism_error_count", fallback_metrics)
    if attempted is None or attempted <= 0 or errors is None:
        return None
    return errors / attempted


def _latest_history_file(path: Path) -> Optional[Path]:
    files = sorted(path.glob("execution_history_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    return files[-1]


def _history_files(path: Path, all_files: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    if all_files:
        return sorted(path.glob("execution_history_*.json"), key=lambda p: p.stat().st_mtime)
    latest = _latest_history_file(path)
    return [latest] if latest else []


def _baseline_rounds(rounds: List[dict], window: int) -> Tuple[List[dict], Optional[dict]]:
    if not rounds:
        return [], None
    if len(rounds) == 1:
        return [], rounds[0]
    window = max(1, window)
    if len(rounds) > window + 1:
        baseline = rounds[-(window + 1):-1]
    else:
        baseline = rounds[:-1]
    return baseline, rounds[-1]


def _avg(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _summarize_file(path: Path, round_limit: Optional[int], baseline_window: int, show_rounds: bool) -> None:
    data = json.loads(path.read_text())
    rounds = data.get("rounds") or []
    if round_limit:
        rounds = rounds[-round_limit:]
    total_rounds = len(rounds)

    print(f"\n=== {path.name} ===")
    print(f"Rounds considered: {total_rounds}")

    prompt_total = 0
    prompt_diversity = 0
    prompt_total_dominant = 0
    prompt_diversity_dominant = 0
    dominant_rounds = 0
    metric_values: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}
    missing_metric_counts = {key: 0 for key in METRIC_KEYS}
    derived_metric_counts = {key: 0 for key in METRIC_KEYS}
    derived_metric_sources: Dict[str, set] = {key: set() for key in METRIC_KEYS}
    expected_missing_counts = {key: 0 for key in EXPECTED_ROUND_METRICS}
    coverage_values: List[float] = []
    legacy_rounds = 0
    mechanism_overruns: List[Tuple[object, float, float]] = []

    for idx, round_record in enumerate(rounds, start=1):
        combined_metrics, derived_metrics, derived_sources, missing_metrics, coverage = (
            combine_round_metrics(round_record, EXPECTED_ROUND_METRICS)
        )
        prompts = _extract_prompts(round_record)
        diversity_hits = _count_diversity_adjustments(prompts)
        prompt_total += len(prompts)
        prompt_diversity += diversity_hits
        dominant_share = safe_float(combined_metrics.get("population_signature_dominant_share"))
        if dominant_share is not None and dominant_share >= DOMINANCE_THRESHOLD:
            dominant_rounds += 1
            prompt_total_dominant += len(prompts)
            prompt_diversity_dominant += diversity_hits

        mech_executed = safe_float(combined_metrics.get("mechanism_executed_count"))
        mech_approved = safe_float(combined_metrics.get("mechanism_approved_count"))
        if mech_executed is not None and mech_approved is not None and mech_executed > mech_approved:
            round_id = round_record.get("round_number", idx)
            mechanism_overruns.append((round_id, mech_executed, mech_approved))

        if show_rounds:
            print(f"  Round {round_record.get('round_number', idx)}: prompts={len(prompts)}, diversity_adjustments={diversity_hits}")

        if not round_record.get("round_metrics"):
            legacy_rounds += 1
        if coverage is not None:
            coverage_values.append(coverage)
        for key in missing_metrics:
            expected_missing_counts[key] += 1

        for key in METRIC_KEYS:
            value, used_fallback = _collect_metric(round_record, key, derived_metrics)
            if value is None:
                missing_metric_counts[key] += 1
                continue
            metric_values[key].append(value)
            if used_fallback:
                derived_metric_counts[key] += 1
                source = derived_sources.get(key)
                if source:
                    derived_metric_sources[key].add(source)

    if prompt_total:
        rate = prompt_diversity / prompt_total
    else:
        rate = None
    print(f"Prompts scanned: {prompt_total} | Diversity adjustment: {prompt_diversity} ({_format_value(rate)})")
    if prompt_total_dominant:
        dominant_rate = prompt_diversity_dominant / prompt_total_dominant
    else:
        dominant_rate = None
    if dominant_rounds:
        print(
            f"Dominant-share prompts (>= {DOMINANCE_THRESHOLD:.2f}): {prompt_total_dominant} "
            f"| Diversity adjustment: {prompt_diversity_dominant} ({_format_value(dominant_rate)}) "
            f"across {dominant_rounds} rounds"
        )

    if coverage_values:
        coverage_avg = _avg(coverage_values)
        print(
            f"Round metrics coverage (avg over rounds): {_format_value(coverage_avg)} "
            f"of {len(EXPECTED_ROUND_METRICS)} expected fields"
        )
    if legacy_rounds:
        print(f"Legacy rounds without round_metrics: {legacy_rounds}/{total_rounds}")
    expected_missing_all = [
        key for key, count in expected_missing_counts.items()
        if total_rounds and count == total_rounds
    ]
    if expected_missing_all:
        print(
            f"Expected round metrics missing in all rounds: "
            f"{len(expected_missing_all)}/{len(EXPECTED_ROUND_METRICS)}"
        )

    missing = [key for key, count in missing_metric_counts.items() if count == total_rounds and total_rounds]
    if missing:
        print("Missing metrics (no data in considered rounds):")
        for key in missing:
            print(f"  - {key}")

    derived = [key for key, count in derived_metric_counts.items() if count]
    if derived:
        print("Derived metrics used (fallbacks applied):")
        for key in derived:
            sources = ", ".join(sorted(derived_metric_sources[key])) or "fallback"
            print(f"  - {key}: {derived_metric_counts[key]}/{total_rounds} rounds via {sources}")

    if mechanism_overruns:
        print("Mechanism execution exceeded approved count:")
        for round_id, executed, approved in mechanism_overruns:
            print(f"  - Round {round_id}: executed={executed:.0f} approved={approved:.0f}")

    print("Metric averages:")
    for key in METRIC_KEYS:
        avg_val = _avg(metric_values[key])
        if avg_val is None:
            continue
        print(f"  {key}: {_format_value(avg_val)}")

    baseline_rounds, last_round = _baseline_rounds(rounds, baseline_window)
    if not baseline_rounds or last_round is None:
        print("Guardrails: skipped (need at least 2 rounds)")
        return

    print(f"Guardrails: baseline rounds={len(baseline_rounds)} window={baseline_window}")
    for key, (direction, threshold) in GUARDRAILS.items():
        baseline_values = []
        for round_record in baseline_rounds:
            derived_metrics, _ = fallback_metrics(round_record)
            if key == "mechanism_error_rate":
                value = _mechanism_error_rate(round_record, derived_metrics)
            else:
                value, _ = _collect_metric(round_record, key, derived_metrics)
            if value is not None:
                baseline_values.append(value)
        last_fallback, _ = fallback_metrics(last_round)
        if key == "mechanism_error_rate":
            last_value = _mechanism_error_rate(last_round, last_fallback)
        else:
            last_value, _ = _collect_metric(last_round, key, last_fallback)
        baseline_avg = _avg(baseline_values)
        if baseline_avg is None or last_value is None:
            continue
        delta = last_value - baseline_avg
        status = "OK"
        if direction == "min" and delta < threshold:
            status = "WARN"
        elif direction == "max" and delta > threshold:
            status = "WARN"
        print(
            f"  [{status}] {key}: baseline {_format_value(baseline_avg)} -> "
            f"last {_format_value(last_value)} (delta {_format_value(delta)}, limit {threshold:+.3f})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect execution history metrics and diversity adjustments.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("execution_histories"),
        help="Execution history file or directory (default: execution_histories)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all execution_history_*.json files in the directory",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Limit to the last N rounds per file",
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=3,
        help="Baseline window size for guardrail checks",
    )
    parser.add_argument(
        "--show-rounds",
        action="store_true",
        help="Show per-round prompt counts",
    )
    args = parser.parse_args()

    files = _history_files(args.path, args.all)
    if not files:
        print(f"No execution history files found at {args.path}")
        return 1

    for file_path in files:
        _summarize_file(file_path, args.rounds, args.baseline_window, args.show_rounds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
