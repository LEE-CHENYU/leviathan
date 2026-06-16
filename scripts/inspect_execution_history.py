#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import eval_metrics
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
    "llm_prompt_char_count_avg",
    "llm_prompt_char_count_min",
    "llm_prompt_char_count_max",
    "memory_active_coverage",
    "memory_missing_count",
    "memory_orphan_count",
    "mechanism_attempted_count",
    "mechanism_approved_count",
    "mechanism_executed_count",
    "mechanism_error_count",
    "mechanism_judge_approved_count",
    "mechanism_judge_rejected_count",
    "mechanism_judge_missing_count",
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
PROMPT_SECTION_TOP_N = 6
PROMPT_SECTION_BASE_TOP_N = 4
ANALYSIS_TAG_TOP_N = 5

_ANALYSIS_TAG_TOKEN_RE = re.compile(r"[^a-z0-9_]+")
_ANALYSIS_TAG_ARG_RE = re.compile(r"\b[a-z_]+\([^)]*\)")

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


def _iter_error_entries(round_record: dict) -> Iterable[Tuple[str, dict]]:
    errors = round_record.get("errors") or {}
    if not isinstance(errors, dict):
        return []
    for key in ("agent_code_errors", "mechanism_errors", "analyze_code_errors"):
        entries = errors.get(key) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                yield key, entry


def _iter_analysis_card_tags(round_record: dict) -> Iterable[Tuple[object, str, str]]:
    cards = round_record.get("analysis_cards")
    if not isinstance(cards, dict) or not cards:
        return
    for card_id, card in cards.items():
        if not isinstance(card, dict):
            continue
        for key in ("baseline_signature", "variation_signature"):
            for raw in eval_metrics._coerce_card_tags(card.get(key)):
                if raw:
                    yield card_id, key, raw


def _is_syntax_error_entry(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("error_category") == "llm_syntax_error":
        return True
    details = entry.get("error_details")
    if isinstance(details, dict) and details.get("error_type") == "SyntaxError":
        return True
    msg = ""
    if isinstance(details, dict):
        msg = str(details.get("error_msg") or "")
    else:
        msg = str(entry.get("error") or "")
    msg = msg.lower()
    return any(token in msg for token in ("syntax", "unterminated", "unexpected eof", "eof while scanning"))


def _syntax_error_category(entry: dict) -> str:
    details = entry.get("error_details") if isinstance(entry, dict) else None
    error_text = None
    error_line = None
    if isinstance(details, dict):
        error_text = details.get("error_text")
        error_line = details.get("error_line")
    if eval_metrics._is_js_comment_line(error_text):
        return "js_comment"
    if eval_metrics._is_missing_in_operator_line(error_text):
        return "missing_in"
    if error_line == 1 and eval_metrics._is_non_code_prefix_line(error_text):
        return "non_code_prefix"
    if error_text:
        return "other"
    return "unknown"


def _analysis_tag_status(raw: object) -> Tuple[str, Optional[str], Optional[str]]:
    strict = eval_metrics._normalize_action_tag(raw, strip_punct=False)
    if strict:
        return "valid", strict, strict
    lenient = eval_metrics._normalize_action_tag(raw, strip_punct=True)
    if lenient:
        return "recoverable", None, lenient
    return "invalid", None, None


def _suggest_action_tag(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    for token in _ANALYSIS_TAG_TOKEN_RE.split(text):
        if not token:
            continue
        candidate = eval_metrics._normalize_action_tag(token, strip_punct=True)
        if candidate:
            return candidate
    text_norm = _ANALYSIS_TAG_TOKEN_RE.sub("_", text).strip("_")
    for tag in eval_metrics.SIGNATURE_TAG_ORDER:
        pattern = rf"(?:^|_){re.escape(tag)}(?:_|$)"
        if re.search(pattern, text_norm):
            return tag
    return None


def _trim_text(text: Optional[str], limit: int = 160) -> str:
    if not text:
        return ""
    cleaned = " ".join(str(text).strip().split())
    if limit and len(cleaned) > limit:
        return cleaned[: max(0, limit - 3)] + "..."
    return cleaned


def _format_syntax_example(round_id: object, source: str, entry: dict) -> str:
    member_id = entry.get("member_id")
    if member_id is None:
        member_id = entry.get("member_index")
    details = entry.get("error_details") if isinstance(entry, dict) else None
    text = None
    if isinstance(details, dict):
        text = details.get("error_text") or details.get("error_msg")
    if not text:
        text = entry.get("error")
    text = _trim_text(text)
    member_text = f" member={member_id}" if member_id is not None else ""
    return f"round={round_id} source={source}{member_text} text={text}"


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


def _summarize_file(
    path: Path,
    round_limit: Optional[int],
    baseline_window: int,
    show_rounds: bool,
    show_syntax_examples: bool,
    syntax_example_limit: int,
    show_analysis_tag_examples: bool,
    analysis_tag_example_limit: int,
) -> None:
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
    prompt_section_totals: Dict[str, float] = {}
    prompt_section_counts: Dict[str, int] = {}
    prompt_section_max: Dict[str, float] = {}
    prompt_top_key_counts: Dict[str, int] = {}
    prompt_top_base_key_counts: Dict[str, int] = {}
    syntax_counts: Counter = Counter()
    syntax_examples: Dict[str, List[str]] = defaultdict(list)
    analysis_tag_counts: Counter = Counter()
    analysis_invalid_counts: Counter = Counter()
    analysis_invalid_pattern_counts: Counter = Counter()
    analysis_invalid_suggestions: Counter = Counter()
    analysis_invalid_examples: List[str] = []
    analysis_recoverable_examples: List[str] = []

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

        section_avgs = combined_metrics.get("llm_prompt_section_chars_avg")
        if isinstance(section_avgs, dict):
            for section_key, section_val in section_avgs.items():
                val = safe_float(section_val)
                if val is None:
                    continue
                prompt_section_totals[section_key] = prompt_section_totals.get(section_key, 0.0) + val
                prompt_section_counts[section_key] = prompt_section_counts.get(section_key, 0) + 1
                current_max = prompt_section_max.get(section_key)
                if current_max is None or val > current_max:
                    prompt_section_max[section_key] = val

        top_key = combined_metrics.get("llm_prompt_section_top_avg_key")
        if isinstance(top_key, str) and top_key:
            prompt_top_key_counts[top_key] = prompt_top_key_counts.get(top_key, 0) + 1
        top_base_key = combined_metrics.get("llm_prompt_section_top_base_code_key")
        if isinstance(top_base_key, str) and top_base_key:
            prompt_top_base_key_counts[top_base_key] = prompt_top_base_key_counts.get(top_base_key, 0) + 1

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

        if show_syntax_examples:
            round_id = round_record.get("round_number", idx)
            for source, entry in _iter_error_entries(round_record):
                if not _is_syntax_error_entry(entry):
                    continue
                category = _syntax_error_category(entry)
                syntax_counts[category] += 1
                if syntax_example_limit > 0 and len(syntax_examples[category]) < syntax_example_limit:
                    syntax_examples[category].append(_format_syntax_example(round_id, source, entry))

        if show_analysis_tag_examples:
            round_id = round_record.get("round_number", idx)
            for card_id, field, raw in _iter_analysis_card_tags(round_record):
                category, _, lenient = _analysis_tag_status(raw)
                analysis_tag_counts[category] += 1
                raw_text = _trim_text(raw)
                if category == "invalid":
                    raw_str = str(raw).strip()
                    analysis_invalid_counts[raw_str] += 1
                    if _ANALYSIS_TAG_ARG_RE.search(raw_str):
                        analysis_invalid_pattern_counts["arguments"] += 1
                    if "+" in raw_str:
                        analysis_invalid_pattern_counts["compound_plus"] += 1
                    if " " in raw_str:
                        analysis_invalid_pattern_counts["whitespace"] += 1
                    if "," in raw_str:
                        analysis_invalid_pattern_counts["comma"] += 1
                    suggestion = _suggest_action_tag(raw)
                    if suggestion:
                        analysis_invalid_suggestions[suggestion] += 1
                    if analysis_tag_example_limit > 0 and len(analysis_invalid_examples) < analysis_tag_example_limit:
                        example = f"round={round_id} card={card_id} field={field} raw={raw_text}"
                        if suggestion:
                            example += f" suggested={suggestion}"
                        analysis_invalid_examples.append(example)
                elif category == "recoverable":
                    if analysis_tag_example_limit > 0 and len(analysis_recoverable_examples) < analysis_tag_example_limit:
                        example = f"round={round_id} card={card_id} field={field} raw={raw_text}"
                        if lenient:
                            example += f" normalized={lenient}"
                        analysis_recoverable_examples.append(example)

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

    prompt_char_avg = _avg(metric_values.get("llm_prompt_char_count_avg", []))
    if prompt_section_totals:
        section_avgs = {
            key: prompt_section_totals[key] / prompt_section_counts[key]
            for key in prompt_section_totals
            if prompt_section_counts.get(key)
        }
        top_sections = sorted(section_avgs.items(), key=lambda item: item[1], reverse=True)[:PROMPT_SECTION_TOP_N]
        if top_sections:
            print("Prompt section averages (chars per prompt):")
            for section_key, section_val in top_sections:
                if prompt_char_avg:
                    ratio = section_val / prompt_char_avg
                    ratio_text = f"{ratio:.1%}"
                else:
                    ratio_text = "n/a"
                print(f"  {section_key}: {section_val:.1f} ({ratio_text} of prompt avg)")

        base_code_total = section_avgs.get("base_code")
        base_sections = {
            key: val
            for key, val in section_avgs.items()
            if key.startswith("base_code_") and key != "base_code"
        }
        top_base_sections = sorted(base_sections.items(), key=lambda item: item[1], reverse=True)[:PROMPT_SECTION_BASE_TOP_N]
        if top_base_sections:
            print("Base-code breakdown (avg chars per prompt):")
            for section_key, section_val in top_base_sections:
                if base_code_total:
                    ratio = section_val / base_code_total
                    ratio_text = f"{ratio:.1%} of base_code"
                else:
                    ratio_text = "n/a"
                print(f"  {section_key}: {section_val:.1f} ({ratio_text})")

        if prompt_top_key_counts:
            top_key, top_count = sorted(prompt_top_key_counts.items(), key=lambda item: item[1], reverse=True)[0]
            print(f"Top prompt section per round: {top_key} ({top_count}/{total_rounds})")
        if prompt_top_base_key_counts:
            top_base_key, top_base_count = sorted(prompt_top_base_key_counts.items(), key=lambda item: item[1], reverse=True)[0]
            print(f"Top base-code section per round: {top_base_key} ({top_base_count}/{total_rounds})")

    if show_syntax_examples:
        total_syntax = sum(syntax_counts.values())
        print(f"Syntax errors found: {total_syntax}")
        if total_syntax:
            for category, count in sorted(syntax_counts.items(), key=lambda item: item[1], reverse=True):
                print(f"  {category}: {count}")
                for example in syntax_examples.get(category, []):
                    print(f"    {example}")

    if show_analysis_tag_examples:
        total_tags = sum(analysis_tag_counts.values())
        print(f"Analysis card tags: {total_tags}")
        if total_tags:
            valid_count = analysis_tag_counts.get("valid", 0)
            recoverable_count = analysis_tag_counts.get("recoverable", 0)
            invalid_count = analysis_tag_counts.get("invalid", 0)
            print(f"  valid: {valid_count} | recoverable: {recoverable_count} | invalid: {invalid_count}")
        if analysis_invalid_counts:
            print("Top invalid analysis tags:")
            for raw, count in analysis_invalid_counts.most_common(ANALYSIS_TAG_TOP_N):
                print(f"  {count}x {_trim_text(raw)}")
        if analysis_invalid_pattern_counts:
            print("Invalid analysis tag patterns:")
            for pattern, count in analysis_invalid_pattern_counts.most_common():
                print(f"  {pattern}: {count}")
        if analysis_invalid_suggestions:
            print("Suggested tags from invalid entries:")
            for tag, count in analysis_invalid_suggestions.most_common(ANALYSIS_TAG_TOP_N):
                print(f"  {tag}: {count}")
        if analysis_recoverable_examples:
            print("Recoverable analysis tag examples:")
            for example in analysis_recoverable_examples:
                print(f"  {example}")
        if analysis_invalid_examples:
            print("Invalid analysis tag examples:")
            for example in analysis_invalid_examples:
                print(f"  {example}")

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
    parser.add_argument(
        "--show-syntax-examples",
        action="store_true",
        help="Show sample LLM syntax error lines (non-code prefixes, JS comments, missing 'in').",
    )
    parser.add_argument(
        "--syntax-example-limit",
        type=int,
        default=3,
        help="Maximum examples per syntax error category.",
    )
    parser.add_argument(
        "--show-analysis-tag-examples",
        action="store_true",
        help="Show sample invalid/recoverable analysis card signature tags.",
    )
    parser.add_argument(
        "--analysis-tag-example-limit",
        type=int,
        default=3,
        help="Maximum examples per analysis tag category.",
    )
    args = parser.parse_args()

    files = _history_files(args.path, args.all)
    if not files:
        print(f"No execution history files found at {args.path}")
        return 1

    for file_path in files:
        _summarize_file(
            file_path,
            args.rounds,
            args.baseline_window,
            args.show_rounds,
            args.show_syntax_examples,
            args.syntax_example_limit,
            args.show_analysis_tag_examples,
            args.analysis_tag_example_limit,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
