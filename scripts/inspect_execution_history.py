#!/usr/bin/env python3
import argparse
import ast
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

METRIC_KEYS = [
    "round_end_population_avg_survival_delta",
    "population_avg_survival_delta",
    "population_signature_unique_ratio",
    "population_signature_entropy",
    "population_signature_dominant_share",
    "plan_alignment_rate",
    "plan_alignment_plan_coverage",
    "plan_ineligible_tag_rate",
    "plan_only_tag_rate",
    "agent_code_error_rate",
    "memory_active_coverage",
]

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
}


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _collect_metric(
    round_record: dict,
    key: str,
    fallback_metrics: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[float], bool]:
    round_metrics = round_record.get("round_metrics") or {}
    if key in round_metrics:
        return _safe_float(round_metrics.get(key)), False
    round_end_metrics = round_record.get("round_end_metrics") or {}
    if key in round_end_metrics:
        return _safe_float(round_end_metrics.get(key)), False
    if fallback_metrics and key in fallback_metrics:
        return _safe_float(fallback_metrics.get(key)), True
    return None, False


def _collect_action_records(round_record: dict) -> Tuple[List[dict], Optional[str]]:
    for key in ("agent_actions", "actions"):
        actions = round_record.get(key)
        if isinstance(actions, list):
            return actions, key
    return [], None


def _collect_action_survival_deltas(round_record: dict) -> Tuple[List[float], int, Optional[str]]:
    actions, source = _collect_action_records(round_record)
    deltas: List[float] = []
    action_count = 0
    if not isinstance(actions, list):
        return deltas, action_count, source

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_count += 1
        delta = _safe_float(action.get("performance_change"))
        if delta is None:
            old_stats = action.get("old_stats") or {}
            new_stats = action.get("new_stats") or {}
            old_survival = _safe_float(old_stats.get("survival_chance"))
            new_survival = _safe_float(new_stats.get("survival_chance"))
            if old_survival is not None and new_survival is not None:
                delta = new_survival - old_survival
        if delta is not None:
            deltas.append(delta)

    return deltas, action_count, source


def _extract_action_signature(code_str: str) -> tuple:
    if not code_str:
        return tuple()
    tag_order = [
        "attack",
        "offer",
        "offer_land",
        "bear",
        "expand",
        "message",
        "contracts",
        "market",
        "resources",
        "businesses",
    ]

    call_tags = {
        "attack": "attack",
        "offer": "offer",
        "offer_land": "offer_land",
        "bear": "bear",
        "expand": "expand",
        "send_message": "message",
    }
    attr_tags = {
        "contracts": "contracts",
        "market": "market",
        "resources": "resources",
        "businesses": "businesses",
    }

    signature = set()
    tree = None
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        tree = None

    if tree is not None:
        engine_aliases = {"execution_engine"}
        message_aliases = {"send_message"}

        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = None
                value = None
                if isinstance(node, ast.Assign) and node.targets:
                    if isinstance(node.targets[0], ast.Name):
                        target = node.targets[0].id
                    value = node.value
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        target = node.target.id
                    value = node.value

                if target and value is not None:
                    if isinstance(value, ast.Name) and value.id in engine_aliases:
                        engine_aliases.add(target)
                    elif isinstance(value, ast.Attribute):
                        if (
                            isinstance(value.value, ast.Name)
                            and value.value.id in engine_aliases
                            and value.attr == "send_message"
                        ):
                            message_aliases.add(target)

            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id in engine_aliases:
                    tag = attr_tags.get(node.attr)
                    if tag:
                        signature.add(tag)

            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name) and func.value.id in engine_aliases:
                        tag = call_tags.get(func.attr)
                        if tag:
                            signature.add(tag)
                    elif isinstance(func.value, ast.Attribute):
                        if (
                            isinstance(func.value.value, ast.Name)
                            and func.value.value.id in engine_aliases
                        ):
                            tag = attr_tags.get(func.value.attr)
                            if tag:
                                signature.add(tag)
                elif isinstance(func, ast.Name):
                    if func.id in message_aliases:
                        signature.add("message")

    if not signature:
        patterns = {
            "attack": ("execution_engine.attack(",),
            "offer": ("execution_engine.offer(",),
            "offer_land": ("execution_engine.offer_land(",),
            "bear": ("execution_engine.bear(",),
            "expand": ("execution_engine.expand(",),
            "message": ("execution_engine.send_message(", "send_message("),
            "contracts": ("execution_engine.contracts",),
            "market": ("execution_engine.market",),
            "resources": ("execution_engine.resources",),
            "businesses": ("execution_engine.businesses",),
        }

        for tag, needles in patterns.items():
            if any(needle in code_str for needle in needles):
                signature.add(tag)

    return tuple([tag for tag in tag_order if tag in signature])


def _collect_action_signatures(round_record: dict) -> Tuple[Dict[object, tuple], Optional[str]]:
    actions, source = _collect_action_records(round_record)
    signatures: Dict[object, tuple] = {}
    if not isinstance(actions, list):
        return signatures, source

    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        member_id = action.get("member_id")
        key = member_id if member_id is not None else f"action_{idx}"
        code = action.get("code_executed") or action.get("code")
        if isinstance(code, str):
            signatures[key] = _extract_action_signature(code)

    return signatures, source


def _fallback_metrics(round_record: dict) -> Tuple[Dict[str, float], Dict[str, str]]:
    metrics: Dict[str, float] = {}
    sources: Dict[str, str] = {}

    deltas, action_count, action_source = _collect_action_survival_deltas(round_record)
    avg_delta = _avg(deltas)
    if avg_delta is not None:
        metrics["population_avg_survival_delta"] = avg_delta
        sources["population_avg_survival_delta"] = action_source or "agent_actions"
        metrics["round_end_population_avg_survival_delta"] = avg_delta
        sources["round_end_population_avg_survival_delta"] = action_source or "agent_actions"

    errors = round_record.get("errors", {}).get("agent_code_errors")
    if isinstance(errors, list) and action_count:
        error_rate = len(errors) / action_count
        metrics["agent_code_error_rate"] = error_rate
        if action_source:
            sources["agent_code_error_rate"] = f"errors/{action_source}"
        else:
            sources["agent_code_error_rate"] = "errors/agent_actions"

    signature_map, signature_source = _collect_action_signatures(round_record)
    if signature_map:
        signature_counts = Counter(signature_map.values())
        total = len(signature_map)
        unique = len(signature_counts)
        diversity_ratio = unique / total if total else 0.0
        dominant_share = (
            signature_counts.most_common(1)[0][1] / total
            if signature_counts and total
            else 0.0
        )
        entropy = 0.0
        if total:
            for count in signature_counts.values():
                if count <= 0:
                    continue
                p = count / total
                entropy -= p * math.log2(p)

        metrics["population_signature_unique_ratio"] = diversity_ratio
        metrics["population_signature_entropy"] = entropy
        metrics["population_signature_dominant_share"] = dominant_share
        source_label = f"{signature_source}/code_executed" if signature_source else "agent_actions/code_executed"
        sources["population_signature_unique_ratio"] = source_label
        sources["population_signature_entropy"] = source_label
        sources["population_signature_dominant_share"] = source_label

    return metrics, sources


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
    metric_values: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}
    missing_metric_counts = {key: 0 for key in METRIC_KEYS}
    derived_metric_counts = {key: 0 for key in METRIC_KEYS}
    derived_metric_sources: Dict[str, set] = {key: set() for key in METRIC_KEYS}

    for idx, round_record in enumerate(rounds, start=1):
        fallback_metrics, fallback_sources = _fallback_metrics(round_record)
        prompts = _extract_prompts(round_record)
        diversity_hits = _count_diversity_adjustments(prompts)
        prompt_total += len(prompts)
        prompt_diversity += diversity_hits

        if show_rounds:
            print(f"  Round {round_record.get('round_number', idx)}: prompts={len(prompts)}, diversity_adjustments={diversity_hits}")

        for key in METRIC_KEYS:
            value, used_fallback = _collect_metric(round_record, key, fallback_metrics)
            if value is None:
                missing_metric_counts[key] += 1
                continue
            metric_values[key].append(value)
            if used_fallback:
                derived_metric_counts[key] += 1
                source = fallback_sources.get(key)
                if source:
                    derived_metric_sources[key].add(source)

    if prompt_total:
        rate = prompt_diversity / prompt_total
    else:
        rate = None
    print(f"Prompts scanned: {prompt_total} | Diversity adjustment: {prompt_diversity} ({_format_value(rate)})")

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
            fallback_metrics, _ = _fallback_metrics(round_record)
            value, _ = _collect_metric(round_record, key, fallback_metrics)
            if value is not None:
                baseline_values.append(value)
        last_fallback, _ = _fallback_metrics(last_round)
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
