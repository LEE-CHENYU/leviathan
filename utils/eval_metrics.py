from __future__ import annotations

import ast
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from utils.error_tags import classify_error_tag

SIGNATURE_TAG_ORDER = [
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

EXPECTED_ROUND_METRICS = [
    "round_end_population_avg_survival_delta",
    "population_avg_survival_delta",
    "population_std_survival_delta",
    "population_signature_unique_ratio",
    "population_signature_entropy",
    "population_signature_dominant_share",
    "plan_alignment_rate",
    "plan_alignment_plan_coverage",
    "plan_alignment_avg_match_score",
    "plan_alignment_plan_samples",
    "plan_alignment_total_actions",
    "plan_alignment_missing_plans",
    "plan_ineligible_tag_rate",
    "plan_only_tag_rate",
    "plan_tag_total",
    "plan_feasible_tag_total",
    "plan_ineligible_tag_count",
    "plan_only_tag_count",
    "plan_feasibility_samples",
    "plan_feasibility_missing",
    "agent_code_error_rate",
    "agent_code_error_count",
    "agent_code_error_tag_counts",
    "agent_code_error_type_counts",
    "memory_active_coverage",
    "memory_missing_count",
    "memory_orphan_count",
    "mechanism_attempted_count",
    "mechanism_approved_count",
    "mechanism_executed_count",
    "mechanism_error_count",
    "mechanism_error_type_counts",
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


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def collect_action_records(round_record: dict) -> Tuple[List[dict], Optional[str]]:
    for key in ("agent_actions", "actions"):
        actions = round_record.get(key)
        if isinstance(actions, list):
            return actions, key
    return [], None


def collect_action_survival_deltas(round_record: dict) -> Tuple[List[float], int, Optional[str]]:
    actions, source = collect_action_records(round_record)
    deltas: List[float] = []
    action_count = 0
    if not isinstance(actions, list):
        return deltas, action_count, source

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_count += 1
        delta = safe_float(action.get("performance_change"))
        if delta is None:
            old_stats = action.get("old_stats") or {}
            new_stats = action.get("new_stats") or {}
            old_survival = safe_float(old_stats.get("survival_chance"))
            new_survival = safe_float(new_stats.get("survival_chance"))
            if old_survival is not None and new_survival is not None:
                delta = new_survival - old_survival
        if delta is not None:
            deltas.append(delta)

    return deltas, action_count, source


def _collect_round_end_survival_deltas(round_record: dict) -> Tuple[List[float], Optional[str]]:
    deltas: List[float] = []
    round_end_deltas = round_record.get("round_end_deltas")
    if isinstance(round_end_deltas, dict) and round_end_deltas:
        for delta in round_end_deltas.values():
            if isinstance(delta, dict):
                val = safe_float(delta.get("survival_chance"))
            else:
                val = safe_float(delta)
            if val is not None:
                deltas.append(val)
        if deltas:
            return deltas, "round_end_deltas"

    start_snapshot = round_record.get("round_start_snapshot") or {}
    end_snapshot = round_record.get("round_end_snapshot") or {}
    if isinstance(start_snapshot, dict) and isinstance(end_snapshot, dict):
        for member_id, start_stats in start_snapshot.items():
            end_stats = end_snapshot.get(member_id)
            if not isinstance(start_stats, dict) or not isinstance(end_stats, dict):
                continue
            start_survival = safe_float(start_stats.get("survival_chance"))
            end_survival = safe_float(end_stats.get("survival_chance"))
            if start_survival is None or end_survival is None:
                continue
            deltas.append(end_survival - start_survival)
        if deltas:
            return deltas, "round_start_snapshot/round_end_snapshot"

    return deltas, None


def extract_action_signature(code_str: str) -> tuple:
    if not code_str:
        return tuple()
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

    return tuple([tag for tag in SIGNATURE_TAG_ORDER if tag in signature])


def collect_action_signatures(round_record: dict) -> Tuple[Dict[object, tuple], Optional[str]]:
    actions, source = collect_action_records(round_record)
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
            signatures[key] = extract_action_signature(code)

    return signatures, source


def fallback_metrics(round_record: dict) -> Tuple[Dict[str, float], Dict[str, str]]:
    metrics: Dict[str, float] = {}
    sources: Dict[str, str] = {}

    deltas, delta_source = _collect_round_end_survival_deltas(round_record)
    action_count = 0
    action_source = None

    if not deltas:
        deltas, action_count, action_source = collect_action_survival_deltas(round_record)
        delta_source = action_source or "agent_actions"

    actions, action_source = collect_action_records(round_record)
    if isinstance(actions, list):
        action_count = len(actions)

    avg_delta = sum(deltas) / len(deltas) if deltas else None
    if avg_delta is not None:
        metrics["population_avg_survival_delta"] = avg_delta
        metrics["round_end_population_avg_survival_delta"] = avg_delta
        sources["population_avg_survival_delta"] = delta_source or "agent_actions"
        sources["round_end_population_avg_survival_delta"] = delta_source or "agent_actions"
        if len(deltas) > 1:
            variance = sum((delta - avg_delta) ** 2 for delta in deltas) / len(deltas)
            std_delta = math.sqrt(variance)
        else:
            std_delta = 0.0
        metrics["population_std_survival_delta"] = std_delta
        sources["population_std_survival_delta"] = delta_source or "agent_actions"

    total_actions_count = None
    total_actions_source = None
    if action_source is not None:
        total_actions_count = action_count
        total_actions_source = action_source
    elif deltas:
        total_actions_count = len(deltas)
        total_actions_source = delta_source or "round_end_deltas"
    if total_actions_count is not None:
        metrics["plan_alignment_total_actions"] = total_actions_count
        sources["plan_alignment_total_actions"] = total_actions_source or "agent_actions"

    errors = round_record.get("errors", {}).get("agent_code_errors")
    if isinstance(errors, list):
        metrics["agent_code_error_count"] = len(errors)
        sources["agent_code_error_count"] = "errors/agent_code_errors"
        tag_counts = Counter()
        type_counts = Counter()
        for error_info in errors:
            error_category = error_info.get("error_category") if isinstance(error_info, dict) else None
            if error_category:
                type_counts[error_category] += 1
            else:
                type_counts["unknown"] += 1
            code_str = error_info.get("code", "") if isinstance(error_info, dict) else ""
            signature = extract_action_signature(code_str)
            if signature:
                for tag in signature:
                    tag_counts[tag] += 1
                continue
            error_tag = classify_error_tag(error_info)
            tag_counts[error_tag or "unknown"] += 1
        metrics["agent_code_error_tag_counts"] = dict(tag_counts)
        sources["agent_code_error_tag_counts"] = "errors/agent_code_errors"
        metrics["agent_code_error_type_counts"] = dict(type_counts)
        sources["agent_code_error_type_counts"] = "errors/agent_code_errors"

        denominator = None
        if delta_source in {"round_end_deltas", "round_start_snapshot/round_end_snapshot"} and deltas:
            denominator = len(deltas)
        elif action_source is not None:
            denominator = action_count
        elif deltas:
            denominator = len(deltas)
        if denominator:
            error_rate = len(errors) / denominator
            metrics["agent_code_error_rate"] = error_rate
            sources["agent_code_error_rate"] = f"errors/{action_source or delta_source or 'agent_actions'}"

    signature_map, signature_source = collect_action_signatures(round_record)
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
        source_label = (
            f"{signature_source}/code_executed" if signature_source else "agent_actions/code_executed"
        )
        sources["population_signature_unique_ratio"] = source_label
        sources["population_signature_entropy"] = source_label
        sources["population_signature_dominant_share"] = source_label

    mods_record = round_record.get("mechanism_modifications")
    if isinstance(mods_record, dict):
        attempts = mods_record.get("attempts") or []
        executed = mods_record.get("executed") or []
        attempts_count = len(attempts) if isinstance(attempts, list) else 0
        executed_count = len(executed) if isinstance(executed, list) else 0
        approved_count = mods_record.get("approved_count")
        approved_ids = mods_record.get("approved_ids")
        approved_source = "mechanism_modifications/approved_count"
        if approved_count is None:
            if isinstance(approved_ids, list):
                approved_count = len(approved_ids)
                approved_source = "mechanism_modifications/approved_ids"
            elif attempts_count:
                approved_count = attempts_count
                approved_source = "mechanism_modifications:implicit_attempts"
            elif executed_count:
                approved_count = executed_count
                approved_source = "mechanism_modifications:implicit_executed"
            else:
                approved_count = 0
                approved_source = "mechanism_modifications:implicit_zero"
        else:
            try:
                approved_count = int(approved_count)
            except (TypeError, ValueError):
                approved_count = 0
                approved_source = "mechanism_modifications:invalid_approved_count"
        metrics["mechanism_attempted_count"] = attempts_count
        metrics["mechanism_approved_count"] = int(approved_count)
        metrics["mechanism_executed_count"] = executed_count
        sources["mechanism_attempted_count"] = "mechanism_modifications"
        sources["mechanism_approved_count"] = approved_source
        sources["mechanism_executed_count"] = "mechanism_modifications"

    mech_errors = round_record.get("errors", {}).get("mechanism_errors") or []
    if isinstance(mech_errors, list):
        metrics["mechanism_error_count"] = len(mech_errors)
        sources["mechanism_error_count"] = "errors/mechanism_errors"
        type_counts = Counter()
        for error_info in mech_errors:
            error_category = error_info.get("error_category") if isinstance(error_info, dict) else None
            if error_category:
                type_counts[error_category] += 1
            else:
                type_counts["unknown"] += 1
        metrics["mechanism_error_type_counts"] = dict(type_counts)
        sources["mechanism_error_type_counts"] = "errors/mechanism_errors"

    contract_stats = round_record.get("contract_stats")
    if isinstance(contract_stats, dict):
        metrics["contract_total"] = contract_stats.get("total_contracts", 0)
        metrics["contract_pending"] = contract_stats.get("pending", 0)
        metrics["contract_active"] = contract_stats.get("active", 0)
        metrics["contract_completed"] = contract_stats.get("completed", 0)
        metrics["contract_failed"] = contract_stats.get("failed", 0)
        sources["contract_total"] = "contract_stats"
        sources["contract_pending"] = "contract_stats"
        sources["contract_active"] = "contract_stats"
        sources["contract_completed"] = "contract_stats"
        sources["contract_failed"] = "contract_stats"

    partner_counts = round_record.get("contract_partner_counts")
    if isinstance(partner_counts, dict):
        count_values = [safe_float(v) for v in partner_counts.values()]
        count_values = [v for v in count_values if v is not None]
        if count_values:
            metrics["contract_partner_unique_avg"] = sum(count_values) / len(count_values)
        else:
            metrics["contract_partner_unique_avg"] = 0.0
        sources["contract_partner_unique_avg"] = "contract_partner_counts"

    partner_top_share = round_record.get("contract_partner_top_share")
    if isinstance(partner_top_share, dict):
        share_values = [safe_float(v) for v in partner_top_share.values()]
        share_values = [v for v in share_values if v is not None]
        if share_values:
            metrics["contract_partner_top_share_avg"] = sum(share_values) / len(share_values)
        else:
            metrics["contract_partner_top_share_avg"] = 0.0
        sources["contract_partner_top_share_avg"] = "contract_partner_top_share"

    physics_stats = round_record.get("physics_stats")
    if isinstance(physics_stats, dict):
        metrics["physics_active_constraints"] = physics_stats.get("active_constraints", 0)
        domains = physics_stats.get("domains") or []
        metrics["physics_domain_count"] = len(domains) if isinstance(domains, list) else 0
        sources["physics_active_constraints"] = "physics_stats"
        sources["physics_domain_count"] = "physics_stats"

    return metrics, sources


def combine_round_metrics(
    round_record: dict,
    expected: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str], List[str], Optional[float]]:
    """
    Combine round metrics with derived fallbacks and report coverage.

    Returns:
        combined_metrics: merged round_metrics with fallback-derived values.
        derived_metrics: metrics derived from fallbacks.
        derived_sources: source labels for derived metrics.
        missing_metrics: expected metrics with missing/None values.
        coverage: fraction of expected metrics present (None if expected empty).
    """
    round_metrics = round_record.get("round_metrics") or {}
    combined = dict(round_metrics) if isinstance(round_metrics, dict) else {}
    derived_metrics, derived_sources = fallback_metrics(round_record)
    for key, value in derived_metrics.items():
        if key not in combined or combined.get(key) is None:
            combined[key] = value

    expected_metrics = expected or EXPECTED_ROUND_METRICS
    missing = [key for key in expected_metrics if combined.get(key) is None]
    coverage = (
        (len(expected_metrics) - len(missing)) / len(expected_metrics)
        if expected_metrics
        else None
    )
    return combined, derived_metrics, derived_sources, missing, coverage
