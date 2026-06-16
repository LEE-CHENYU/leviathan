from __future__ import annotations

import ast
import json
import math
import re
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
_LENIENT_TAG_SEARCH_ORDER = sorted(SIGNATURE_TAG_ORDER, key=lambda tag: (-len(tag), tag))

_ACTION_TAG_ALIASES = {
    "messages": "message",
    "msg": "message",
    "contract": "contracts",
    "contracts": "contracts",
    "markets": "market",
    "resource": "resources",
    "business": "businesses",
    "offerland": "offer_land",
    "offer-land": "offer_land",
}

_CODE_START_PREFIXES = (
    "def ",
    "class ",
    "import ",
    "from ",
    "@",
    "#",
    "'''",
    '"""',
)

_MISSING_IN_PATTERN = re.compile(r"^\s*(if|elif)\s+\S+\s+not\s+(?!in\b)\S+")
_TAG_EDGE_PUNCT_RE = re.compile(r"^[^a-z0-9_]+|[^a-z0-9_]+$")
_SYNTAX_ERROR_SUBSET_KEYS = {
    "count": "total",
    "near_end_count": "near_end",
    "near_end_10pct_count": "near_end_10pct",
    "js_comment_count": "js_comment",
    "non_code_prefix_count": "non_code_prefix",
    "missing_in_count": "missing_in",
}

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
    "plan_feasibility_missing_reason_counts",
    "agent_code_error_rate",
    "agent_code_error_count",
    "agent_code_error_tag_counts",
    "agent_code_error_type_counts",
    "llm_finish_reason_counts",
    "llm_finish_reason_total",
    "llm_finish_reason_length_count",
    "llm_finish_reason_missing_count",
    "llm_syntax_error_count",
    "llm_syntax_error_near_end_count",
    "llm_syntax_error_near_end_10pct_count",
    "llm_syntax_error_mid_count",
    "llm_syntax_error_unknown_count",
    "llm_syntax_error_near_end_rate",
    "llm_syntax_error_near_end_10pct_rate",
    "llm_syntax_error_line_ratio_avg",
    "llm_syntax_error_line_ratio_min",
    "llm_syntax_error_line_ratio_max",
    "llm_syntax_error_line_ratio_samples",
    "llm_syntax_error_js_comment_count",
    "llm_syntax_error_non_code_prefix_count",
    "llm_syntax_error_missing_in_count",
    "llm_syntax_error_agent_count",
    "llm_syntax_error_agent_near_end_count",
    "llm_syntax_error_agent_near_end_10pct_count",
    "llm_syntax_error_agent_js_comment_count",
    "llm_syntax_error_agent_non_code_prefix_count",
    "llm_syntax_error_agent_missing_in_count",
    "llm_syntax_error_mechanism_count",
    "llm_syntax_error_mechanism_near_end_count",
    "llm_syntax_error_mechanism_near_end_10pct_count",
    "llm_syntax_error_mechanism_js_comment_count",
    "llm_syntax_error_mechanism_non_code_prefix_count",
    "llm_syntax_error_mechanism_missing_in_count",
    "llm_syntax_error_finish_reason_counts",
    "llm_syntax_error_finish_reason_total",
    "llm_syntax_error_finish_reason_length_count",
    "llm_syntax_error_finish_reason_missing_count",
    "llm_syntax_error_request_cap_count",
    "llm_syntax_error_completion_at_request_cap_count",
    "llm_syntax_error_completion_at_request_cap_rate",
    "llm_prompt_char_count_avg",
    "llm_prompt_char_count_min",
    "llm_prompt_char_count_max",
    "llm_prompt_section_chars_total",
    "llm_prompt_section_chars_avg",
    "llm_prompt_section_chars_max",
    "llm_prompt_section_entry_count",
    "llm_prompt_section_top_avg_key",
    "llm_prompt_section_top_avg_chars",
    "llm_prompt_section_top_base_code_key",
    "llm_prompt_section_top_base_code_chars",
    "llm_prompt_section_base_code_ratio",
    "llm_prompt_section_top_avg_ratio",
    "llm_prompt_section_top_base_code_ratio",
    "analysis_card_signature_card_count",
    "analysis_card_signature_tag_total",
    "analysis_card_signature_invalid_tag_count",
    "analysis_card_signature_invalid_tag_rate",
    "analysis_card_signature_recoverable_tag_count",
    "analysis_card_signature_recoverable_tag_rate",
    "analysis_card_signature_empty_card_count",
    "analysis_card_signature_empty_card_rate",
    "memory_active_coverage",
    "memory_missing_count",
    "memory_orphan_count",
    "mechanism_attempted_count",
    "mechanism_approved_count",
    "mechanism_executed_count",
    "mechanism_error_count",
    "mechanism_error_type_counts",
    "mechanism_judge_approved_count",
    "mechanism_judge_rejected_count",
    "mechanism_judge_missing_count",
    "mechanism_judge_rejection_reason_counts",
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


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_js_comment_line(text: object) -> bool:
    if not isinstance(text, str):
        return False
    return text.lstrip().startswith("//")


def _is_non_code_prefix_line(text: object) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.lstrip()
    if not stripped:
        return False
    if _is_js_comment_line(stripped):
        return False
    for prefix in _CODE_START_PREFIXES:
        if stripped.startswith(prefix):
            return False
    return True


def _is_missing_in_operator_line(text: object) -> bool:
    if not isinstance(text, str):
        return False
    return bool(_MISSING_IN_PATTERN.match(text))


def _is_syntax_error(error_info: dict) -> bool:
    if not isinstance(error_info, dict):
        return False
    if error_info.get("error_category") == "llm_syntax_error":
        return True
    details = error_info.get("error_details")
    if not isinstance(details, dict):
        return False
    if details.get("error_type") == "SyntaxError":
        return True
    msg = str(details.get("error_msg") or "").lower()
    return any(token in msg for token in ("syntax", "unterminated", "unexpected eof", "eof while scanning"))


def _collect_syntax_error_stats(error_lists: List[object]) -> Dict[str, float]:
    total = 0
    near_end = 0
    near_end_10pct = 0
    mid = 0
    unknown = 0
    with_line = 0
    line_ratios: List[float] = []
    js_comment = 0
    non_code_prefix = 0
    missing_in = 0
    for errors in error_lists:
        if not isinstance(errors, list):
            continue
        for error_info in errors:
            if not _is_syntax_error(error_info):
                continue
            total += 1
            details = error_info.get("error_details") if isinstance(error_info, dict) else None
            error_line = None
            line_count = None
            out_of_range = False
            error_text = None
            if isinstance(details, dict):
                error_line = _safe_int(details.get("error_line"))
                line_count = _safe_int(details.get("code_line_count"))
                out_of_range = bool(details.get("error_line_out_of_range"))
                error_text = details.get("error_text")
            if _is_js_comment_line(error_text):
                js_comment += 1
            elif error_line == 1 and _is_non_code_prefix_line(error_text):
                non_code_prefix += 1
            if _is_missing_in_operator_line(error_text):
                missing_in += 1
            if out_of_range:
                error_line = None
            if line_count is None and isinstance(error_info, dict):
                code_stats = error_info.get("code_stats") or {}
                if isinstance(code_stats, dict):
                    line_count = _safe_int(
                        code_stats.get("cleaned_lines") or code_stats.get("raw_lines")
                    )
            if error_line is None or line_count is None or line_count <= 0:
                unknown += 1
                continue
            if error_line < 1 or error_line > line_count:
                unknown += 1
                continue
            with_line += 1
            tail_threshold = max(1, line_count - 3)
            if error_line >= tail_threshold:
                near_end += 1
            else:
                mid += 1
            line_ratio = error_line / float(line_count)
            line_ratios.append(line_ratio)
            if line_ratio >= 0.9:
                near_end_10pct += 1
    stats: Dict[str, float] = {
        "total": total,
        "near_end": near_end,
        "near_end_10pct": near_end_10pct,
        "mid": mid,
        "unknown": unknown,
        "with_line": with_line,
        "js_comment": js_comment,
        "non_code_prefix": non_code_prefix,
        "missing_in": missing_in,
    }
    if line_ratios:
        stats["line_ratio_avg"] = sum(line_ratios) / len(line_ratios)
        stats["line_ratio_min"] = min(line_ratios)
        stats["line_ratio_max"] = max(line_ratios)
        stats["line_ratio_samples"] = len(line_ratios)
    else:
        stats["line_ratio_avg"] = 0.0
        stats["line_ratio_min"] = 0.0
        stats["line_ratio_max"] = 0.0
        stats["line_ratio_samples"] = 0
    if with_line:
        stats["near_end_rate"] = near_end / with_line
        stats["near_end_10pct_rate"] = near_end_10pct / with_line
    return stats


def _collect_syntax_error_llm_metadata_stats(error_lists: List[object]) -> Dict[str, float]:
    finish_reason_counts = Counter()
    finish_reason_total = 0
    finish_reason_missing = 0
    request_cap_count = 0
    completion_at_cap = 0
    completion_cap_pairs = 0

    for errors in error_lists:
        if not isinstance(errors, list):
            continue
        for error_info in errors:
            if not _is_syntax_error(error_info):
                continue
            if not isinstance(error_info, dict):
                continue
            metadata = error_info.get("llm_metadata")
            if not isinstance(metadata, dict):
                metadata = error_info.get("llm_metadata_primary")
            if not isinstance(metadata, dict):
                continue

            finish_reason_total += 1
            finish_reason = metadata.get("finish_reason")
            if finish_reason is None or finish_reason == "":
                finish_reason_missing += 1
            else:
                finish_reason_counts[str(finish_reason)] += 1

            cap = None
            if "request_max_completion_tokens" in metadata:
                cap = safe_float(metadata.get("request_max_completion_tokens"))
            if cap is None and "request_max_tokens" in metadata:
                cap = safe_float(metadata.get("request_max_tokens"))
            if cap is not None:
                request_cap_count += 1

            completion_val = safe_float(metadata.get("completion_tokens"))
            if cap is not None and completion_val is not None:
                completion_cap_pairs += 1
                if completion_val >= cap:
                    completion_at_cap += 1

    stats: Dict[str, float] = {
        "finish_reason_counts": dict(finish_reason_counts),
        "finish_reason_total": finish_reason_total,
        "finish_reason_missing": finish_reason_missing,
        "finish_reason_length_count": finish_reason_counts.get("length", 0),
        "request_cap_count": request_cap_count,
        "completion_at_cap_count": completion_at_cap,
        "completion_at_cap_rate": 0.0,
    }
    if completion_cap_pairs:
        stats["completion_at_cap_rate"] = completion_at_cap / completion_cap_pairs
    return stats


def _apply_syntax_error_subset(
    metrics: Dict[str, float],
    sources: Dict[str, str],
    stats: Dict[str, float],
    prefix: str,
    source_label: str,
) -> None:
    if not stats:
        return
    for suffix, stat_key in _SYNTAX_ERROR_SUBSET_KEYS.items():
        metric_key = f"{prefix}_{suffix}"
        metrics[metric_key] = stats.get(stat_key, 0)
        sources[metric_key] = source_label


def _collect_llm_metadata(round_record: dict) -> List[dict]:
    metadata_entries: List[dict] = []

    def add_metadata(meta: object) -> None:
        if isinstance(meta, dict) and meta:
            metadata_entries.append(meta)

    def add_entry(entry: object) -> None:
        if not isinstance(entry, dict):
            return
        add_metadata(entry.get("llm_metadata"))
        add_metadata(entry.get("llm_metadata_primary"))

    generated = round_record.get("generated_code") or {}
    if isinstance(generated, dict):
        for entry in generated.values():
            add_entry(entry)

    mods_record = round_record.get("mechanism_modifications")
    if isinstance(mods_record, dict):
        attempts = mods_record.get("attempts") or []
        if isinstance(attempts, list):
            for attempt in attempts:
                add_entry(attempt)

    return metadata_entries


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


def _normalize_action_tag(raw: object, strip_punct: bool = False) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if not strip_punct:
        text = text.replace(" ", "_").replace("-", "_")
        text = _ACTION_TAG_ALIASES.get(text, text)
        return text if text in SIGNATURE_TAG_ORDER else None

    candidate = _TAG_EDGE_PUNCT_RE.sub("", text)
    candidate = candidate.replace(" ", "_").replace("-", "_")
    candidate = _ACTION_TAG_ALIASES.get(candidate, candidate)
    if candidate in SIGNATURE_TAG_ORDER:
        return candidate

    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not normalized:
        return None
    candidate = _ACTION_TAG_ALIASES.get(normalized, normalized)
    if candidate in SIGNATURE_TAG_ORDER:
        return candidate

    for tag in _LENIENT_TAG_SEARCH_ORDER:
        pattern = rf"(?:^|_){re.escape(tag)}(?:_|$)"
        if re.search(pattern, normalized):
            return tag

    for alias, tag in _ACTION_TAG_ALIASES.items():
        pattern = rf"(?:^|_){re.escape(alias)}(?:_|$)"
        if re.search(pattern, normalized):
            return tag

    return None


def _coerce_card_tags(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        tags: List[str] = []
        for item in value:
            tags.extend(_coerce_card_tags(item))
        return tags
    if isinstance(value, dict):
        for key in ("actions", "action", "tags", "signature", "baseline_signature", "variation_signature"):
            if key in value:
                return _coerce_card_tags(value.get(key))
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith(("{", "[")) and text.endswith(("}", "]")):
            parsed = None
            try:
                parsed = json.loads(text)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    parsed = None
            if parsed is not None and parsed is not value:
                return _coerce_card_tags(parsed)
        if any(token in text for token in (",", "|", "+")):
            return [item.strip() for item in re.split(r"[,+|]", text) if item.strip()]
        return [text]
    text = str(value).strip()
    return [text] if text else []


def _collect_analysis_card_signature_metrics(
    round_record: dict,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    cards = round_record.get("analysis_cards")
    if not isinstance(cards, dict) or not cards:
        return {}, {}

    card_count = 0
    total_tags = 0
    strict_valid = 0
    lenient_valid = 0
    empty_cards = 0

    for card in cards.values():
        if not isinstance(card, dict):
            continue
        card_count += 1
        tags: List[str] = []
        for key in ("baseline_signature", "variation_signature"):
            tags.extend(_coerce_card_tags(card.get(key)))
        card_strict_valid = 0
        card_lenient_valid = 0
        for raw in tags:
            total_tags += 1
            if _normalize_action_tag(raw, strip_punct=False):
                strict_valid += 1
                card_strict_valid += 1
            if _normalize_action_tag(raw, strip_punct=True):
                lenient_valid += 1
                card_lenient_valid += 1
        if card_lenient_valid == 0:
            empty_cards += 1

    if card_count == 0:
        return {}, {}

    invalid_count = max(total_tags - lenient_valid, 0)
    recoverable_count = max(lenient_valid - strict_valid, 0)

    metrics: Dict[str, float] = {
        "analysis_card_signature_card_count": card_count,
        "analysis_card_signature_tag_total": total_tags,
        "analysis_card_signature_invalid_tag_count": invalid_count,
        "analysis_card_signature_recoverable_tag_count": recoverable_count,
        "analysis_card_signature_empty_card_count": empty_cards,
    }
    if total_tags:
        metrics["analysis_card_signature_invalid_tag_rate"] = invalid_count / total_tags
        metrics["analysis_card_signature_recoverable_tag_rate"] = recoverable_count / total_tags
    else:
        metrics["analysis_card_signature_invalid_tag_rate"] = None
        metrics["analysis_card_signature_recoverable_tag_rate"] = None
    if card_count:
        metrics["analysis_card_signature_empty_card_rate"] = empty_cards / card_count
    else:
        metrics["analysis_card_signature_empty_card_rate"] = None

    sources = {key: "analysis_cards" for key in metrics}
    return metrics, sources


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

    llm_metadata_entries = _collect_llm_metadata(round_record)
    if llm_metadata_entries:
        finish_reason_counts = Counter()
        finish_reason_missing = 0
        request_caps: List[float] = []
        request_cap_sources = Counter()
        prompt_tokens: List[float] = []
        completion_tokens: List[float] = []
        completion_at_cap = 0
        completion_cap_pairs = 0
        prompt_char_counts: List[float] = []
        section_totals = Counter()
        section_counts = Counter()
        section_max: Dict[str, float] = {}
        section_entry_count = 0
        for meta in llm_metadata_entries:
            finish_reason = meta.get("finish_reason") if isinstance(meta, dict) else None
            if finish_reason is None or finish_reason == "":
                finish_reason_missing += 1
            else:
                finish_reason_counts[str(finish_reason)] += 1
            if not isinstance(meta, dict):
                continue
            cap = None
            cap_source = None
            if "request_max_completion_tokens" in meta:
                cap = safe_float(meta.get("request_max_completion_tokens"))
                cap_source = "max_completion_tokens"
            if cap is None and "request_max_tokens" in meta:
                cap = safe_float(meta.get("request_max_tokens"))
                cap_source = "max_tokens"
            if cap is not None:
                request_caps.append(cap)
                if cap_source:
                    request_cap_sources[cap_source] += 1
            prompt_val = safe_float(meta.get("prompt_tokens"))
            if prompt_val is not None:
                prompt_tokens.append(prompt_val)
            completion_val = safe_float(meta.get("completion_tokens"))
            if completion_val is not None:
                completion_tokens.append(completion_val)
            if cap is not None and completion_val is not None:
                completion_cap_pairs += 1
                if completion_val >= cap:
                    completion_at_cap += 1
            prompt_char_val = safe_float(meta.get("prompt_char_count"))
            if prompt_char_val is not None:
                prompt_char_counts.append(prompt_char_val)
            section_chars = meta.get("prompt_section_chars")
            if isinstance(section_chars, dict) and section_chars:
                section_entry_count += 1
                for key, val in section_chars.items():
                    val_float = safe_float(val)
                    if val_float is None:
                        continue
                    section_totals[key] += val_float
                    section_counts[key] += 1
                    current_max = section_max.get(key)
                    if current_max is None or val_float > current_max:
                        section_max[key] = val_float
        metrics["llm_finish_reason_counts"] = dict(finish_reason_counts)
        metrics["llm_finish_reason_total"] = len(llm_metadata_entries)
        metrics["llm_finish_reason_length_count"] = finish_reason_counts.get("length", 0)
        metrics["llm_finish_reason_missing_count"] = finish_reason_missing
        source_label = "generated_code/llm_metadata+mechanism_modifications/attempts"
        sources["llm_finish_reason_counts"] = source_label
        sources["llm_finish_reason_total"] = source_label
        sources["llm_finish_reason_length_count"] = source_label
        sources["llm_finish_reason_missing_count"] = source_label
        metrics["llm_request_cap_count"] = len(request_caps)
        sources["llm_request_cap_count"] = source_label
        if request_cap_sources:
            metrics["llm_request_cap_source_counts"] = dict(request_cap_sources)
            sources["llm_request_cap_source_counts"] = source_label
        if request_caps:
            metrics["llm_request_cap_min"] = min(request_caps)
            metrics["llm_request_cap_max"] = max(request_caps)
            metrics["llm_request_cap_avg"] = sum(request_caps) / len(request_caps)
            sources["llm_request_cap_min"] = source_label
            sources["llm_request_cap_max"] = source_label
            sources["llm_request_cap_avg"] = source_label
        if prompt_tokens:
            metrics["llm_prompt_tokens_avg"] = sum(prompt_tokens) / len(prompt_tokens)
            metrics["llm_prompt_tokens_min"] = min(prompt_tokens)
            metrics["llm_prompt_tokens_max"] = max(prompt_tokens)
            sources["llm_prompt_tokens_avg"] = source_label
            sources["llm_prompt_tokens_min"] = source_label
            sources["llm_prompt_tokens_max"] = source_label
        if completion_tokens:
            metrics["llm_completion_tokens_avg"] = sum(completion_tokens) / len(completion_tokens)
            metrics["llm_completion_tokens_min"] = min(completion_tokens)
            metrics["llm_completion_tokens_max"] = max(completion_tokens)
            sources["llm_completion_tokens_avg"] = source_label
            sources["llm_completion_tokens_min"] = source_label
            sources["llm_completion_tokens_max"] = source_label
        if completion_cap_pairs:
            metrics["llm_completion_at_request_cap_count"] = completion_at_cap
            metrics["llm_completion_at_request_cap_rate"] = (
                completion_at_cap / completion_cap_pairs
            )
            sources["llm_completion_at_request_cap_count"] = source_label
            sources["llm_completion_at_request_cap_rate"] = source_label
        prompt_char_avg = None
        if prompt_char_counts:
            prompt_char_avg = sum(prompt_char_counts) / len(prompt_char_counts)
            metrics["llm_prompt_char_count_avg"] = prompt_char_avg
            metrics["llm_prompt_char_count_min"] = min(prompt_char_counts)
            metrics["llm_prompt_char_count_max"] = max(prompt_char_counts)
            sources["llm_prompt_char_count_avg"] = source_label
            sources["llm_prompt_char_count_min"] = source_label
            sources["llm_prompt_char_count_max"] = source_label
        if section_totals:
            metrics["llm_prompt_section_chars_total"] = dict(section_totals)
            section_avgs = {
                key: section_totals[key] / section_counts[key]
                for key in section_totals
                if section_counts.get(key)
            }
            metrics["llm_prompt_section_chars_avg"] = dict(section_avgs)
            metrics["llm_prompt_section_chars_max"] = dict(section_max)
            metrics["llm_prompt_section_entry_count"] = section_entry_count
            sources["llm_prompt_section_chars_total"] = source_label
            sources["llm_prompt_section_chars_avg"] = source_label
            sources["llm_prompt_section_chars_max"] = source_label
            sources["llm_prompt_section_entry_count"] = source_label
            if section_avgs:
                top_key, top_val = max(section_avgs.items(), key=lambda item: item[1])
                metrics["llm_prompt_section_top_avg_key"] = top_key
                metrics["llm_prompt_section_top_avg_chars"] = top_val
                sources["llm_prompt_section_top_avg_key"] = source_label
                sources["llm_prompt_section_top_avg_chars"] = source_label
                if prompt_char_avg:
                    metrics["llm_prompt_section_top_avg_ratio"] = top_val / prompt_char_avg
                    sources["llm_prompt_section_top_avg_ratio"] = source_label
                base_code_avg = section_avgs.get("base_code")
                if base_code_avg is not None and prompt_char_avg:
                    metrics["llm_prompt_section_base_code_ratio"] = base_code_avg / prompt_char_avg
                    sources["llm_prompt_section_base_code_ratio"] = source_label
                base_code_sections = {
                    key: value
                    for key, value in section_avgs.items()
                    if key.startswith("base_code_")
                }
                if base_code_sections:
                    base_key, base_val = max(
                        base_code_sections.items(), key=lambda item: item[1]
                    )
                    metrics["llm_prompt_section_top_base_code_key"] = base_key
                    metrics["llm_prompt_section_top_base_code_chars"] = base_val
                    sources["llm_prompt_section_top_base_code_key"] = source_label
                    sources["llm_prompt_section_top_base_code_chars"] = source_label
                    if base_code_avg:
                        metrics["llm_prompt_section_top_base_code_ratio"] = base_val / base_code_avg
                        sources["llm_prompt_section_top_base_code_ratio"] = source_label

    analysis_card_metrics, analysis_card_sources = _collect_analysis_card_signature_metrics(
        round_record
    )
    if analysis_card_metrics:
        metrics.update(analysis_card_metrics)
        sources.update(analysis_card_sources)

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

        judge_results = []
        judge_source = None
        raw_judge_results = mods_record.get("judge_results")
        if isinstance(raw_judge_results, list) and raw_judge_results:
            judge_results = [entry for entry in raw_judge_results if isinstance(entry, dict)]
            judge_source = "mechanism_modifications/judge_results"
        else:
            judge_entries = []
            if isinstance(attempts, list):
                for attempt in attempts:
                    if not isinstance(attempt, dict):
                        continue
                    judge = attempt.get("judge")
                    if isinstance(judge, dict):
                        judge_entries.append({
                            "approved": judge.get("approved"),
                            "reason": judge.get("reason"),
                        })
            if judge_entries:
                judge_results = judge_entries
                judge_source = "mechanism_modifications/attempts/judge"

        approved_judge = 0
        rejected_judge = 0
        missing_judge = 0
        reason_counts = Counter()
        for entry in judge_results:
            if not isinstance(entry, dict):
                missing_judge += 1
                continue
            approved_flag = entry.get("approved")
            if approved_flag is True:
                approved_judge += 1
            elif approved_flag is False:
                rejected_judge += 1
                reason = entry.get("reason")
                if reason:
                    reason_counts[str(reason).strip()] += 1
            else:
                missing_judge += 1
        if attempts_count and attempts_count > len(judge_results):
            missing_judge += attempts_count - len(judge_results)

        metrics["mechanism_judge_approved_count"] = approved_judge
        metrics["mechanism_judge_rejected_count"] = rejected_judge
        metrics["mechanism_judge_missing_count"] = missing_judge
        metrics["mechanism_judge_rejection_reason_counts"] = dict(reason_counts)
        judge_source = judge_source or "mechanism_modifications"
        sources["mechanism_judge_approved_count"] = judge_source
        sources["mechanism_judge_rejected_count"] = judge_source
        sources["mechanism_judge_missing_count"] = judge_source
        sources["mechanism_judge_rejection_reason_counts"] = judge_source

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

    errors_root = round_record.get("errors")
    agent_errors = []
    mechanism_errors = []
    if isinstance(errors_root, dict):
        agent_errors = errors_root.get("agent_code_errors") or []
        mechanism_errors = errors_root.get("mechanism_errors") or []
    if isinstance(agent_errors, list) or isinstance(mechanism_errors, list):
        syntax_stats = _collect_syntax_error_stats([agent_errors, mechanism_errors])
        syntax_llm_stats = _collect_syntax_error_llm_metadata_stats(
            [agent_errors, mechanism_errors]
        )
        metrics["llm_syntax_error_count"] = syntax_stats.get("total", 0)
        metrics["llm_syntax_error_near_end_count"] = syntax_stats.get("near_end", 0)
        metrics["llm_syntax_error_near_end_10pct_count"] = syntax_stats.get("near_end_10pct", 0)
        metrics["llm_syntax_error_mid_count"] = syntax_stats.get("mid", 0)
        metrics["llm_syntax_error_unknown_count"] = syntax_stats.get("unknown", 0)
        metrics["llm_syntax_error_line_ratio_avg"] = syntax_stats.get("line_ratio_avg", 0.0)
        metrics["llm_syntax_error_line_ratio_min"] = syntax_stats.get("line_ratio_min", 0.0)
        metrics["llm_syntax_error_line_ratio_max"] = syntax_stats.get("line_ratio_max", 0.0)
        metrics["llm_syntax_error_line_ratio_samples"] = syntax_stats.get("line_ratio_samples", 0)
        metrics["llm_syntax_error_js_comment_count"] = syntax_stats.get("js_comment", 0)
        metrics["llm_syntax_error_non_code_prefix_count"] = syntax_stats.get("non_code_prefix", 0)
        metrics["llm_syntax_error_missing_in_count"] = syntax_stats.get("missing_in", 0)
        metrics["llm_syntax_error_finish_reason_counts"] = syntax_llm_stats.get(
            "finish_reason_counts", {}
        )
        metrics["llm_syntax_error_finish_reason_total"] = syntax_llm_stats.get(
            "finish_reason_total", 0
        )
        metrics["llm_syntax_error_finish_reason_length_count"] = syntax_llm_stats.get(
            "finish_reason_length_count", 0
        )
        metrics["llm_syntax_error_finish_reason_missing_count"] = syntax_llm_stats.get(
            "finish_reason_missing", 0
        )
        metrics["llm_syntax_error_request_cap_count"] = syntax_llm_stats.get(
            "request_cap_count", 0
        )
        metrics["llm_syntax_error_completion_at_request_cap_count"] = syntax_llm_stats.get(
            "completion_at_cap_count", 0
        )
        metrics["llm_syntax_error_completion_at_request_cap_rate"] = syntax_llm_stats.get(
            "completion_at_cap_rate", 0.0
        )
        if "near_end_rate" in syntax_stats:
            metrics["llm_syntax_error_near_end_rate"] = syntax_stats["near_end_rate"]
        else:
            metrics["llm_syntax_error_near_end_rate"] = 0.0
        if "near_end_10pct_rate" in syntax_stats:
            metrics["llm_syntax_error_near_end_10pct_rate"] = syntax_stats["near_end_10pct_rate"]
        else:
            metrics["llm_syntax_error_near_end_10pct_rate"] = 0.0
        source_label = "errors/agent_code_errors+mechanism_errors"
        sources["llm_syntax_error_count"] = source_label
        sources["llm_syntax_error_near_end_count"] = source_label
        sources["llm_syntax_error_near_end_10pct_count"] = source_label
        sources["llm_syntax_error_mid_count"] = source_label
        sources["llm_syntax_error_unknown_count"] = source_label
        sources["llm_syntax_error_line_ratio_avg"] = source_label
        sources["llm_syntax_error_line_ratio_min"] = source_label
        sources["llm_syntax_error_line_ratio_max"] = source_label
        sources["llm_syntax_error_line_ratio_samples"] = source_label
        sources["llm_syntax_error_js_comment_count"] = source_label
        sources["llm_syntax_error_non_code_prefix_count"] = source_label
        sources["llm_syntax_error_missing_in_count"] = source_label
        sources["llm_syntax_error_finish_reason_counts"] = source_label
        sources["llm_syntax_error_finish_reason_total"] = source_label
        sources["llm_syntax_error_finish_reason_length_count"] = source_label
        sources["llm_syntax_error_finish_reason_missing_count"] = source_label
        sources["llm_syntax_error_request_cap_count"] = source_label
        sources["llm_syntax_error_completion_at_request_cap_count"] = source_label
        sources["llm_syntax_error_completion_at_request_cap_rate"] = source_label
        sources["llm_syntax_error_near_end_rate"] = source_label
        sources["llm_syntax_error_near_end_10pct_rate"] = source_label
        if isinstance(agent_errors, list):
            agent_stats = _collect_syntax_error_stats([agent_errors])
            _apply_syntax_error_subset(
                metrics,
                sources,
                agent_stats,
                "llm_syntax_error_agent",
                "errors/agent_code_errors",
            )
        if isinstance(mechanism_errors, list):
            mechanism_stats = _collect_syntax_error_stats([mechanism_errors])
            _apply_syntax_error_subset(
                metrics,
                sources,
                mechanism_stats,
                "llm_syntax_error_mechanism",
                "errors/mechanism_errors",
            )

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
