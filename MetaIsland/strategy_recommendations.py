from typing import Tuple
import math
import numpy as np
from collections import Counter, defaultdict


def build_strategy_recommendations(
    self,
    member_id: int,
    window: int = 8,
    min_samples: int = 2,
    exploration_bonus: float = 0.6,
    population_window_rounds: int = 3,
    max_items: int = 3
) -> str:
    """Provide lightweight decision support without forcing convergence."""
    _, memory = self._get_member_history(self.code_memory, member_id)
    if not memory:
        return "No strategy recommendations available."
    window = max(1, window)
    recent = memory[-window:]
    if not recent:
        return "No strategy recommendations available."
    member_count = None
    if hasattr(self, "current_members") and self.current_members is not None:
        try:
            member_count = len(self.current_members)
        except TypeError:
            member_count = None

    signature_counts = Counter()
    perf_by_sig = {}
    novelty_by_sig = {}
    cargo_by_sig = {}
    survival_by_sig = {}
    vitality_by_sig = {}
    drain_gap_cargo_by_sig = {}
    drain_gap_survival_by_sig = {}
    drain_gap_vitality_by_sig = {}
    for mem in recent:
        sig = mem.get('signature')
        if sig is None:
            sig = self._extract_action_signature(mem.get('code', ''))
        sig = tuple(sig) if sig else tuple()
        signature_counts[sig] += 1
        perf_by_sig.setdefault(sig, []).append(self._get_memory_performance(mem))
        metrics = mem.get("metrics") or {}
        cargo_delta = metrics.get("round_delta_cargo", metrics.get("delta_cargo"))
        if cargo_delta is not None:
            try:
                cargo_by_sig.setdefault(sig, []).append(float(cargo_delta))
            except (TypeError, ValueError):
                pass
        survival_delta = metrics.get("round_delta_survival", metrics.get("delta_survival"))
        if survival_delta is not None:
            try:
                survival_by_sig.setdefault(sig, []).append(float(survival_delta))
            except (TypeError, ValueError):
                pass
        vitality_delta = metrics.get("round_delta_vitality", metrics.get("delta_vitality"))
        if vitality_delta is not None:
            try:
                vitality_by_sig.setdefault(sig, []).append(float(vitality_delta))
            except (TypeError, ValueError):
                pass
        action_cargo = metrics.get("delta_cargo")
        round_cargo = metrics.get("round_delta_cargo")
        if action_cargo is not None and round_cargo is not None:
            try:
                gap = float(round_cargo) - float(action_cargo)
                drain_gap_cargo_by_sig.setdefault(sig, []).append(gap)
            except (TypeError, ValueError):
                pass
        action_survival = metrics.get("delta_survival")
        round_survival = metrics.get("round_delta_survival")
        if action_survival is not None and round_survival is not None:
            try:
                gap = float(round_survival) - float(action_survival)
                drain_gap_survival_by_sig.setdefault(sig, []).append(gap)
            except (TypeError, ValueError):
                pass
        action_vitality = metrics.get("delta_vitality")
        round_vitality = metrics.get("round_delta_vitality")
        if action_vitality is not None and round_vitality is not None:
            try:
                gap = float(round_vitality) - float(action_vitality)
                drain_gap_vitality_by_sig.setdefault(sig, []).append(gap)
            except (TypeError, ValueError):
                pass
        novelty = mem.get('signature_novelty')
        if novelty is not None:
            try:
                novelty_by_sig.setdefault(sig, []).append(float(novelty))
            except (TypeError, ValueError):
                pass
    total = len(recent)
    recent_unique = len(signature_counts)
    personal_diversity = recent_unique / total if total else 0.0
    member_recent_signatures = set(signature_counts.keys())
    current_tags = self._get_member_context_tags(member_id)
    current_key = self._context_key_from_tags(current_tags)
    context_stats = self._collect_context_weighted_signature_stats(
        recent,
        current_tags,
        min_similarity=0.35,
    )
    context_ranked = []
    context_top = None
    context_candidate = None
    if context_stats:
        context_ranked = sorted(
            [
                (
                    record.get("avg", 0.0),
                    record.get("weight_sum", 0.0),
                    record.get("count", 0),
                    record.get("similarity_avg", 0.0),
                    sig,
                )
                for sig, record in context_stats.items()
            ],
            key=lambda x: (x[0], x[1], x[2]),
            reverse=True,
        )
        if context_ranked:
            context_top = context_ranked[0]
            min_context_weight = max(0.6, 0.4 * float(min_samples))
            for avg, weight_sum, count, sim_avg, sig in context_ranked:
                if count >= min_samples or weight_sum >= min_context_weight:
                    context_candidate = (avg, weight_sum, count, sim_avg, sig)
                    break
    risk_budget = 0.7
    survival_tag = current_tags.get("survival") if current_tags else None
    relation_tag = current_tags.get("relations") if current_tags else None
    cargo_tag = current_tags.get("cargo") if current_tags else None
    if survival_tag == "fragile":
        risk_budget = 0.45
    elif survival_tag == "dominant":
        risk_budget = 0.95
    if relation_tag == "hostile":
        risk_budget -= 0.1
    elif relation_tag == "friendly":
        risk_budget += 0.05
    if cargo_tag == "cargo_low":
        risk_budget -= 0.1
    elif cargo_tag == "cargo_high":
        risk_budget += 0.05
    risk_budget = min(1.0, max(0.35, risk_budget))
    recent_perfs = [self._get_memory_performance(mem) for mem in recent]
    recent_perf_avg = 0.0
    recent_perf_abs_avg = 0.0
    performance_drag = None
    if recent_perfs:
        recent_perf_avg = sum(recent_perfs) / len(recent_perfs)
        recent_perf_abs_avg = sum(abs(perf) for perf in recent_perfs) / len(recent_perfs)
    if (
        len(recent_perfs) >= min_samples
        and recent_perf_abs_avg > 0.0
        and recent_perf_avg < 0.0
    ):
        drag_ratio = abs(recent_perf_avg) / recent_perf_abs_avg
        if drag_ratio >= 0.5:
            performance_drag = {
                "avg": recent_perf_avg,
                "abs_avg": recent_perf_abs_avg,
                "ratio": drag_ratio,
            }
            risk_budget = max(0.35, risk_budget - 0.1)
    production = self._get_latest_total(getattr(self, "record_total_production", 0.0))
    consumption = self._get_latest_total(getattr(self, "record_total_consumption", 0.0))
    net_production = production - consumption
    recent_round_cargo = []
    recent_action_cargo = []
    for mem in recent:
        metrics = mem.get("metrics") or {}
        if "round_delta_cargo" in metrics:
            try:
                recent_round_cargo.append(float(metrics["round_delta_cargo"]))
            except (TypeError, ValueError):
                pass
        if "delta_cargo" in metrics:
            try:
                recent_action_cargo.append(float(metrics["delta_cargo"]))
            except (TypeError, ValueError):
                pass
    cargo_samples = recent_round_cargo or recent_action_cargo
    cargo_drag = None
    if cargo_samples:
        cargo_avg = sum(cargo_samples) / len(cargo_samples)
        cargo_abs_avg = sum(abs(val) for val in cargo_samples) / len(cargo_samples)
        if cargo_abs_avg > 0.0 and cargo_avg < 0.0:
            cargo_ratio = abs(cargo_avg) / cargo_abs_avg
            if cargo_ratio >= 0.5:
                cargo_drag = {
                    "avg": cargo_avg,
                    "abs_avg": cargo_abs_avg,
                    "ratio": cargo_ratio,
                }
                risk_budget = max(0.35, risk_budget - 0.05)
    resource_pressure = False
    if net_production < 0.0:
        resource_pressure = True
    if cargo_drag is not None:
        resource_pressure = True
    round_end_metrics = self._get_latest_round_end_metrics()
    round_end_survival = None
    round_end_cargo = None
    round_end_vitality = None
    round_end_member_count = None
    round_end_pressure = False
    round_end_small_sample = False
    round_end_notes = []
    round_end_severity = 0.0
    round_end_loss_axes = 0
    round_end_loss_weight = 1.0
    critical_loss_axes = 0
    critical_pressure = False
    critical_loss_weight = 1.0
    round_end_survival_negative = False
    round_end_cargo_negative = False
    round_end_vitality_negative = False
    round_metrics = None
    action_survival = None
    action_cargo = None
    action_vitality = None
    drain_notes = []
    drain_pressure = False
    drain_severity = 0.0
    drain_loss_axes = 0
    drain_loss_weight = 1.0
    drain_cargo = False

    def _to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    if round_end_metrics:
        round_end_survival = _to_float(
            round_end_metrics.get("round_end_population_avg_survival_delta")
        )
        round_end_cargo = _to_float(
            round_end_metrics.get("round_end_population_avg_cargo_delta")
        )
        round_end_vitality = _to_float(
            round_end_metrics.get("round_end_population_avg_vitality_delta")
        )
        round_end_member_count = round_end_metrics.get("round_end_member_count")
        try:
            round_end_member_count = int(round_end_member_count)
        except (TypeError, ValueError):
            round_end_member_count = None
        if member_count is None or member_count == 0:
            member_count = round_end_member_count
        if round_end_member_count is not None and round_end_member_count < min_samples:
            round_end_small_sample = True
        if round_end_survival is not None and round_end_survival < 0.0:
            round_end_survival_negative = True
            round_end_notes.append(f"survival {round_end_survival:.2f}")
            if not round_end_small_sample:
                round_end_pressure = True
                round_end_loss_axes += 1
            if round_end_survival <= -30.0:
                critical_loss_axes += 1
        if round_end_cargo is not None and round_end_cargo < 0.0:
            round_end_cargo_negative = True
            round_end_notes.append(f"cargo {round_end_cargo:.2f}")
            if not round_end_small_sample:
                round_end_pressure = True
                round_end_loss_axes += 1
                resource_pressure = True
            if round_end_cargo <= -30.0:
                critical_loss_axes += 1
        if round_end_vitality is not None and round_end_vitality < 0.0:
            round_end_vitality_negative = True
            round_end_notes.append(f"vitality {round_end_vitality:.2f}")
            if not round_end_small_sample:
                round_end_pressure = True
                round_end_loss_axes += 1
            if round_end_vitality <= -20.0:
                critical_loss_axes += 1
    round_metrics = self._get_latest_round_metrics()
    if round_metrics:
        action_survival = _to_float(round_metrics.get("population_avg_survival_delta"))
        action_cargo = _to_float(round_metrics.get("population_avg_cargo_delta"))
        action_vitality = _to_float(round_metrics.get("population_avg_vitality_delta"))

    def _drain_gap(action_val, end_val, threshold):
        if action_val is None or end_val is None:
            return None
        gap = end_val - action_val
        if end_val < 0.0 and gap <= -threshold:
            return gap
        return None

    def _drain_severity(gap, scale):
        if gap is None:
            return 0.0
        return min(1.0, abs(gap) / max(1.0, scale * 2.0))

    if not round_end_small_sample:
        gap = _drain_gap(action_survival, round_end_survival, 6.0)
        if gap is not None:
            drain_notes.append(
                f"survival {action_survival:.2f}->{round_end_survival:.2f}"
            )
            drain_loss_axes += 1
            drain_severity = max(drain_severity, _drain_severity(gap, 6.0))
        gap = _drain_gap(action_cargo, round_end_cargo, 15.0)
        if gap is not None:
            drain_notes.append(
                f"cargo {action_cargo:.2f}->{round_end_cargo:.2f}"
            )
            drain_loss_axes += 1
            drain_severity = max(drain_severity, _drain_severity(gap, 15.0))
            drain_cargo = True
        gap = _drain_gap(action_vitality, round_end_vitality, 12.0)
        if gap is not None:
            drain_notes.append(
                f"vitality {action_vitality:.2f}->{round_end_vitality:.2f}"
            )
            drain_loss_axes += 1
            drain_severity = max(drain_severity, _drain_severity(gap, 12.0))
    if drain_loss_axes:
        drain_pressure = True
        if drain_loss_axes > 1:
            drain_loss_weight = 1.0 + 0.15 * (drain_loss_axes - 1)
        if drain_cargo:
            resource_pressure = True
    if round_end_pressure and round_end_loss_axes > 1:
        round_end_loss_weight = 1.0 + 0.2 * (round_end_loss_axes - 1)
    if round_end_pressure and not round_end_small_sample and critical_loss_axes:
        critical_pressure = True
        critical_loss_weight = 1.0 + 0.15 * max(0, critical_loss_axes - 1)
    if round_end_pressure and round_end_survival_negative:
        risk_delta = 0.08 if round_end_survival is not None and round_end_survival <= -5.0 else 0.04
        risk_budget = max(0.35, risk_budget - risk_delta * round_end_loss_weight)
    if round_end_pressure and round_end_cargo_negative:
        risk_delta = 0.05 if round_end_cargo is not None and round_end_cargo <= -5.0 else 0.03
        risk_budget = max(0.35, risk_budget - risk_delta * round_end_loss_weight)
    if round_end_pressure and round_end_vitality_negative:
        risk_delta = 0.04 if round_end_vitality is not None and round_end_vitality <= -10.0 else 0.02
        if round_end_survival_negative:
            risk_delta *= 0.5
        risk_budget = max(0.35, risk_budget - risk_delta * round_end_loss_weight)
    if round_end_survival is not None and round_end_survival < 0.0:
        round_end_severity = max(
            round_end_severity,
            min(1.0, abs(round_end_survival) / 12.0),
        )
    if round_end_cargo is not None and round_end_cargo < 0.0:
        round_end_severity = max(
            round_end_severity,
            min(1.0, abs(round_end_cargo) / 25.0),
        )
    if round_end_vitality is not None and round_end_vitality < 0.0:
        round_end_severity = max(
            round_end_severity,
            min(1.0, abs(round_end_vitality) / 30.0),
        )
    if round_end_pressure and round_end_severity >= 0.7:
        risk_budget = max(
            0.35,
            risk_budget - 0.04 * round_end_severity * round_end_loss_weight
        )
    if critical_pressure:
        risk_budget = max(0.35, risk_budget - 0.05 * critical_loss_weight)
    if drain_pressure:
        risk_budget = max(
            0.35,
            risk_budget - 0.04 * drain_severity * drain_loss_weight
        )
    if risk_budget >= 0.85:
        risk_label = "high"
    elif risk_budget <= 0.55:
        risk_label = "low"
    else:
        risk_label = "medium"
    sig_cargo_avg = {
        sig: sum(values) / len(values) for sig, values in cargo_by_sig.items() if values
    }
    sig_survival_avg = {
        sig: sum(values) / len(values) for sig, values in survival_by_sig.items() if values
    }
    sig_vitality_avg = {
        sig: sum(values) / len(values) for sig, values in vitality_by_sig.items() if values
    }
    sig_drain_gap_cargo_avg = {
        sig: sum(values) / len(values)
        for sig, values in drain_gap_cargo_by_sig.items()
        if values
    }
    sig_drain_gap_survival_avg = {
        sig: sum(values) / len(values)
        for sig, values in drain_gap_survival_by_sig.items()
        if values
    }
    sig_drain_gap_vitality_avg = {
        sig: sum(values) / len(values)
        for sig, values in drain_gap_vitality_by_sig.items()
        if values
    }
    drain_sig_available = bool(sig_drain_gap_cargo_avg or sig_drain_gap_survival_avg or sig_drain_gap_vitality_avg)
    def _cargo_adjustment(
        sig: tuple,
        base_scale_neg: float,
        base_scale_pos: float,
        cap: float,
    ) -> float:
        if not sig_cargo_avg:
            return 0.0
        if not (resource_pressure or round_end_pressure):
            return 0.0
        cargo_avg = sig_cargo_avg.get(sig)
        if cargo_avg is None or cargo_avg == 0.0:
            return 0.0
        base_scale = base_scale_neg if cargo_avg < 0.0 else base_scale_pos
        scale = base_scale
        if round_end_pressure:
            scale *= (1.0 + round_end_severity) * round_end_loss_weight
            if critical_pressure:
                scale *= (1.0 + 0.3 * critical_loss_weight)
            scale = min(cap, scale)
        return math.copysign(
            min(scale, abs(cargo_avg) / 20.0 * scale), cargo_avg
        )

    def _bounded_ratio(val, denom):
        if val is None:
            return 0.0
        try:
            ratio = float(val) / float(denom)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
        return max(-1.0, min(1.0, ratio))

    def _drain_gap_adjustment(
        sig: tuple,
        gap_map: dict,
        base_scale_neg: float,
        base_scale_pos: float,
        cap: float,
        denom: float,
    ) -> float:
        if not drain_pressure or not gap_map:
            return 0.0
        gap = gap_map.get(sig)
        if gap is None or gap == 0.0:
            return 0.0
        base_scale = base_scale_neg if gap < 0.0 else base_scale_pos
        scale = base_scale * (1.0 + drain_severity) * drain_loss_weight
        if critical_pressure:
            scale *= (1.0 + 0.2 * critical_loss_weight)
        scale = min(cap, scale)
        return math.copysign(
            min(scale, abs(gap) / max(1.0, denom) * scale), gap
        )

    def _vitality_adjustment(
        sig: tuple,
        base_scale_neg: float,
        base_scale_pos: float,
        cap: float,
    ) -> float:
        if not sig_vitality_avg:
            return 0.0
        if not round_end_pressure:
            return 0.0
        vitality_avg = sig_vitality_avg.get(sig)
        if vitality_avg is None or vitality_avg == 0.0:
            return 0.0
        base_scale = base_scale_neg if vitality_avg < 0.0 else base_scale_pos
        scale = base_scale * (1.0 + round_end_severity) * round_end_loss_weight
        if critical_pressure:
            scale *= (1.0 + 0.3 * critical_loss_weight)
        scale = min(cap, scale)
        return math.copysign(
            min(scale, abs(vitality_avg) / 30.0 * scale), vitality_avg
        )

    def _survival_adjustment(
        sig: tuple,
        base_scale_neg: float,
        base_scale_pos: float,
        cap: float,
    ) -> float:
        if not sig_survival_avg:
            return 0.0
        if not round_end_pressure:
            return 0.0
        survival_avg = sig_survival_avg.get(sig)
        if survival_avg is None or survival_avg == 0.0:
            return 0.0
        base_scale = base_scale_neg if survival_avg < 0.0 else base_scale_pos
        scale = base_scale * (1.0 + round_end_severity) * round_end_loss_weight
        if critical_pressure:
            scale *= (1.0 + 0.3 * critical_loss_weight)
        scale = min(cap, scale)
        return math.copysign(
            min(scale, abs(survival_avg) / 12.0 * scale), survival_avg
        )

    def _recovery_score(sig: tuple) -> float:
        score = 0.0
        cargo_avg = sig_cargo_avg.get(sig)
        if cargo_avg is not None:
            score += 0.55 * _bounded_ratio(cargo_avg, 25.0)
        survival_avg = sig_survival_avg.get(sig)
        if survival_avg is not None:
            score += 0.3 * _bounded_ratio(survival_avg, 12.0)
        vitality_avg = sig_vitality_avg.get(sig)
        if vitality_avg is not None:
            score += 0.15 * _bounded_ratio(vitality_avg, 30.0)
        return score

    recovery_bias = 0.0
    if round_end_pressure and round_end_severity >= 0.7:
        recovery_bias = 0.08 * round_end_severity * round_end_loss_weight
    if critical_pressure:
        recovery_bias += 0.05 * critical_loss_weight
    if drain_pressure:
        drain_bias = min(0.06, 0.04 * drain_severity * drain_loss_weight)
        recovery_bias += drain_bias

    recent_context_keys = []
    missing_context = 0
    for mem in recent:
        context_key = self._get_memory_context_key(mem)
        if context_key:
            recent_context_keys.append(context_key)
        else:
            missing_context += 1
    context_counts = Counter(recent_context_keys)
    context_total = len(recent_context_keys)
    context_unique = len(context_counts)
    dominant_context = None
    dominant_context_share = 0.0
    if context_counts:
        dominant_context, dominant_count = context_counts.most_common(1)[0]
        dominant_context_share = dominant_count / context_total if context_total else 0.0
    avg_sorted = sorted(
        [(sum(perfs) / len(perfs), len(perfs), sig) for sig, perfs in perf_by_sig.items()],
        key=lambda x: (x[0], x[1]),
        reverse=True
    )
    best_by_avg = None
    for avg, count, sig in avg_sorted:
        if count >= min_samples:
            best_by_avg = (avg, count, sig)
            break
    if best_by_avg is None and avg_sorted:
        best_by_avg = avg_sorted[0]
    sig_perf_stats = []
    for sig, perfs in perf_by_sig.items():
        count = len(perfs)
        avg = sum(perfs) / count if count else 0.0
        std = float(np.std(perfs)) if count > 1 else 0.0
        novelty_vals = novelty_by_sig.get(sig)
        novelty_avg = (
            sum(novelty_vals) / len(novelty_vals)
            if novelty_vals else 0.0
        )
        sig_perf_stats.append((avg, std, count, novelty_avg, sig))
    sig_perf_lookup = {
        sig: (avg, std, count, novelty_avg)
        for avg, std, count, novelty_avg, sig in sig_perf_stats
    }

    std_cutoff = 0.0
    if sig_perf_stats:
        std_values = sorted(stat[1] for stat in sig_perf_stats)
        std_cutoff = std_values[len(std_values) // 2]
    baseline_candidate = None
    baseline_reason = "stable"
    def _baseline_rank(stat):
        avg, std, count, _novelty_avg, sig = stat
        score = avg
        if recovery_bias:
            score += recovery_bias * _recovery_score(sig)
        score += _cargo_adjustment(
            sig,
            base_scale_neg=0.05,
            base_scale_pos=0.03,
            cap=0.12,
        )
        score += _survival_adjustment(
            sig,
            base_scale_neg=0.05,
            base_scale_pos=0.03,
            cap=0.12,
        )
        score += _vitality_adjustment(
            sig,
            base_scale_neg=0.04,
            base_scale_pos=0.03,
            cap=0.1,
        )
        score += _drain_gap_adjustment(sig, sig_drain_gap_cargo_avg, base_scale_neg=0.05, base_scale_pos=0.03, cap=0.12, denom=15.0)
        score += _drain_gap_adjustment(sig, sig_drain_gap_survival_avg, base_scale_neg=0.05, base_scale_pos=0.03, cap=0.1, denom=6.0)
        score += _drain_gap_adjustment(sig, sig_drain_gap_vitality_avg, base_scale_neg=0.04, base_scale_pos=0.03, cap=0.1, denom=12.0)
        return (score, -std, count)

    stable_candidates = [
        stat for stat in sig_perf_stats
        if stat[2] >= min_samples and stat[1] <= std_cutoff
    ]
    if stable_candidates:
        baseline_candidate = max(
            stable_candidates,
            key=_baseline_rank
        )
    elif sig_perf_stats:
        baseline_candidate = max(
            sig_perf_stats,
            key=_baseline_rank
        )
        baseline_reason = "average"
    baseline_sig = None
    if baseline_candidate:
        baseline_sig = baseline_candidate[-1]
    elif best_by_avg is not None:
        baseline_sig = best_by_avg[2]
        baseline_reason = "average"
    if context_candidate is not None:
        ctx_avg, _ctx_weight, _ctx_count, _ctx_sim, ctx_sig = context_candidate
        baseline_avg = None
        if baseline_sig is not None:
            baseline_avg = sig_perf_lookup.get(baseline_sig, (None, None, None, None))[0]
        if baseline_sig is None or baseline_avg is None or ctx_avg >= (baseline_avg - 0.02):
            baseline_sig = ctx_sig
            baseline_reason = "context"
    dominant_sig = None
    dominant_share = 0.0
    if signature_counts:
        dominant_sig, dominant_count = signature_counts.most_common(1)[0]
        dominant_share = dominant_count / total if total else 0.0
    negative = sorted(
        [(avg, count, sig) for avg, count, sig in avg_sorted if count >= min_samples and avg < 0.0],
        key=lambda x: x[0]
    )

    member_tag_counts = Counter()
    for mem in recent:
        sig = mem.get('signature')
        if sig is None:
            sig = self._extract_action_signature(mem.get('code', ''))
        if sig:
            member_tag_counts.update(sig)
    member_signature_shares = []
    dominant_share_hits = 0
    for mem in recent:
        metrics = mem.get('metrics', {}) or {}
        share = metrics.get('round_signature_share')
        if share is None:
            continue
        try:
            share_val = float(share)
        except (TypeError, ValueError):
            continue
        member_signature_shares.append(share_val)
        if share_val >= 0.5:
            dominant_share_hits += 1
    known_tags = [
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
    member_underuse_threshold = max(1, int(0.2 * total))
    pop_entries = []
    min_round = None
    current_round = len(self.execution_history.get('rounds', []))
    if current_round:
        population_window_rounds = max(1, population_window_rounds)
        min_round = max(1, current_round - population_window_rounds + 1)
        for memory_list in self.code_memory.values():
            for mem in memory_list:
                round_num = mem.get('context', {}).get('round')
                if round_num is None or round_num < min_round:
                    continue
                sig = mem.get('signature')
                if sig is None:
                    sig = self._extract_action_signature(mem.get('code', ''))
                sig = tuple(sig) if sig else tuple()
                perf = self._get_memory_performance(mem)
                pop_entries.append((sig, perf))
    pop_signature_counts = Counter(sig for sig, _ in pop_entries)
    pop_total = len(pop_entries)
    pop_unique = len(pop_signature_counts)
    pop_dominant_sig = None
    pop_dominant_share = 0.0
    if pop_signature_counts and pop_total:
        pop_dominant_sig, pop_dominant_count = pop_signature_counts.most_common(1)[0]
        pop_dominant_share = pop_dominant_count / pop_total
    pop_diversity_ratio = pop_unique / pop_total if pop_total else 0.0
    pop_entropy = 0.0
    pop_entropy_norm = 0.0
    if pop_signature_counts and pop_total:
        for count in pop_signature_counts.values():
            prob = count / pop_total
            if prob > 0:
                pop_entropy -= prob * math.log(prob)
        max_entropy = math.log(pop_unique) if pop_unique > 1 else 0.0
        pop_entropy_norm = pop_entropy / max_entropy if max_entropy > 0 else 0.0

    diversity_emergency = False
    if pop_total >= min_samples:
        if (
            pop_dominant_share >= 0.85
            or pop_diversity_ratio <= 0.4
            or (pop_entropy_norm is not None and pop_entropy_norm <= 0.2)
        ):
            diversity_emergency = True

    diversity_rescue = False
    if (
        round_end_pressure
        and round_end_severity >= 0.7
        and pop_total >= min_samples
        and pop_dominant_share >= 0.6
        and pop_diversity_ratio <= 0.7
    ):
        diversity_rescue = True

    diversity_override = None
    diversity_risk_floor = 0.55 if performance_drag is None else 0.6
    if diversity_emergency:
        diversity_risk_floor = max(0.5, diversity_risk_floor - 0.05)
    if (
        baseline_sig is not None
        and pop_total
        and pop_dominant_sig is not None
        and baseline_sig == pop_dominant_sig
        and pop_dominant_share >= 0.6
        and risk_budget >= diversity_risk_floor
    ):
        base_stats = sig_perf_lookup.get(baseline_sig)
        baseline_avg = base_stats[0] if base_stats else None
        if baseline_avg is not None:
            perf_floor = baseline_avg - 0.03
            if baseline_avg < 0.0:
                perf_floor = baseline_avg
        else:
            perf_floor = None
        max_share = max(0.45, pop_dominant_share - 0.15)
        candidates = []
        for avg, _std, count, _novelty_avg, sig in sig_perf_stats:
            if sig == baseline_sig:
                continue
            if count < min_samples:
                continue
            share = pop_signature_counts.get(sig, 0) / pop_total if pop_total else 0.0
            if share >= max_share:
                continue
            if perf_floor is not None and avg < perf_floor:
                continue
            candidates.append((avg, -share, count, sig))
        if candidates:
            candidates.sort(reverse=True)
            best_avg, _neg_share, best_count, best_sig = candidates[0]
            diversity_override = {
                "previous": baseline_sig,
                "previous_share": pop_dominant_share,
                "candidate_share": pop_signature_counts.get(best_sig, 0) / pop_total if pop_total else 0.0,
                "baseline_avg": baseline_avg,
                "candidate_avg": best_avg,
            }
            baseline_sig = best_sig
            baseline_reason = "diversity"

    niche_candidate = None
    if pop_entries and pop_total:
        pop_sig_perf = defaultdict(list)
        for sig, perf in pop_entries:
            pop_sig_perf[sig].append(perf)
        pop_sig_stats = []
        for sig, perfs in pop_sig_perf.items():
            count = len(perfs)
            if count == 0:
                continue
            avg = sum(perfs) / count
            share = count / pop_total
            pop_sig_stats.append((avg, share, count, sig))
        pop_sig_stats.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        for avg, share, count, sig in pop_sig_stats:
            if not sig:
                continue
            if sig in member_recent_signatures:
                continue
            if count >= min_samples and avg > 0.0 and share <= 0.35:
                niche_candidate = (avg, share, count, sig)
                break

    base_exploration_bonus = exploration_bonus
    exploration_pressure = 0.0
    pressure_reasons = []
    if pop_total >= min_samples * 2 and (
        pop_dominant_share >= 0.6 or pop_diversity_ratio < 0.3
    ):
        exploration_pressure += 0.2
        pressure_reasons.append("population convergence")
    if diversity_emergency:
        exploration_pressure += 0.08
        pressure_reasons.append("dominance emergency")
    if total >= 3 and personal_diversity < 0.4:
        exploration_pressure += 0.2
        pressure_reasons.append("personal repetition")
    if context_counts and context_unique > 1 and dominant_context_share >= 0.7:
        exploration_pressure += 0.1
        pressure_reasons.append("context concentration")
    if current_key and context_counts and current_key not in context_counts:
        exploration_pressure += 0.15
        pressure_reasons.append("novel context")
    avg_signature_share = None
    if member_signature_shares:
        avg_signature_share = sum(member_signature_shares) / len(member_signature_shares)
        dominant_ratio = dominant_share_hits / len(member_signature_shares)
        if dominant_ratio >= 0.5 and avg_signature_share >= 0.45:
            exploration_pressure += 0.1
            pressure_reasons.append("own signature dominance")
    if performance_drag is not None:
        exploration_pressure -= 0.05
        pressure_reasons.append("recent losses")
    if resource_pressure:
        exploration_pressure -= 0.05
        pressure_reasons.append("resource pressure")
    if round_end_pressure:
        extra_pull = 0.05 * round_end_severity if round_end_severity else 0.0
        if round_end_loss_axes > 1:
            extra_pull += 0.02 * round_end_loss_axes
        loss_pull = 0.05 + extra_pull
        if diversity_rescue:
            loss_pull *= 0.6
        if diversity_emergency:
            loss_pull *= 0.7
        exploration_pressure -= loss_pull
        reason = "round-end losses"
        if round_end_severity:
            reason += f" (severity {round_end_severity:.2f})"
        if round_end_loss_axes > 1:
            reason += f", deficits {round_end_loss_axes}"
        if critical_pressure:
            reason += ", critical"
        if diversity_rescue:
            reason += ", diversity rescue"
        if diversity_emergency:
            reason += ", dominance emergency"
        pressure_reasons.append(reason)
    if pop_total:
        diversity_adjustment, diversity_error = self._update_diversity_controller(
            pop_diversity_ratio,
            pop_entropy_norm,
        )
        if abs(diversity_adjustment) > 1e-6:
            exploration_pressure += diversity_adjustment
            alpha = float(self._diversity_controller.get("alpha", 0.0))
            pressure_reasons.append(
                f"diversity α={alpha:.2f} err={diversity_error:.2f}"
            )
    exploration_bonus = max(0.25, base_exploration_bonus * risk_budget) + exploration_pressure
    safety_bias = 0.0
    if round_end_pressure and round_end_severity >= 0.7:
        safety_bias = 0.12 * round_end_severity * round_end_loss_weight
        if critical_pressure:
            safety_bias += 0.05 * critical_loss_weight
    stabilization_bias = 0.0
    if critical_pressure:
        stabilization_bias = 0.06 * critical_loss_weight
    high_risk_tags = {"attack", "expand"}
    if resource_pressure or round_end_pressure:
        high_risk_tags.add("bear")

    def _blend_with_context(avg_value: float, sig: tuple) -> Tuple[float, float]:
        info = context_stats.get(sig) if context_stats else None
        if not info:
            return avg_value, 0.0
        weight_sum = float(info.get("weight_sum", 0.0))
        if weight_sum <= 0.0:
            return avg_value, 0.0
        context_avg = float(info.get("avg", avg_value))
        weight = min(0.35, weight_sum / (weight_sum + 1.5))
        blended = avg_value * (1.0 - weight) + context_avg * weight
        return blended, weight

    sig_stats = []
    for sig, perfs in perf_by_sig.items():
        count = len(perfs)
        avg = sum(perfs) / count if count else 0.0
        blended_avg, ctx_weight = _blend_with_context(avg, sig)
        ucb = blended_avg + exploration_bonus * math.sqrt(math.log(total + 1) / count)
        sig_stats.append((ucb, blended_avg, avg, count, sig, ctx_weight))
    sig_stats.sort(key=lambda x: x[0], reverse=True)

    variation_candidate = None
    if sig_stats:
        scored = []
        for ucb, blended_avg, avg, count, sig, ctx_weight in sig_stats:
            if baseline_sig is not None and sig == baseline_sig:
                continue
            if baseline_sig:
                overlap = self._signature_overlap(baseline_sig, sig)
                distance_penalty = (1.0 - risk_budget) * (1.0 - overlap)
            else:
                overlap = 0.0
                distance_penalty = 0.0
            is_high_risk = bool(sig and any(tag in high_risk_tags for tag in sig))
            if diversity_emergency and overlap >= 0.5 and not is_high_risk:
                distance_penalty *= 0.6
            dominance_penalty = 0.0
            if pop_dominant_sig is not None and pop_dominant_share >= 0.6:
                if sig == pop_dominant_sig:
                    dominance_penalty = 0.15 + 0.15 * pop_dominant_share
            safety_penalty = 0.0
            if safety_bias and is_high_risk:
                safety_penalty = safety_bias
            resource_adjustment = 0.0
            if sig_cargo_avg and (resource_pressure or round_end_pressure):
                cargo_avg = sig_cargo_avg.get(sig)
                if cargo_avg:
                    base_scale = 0.06 if cargo_avg < 0.0 else 0.04
                    scale = base_scale
                    if round_end_pressure:
                        scale *= (1.0 + round_end_severity) * round_end_loss_weight
                        if critical_pressure:
                            scale *= (1.0 + 0.3 * critical_loss_weight)
                        scale = min(0.18, scale)
                    resource_adjustment = math.copysign(
                        min(scale, abs(cargo_avg) / 20.0 * scale), cargo_avg
                    )
            survival_adjustment = 0.0
            if sig_survival_avg and round_end_pressure:
                survival_avg = sig_survival_avg.get(sig)
                if survival_avg is not None:
                    base_scale = 0.08 if survival_avg < 0.0 else 0.05
                    scale = base_scale * (1.0 + round_end_severity) * round_end_loss_weight
                    if critical_pressure:
                        scale *= (1.0 + 0.3 * critical_loss_weight)
                    scale = min(0.2, scale)
                    survival_adjustment = math.copysign(
                        min(scale, abs(survival_avg) / 12.0 * scale), survival_avg
                    )
            vitality_adjustment = 0.0
            if sig_vitality_avg and round_end_pressure:
                vitality_avg = sig_vitality_avg.get(sig)
                if vitality_avg is not None:
                    base_scale = 0.06 if vitality_avg < 0.0 else 0.04
                    scale = base_scale * (1.0 + round_end_severity) * round_end_loss_weight
                    if critical_pressure:
                        scale *= (1.0 + 0.3 * critical_loss_weight)
                    scale = min(0.16, scale)
                    vitality_adjustment = math.copysign(
                        min(scale, abs(vitality_avg) / 30.0 * scale), vitality_avg
                    )
            drain_adjustment = 0.0
            if drain_pressure:
                drain_adjustment += _drain_gap_adjustment(
                    sig,
                    sig_drain_gap_cargo_avg,
                    base_scale_neg=0.06,
                    base_scale_pos=0.04,
                    cap=0.15,
                    denom=15.0,
                )
                drain_adjustment += _drain_gap_adjustment(
                    sig,
                    sig_drain_gap_survival_avg,
                    base_scale_neg=0.06,
                    base_scale_pos=0.04,
                    cap=0.12,
                    denom=6.0,
                )
                drain_adjustment += _drain_gap_adjustment(
                    sig,
                    sig_drain_gap_vitality_avg,
                    base_scale_neg=0.05,
                    base_scale_pos=0.04,
                    cap=0.12,
                    denom=12.0,
                )
            recovery_adjustment = 0.0
            if recovery_bias:
                recovery_score = _recovery_score(sig)
                if abs(recovery_score) > 1e-6:
                    recovery_adjustment = recovery_bias * recovery_score
            stabilization_penalty = 0.0
            if stabilization_bias:
                negative_axes = sum(
                    1
                    for val in (resource_adjustment, survival_adjustment, vitality_adjustment)
                    if val < -1e-6
                )
                if negative_axes:
                    stabilization_penalty = stabilization_bias * negative_axes
            diversity_rescue_bonus = 0.0
            if diversity_rescue and baseline_sig is not None:
                rescue_score = _recovery_score(sig)
                safe_overlap = overlap >= 0.5
                non_negative = blended_avg >= 0.0
                if safe_overlap and (rescue_score > 0.0 or non_negative):
                    diversity_rescue_bonus = 0.04 + 0.04 * max(0.0, rescue_score)
                    if safety_penalty > 0.0:
                        diversity_rescue_bonus *= 0.5
            diversity_emergency_bonus = 0.0
            if diversity_emergency and baseline_sig is not None and sig != baseline_sig:
                if overlap >= 0.5 and not is_high_risk:
                    diversity_emergency_bonus = 0.03 + 0.02 * overlap
            adjusted = (
                ucb
                - distance_penalty
                - dominance_penalty
                - safety_penalty
                - stabilization_penalty
                + resource_adjustment
                + survival_adjustment
                + vitality_adjustment
                + drain_adjustment
                + recovery_adjustment
                + diversity_rescue_bonus
                + diversity_emergency_bonus
            )
            scored.append(
                (
                    adjusted,
                    ucb,
                    blended_avg,
                    avg,
                    count,
                    sig,
                    overlap,
                    dominance_penalty,
                    safety_penalty,
                    stabilization_penalty,
                    resource_adjustment,
                    survival_adjustment,
                    vitality_adjustment,
                    drain_adjustment,
                    recovery_adjustment,
                    diversity_rescue_bonus,
                    diversity_emergency_bonus,
                    ctx_weight,
                )
            )
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            variation_candidate = scored[0]

    novelty_candidate = None
    if sig_perf_stats:
        novelty_sorted = sorted(
            sig_perf_stats,
            key=lambda x: (x[3], x[0], -x[1]),
            reverse=True
        )
        for avg, std, count, novelty_avg, sig in novelty_sorted:
            if sig != baseline_sig and novelty_avg > 0.0:
                novelty_candidate = (avg, std, count, novelty_avg, sig)
                break

    tag_perf = {tag: [] for tag in known_tags}
    for sig, perf in pop_entries:
        for tag in sig:
            if tag not in tag_perf:
                tag_perf[tag] = []
            tag_perf[tag].append(perf)

    pop_stats = []
    for tag in known_tags:
        perfs = tag_perf.get(tag, [])
        count = len(perfs)
        avg = sum(perfs) / count if count else 0.0
        pop_stats.append((avg, count, tag))

    promising_tags = sorted(
        [
            (avg, count, tag)
            for avg, count, tag in pop_stats
            if count >= min_samples
            and avg > 0.0
            and member_tag_counts.get(tag, 0) <= member_underuse_threshold
        ],
        reverse=True
    )
    caution_tags = sorted(
        [
            (avg, count, tag)
            for avg, count, tag in pop_stats
            if count >= min_samples and avg < 0.0
        ]
    )

    fallback_variation = None
    if (
        variation_candidate is None
        and baseline_sig is not None
        and (diversity_rescue or diversity_emergency)
    ):
        baseline_tags = list(baseline_sig)
        candidate_tags = [
            tag
            for avg, count, tag in promising_tags
            if tag not in baseline_sig and tag not in high_risk_tags
        ]
        source = "population"
        if not candidate_tags:
            candidate_tags = [
                tag
                for tag in known_tags
                if tag not in baseline_sig and tag not in high_risk_tags
            ]
            source = "low-risk"
        if candidate_tags:
            seed_tag = candidate_tags[0]
            seed_sig = self._normalize_action_tags(baseline_tags + [seed_tag])
            if seed_sig and seed_sig != baseline_sig:
                fallback_variation = {
                    "signature": seed_sig,
                    "seed_tag": seed_tag,
                    "source": source,
                }

    def _format_tag_list(stats):
        if not stats:
            return "none"
        return ", ".join(
            f"{tag} (avg {avg:.2f}, n={count})"
            for avg, count, tag in stats[:max_items]
        )

    lines = [
        "Strategy recommendation (decision support):",
        f"- Recent window: {total} actions",
    ]
    lines.append(
        f"- Risk posture: {risk_label} (budget {risk_budget:.2f})"
        + (f"; context {self._format_context_tags(current_tags)}" if current_tags else "")
    )
    if performance_drag is not None:
        lines.append(
            f"- Recent performance drag: avg {recent_perf_avg:.2f}, "
            f"abs avg {recent_perf_abs_avg:.2f}"
        )
    if net_production < 0.0 or cargo_drag is not None:
        details = []
        if net_production < 0.0:
            per_member_net = None
            if member_count:
                per_member_net = net_production / float(member_count)
            details.append(
                f"net production {net_production:.1f} "
                f"(prod {production:.1f}, cons {consumption:.1f}"
                + (f", per member {per_member_net:.1f}" if per_member_net is not None else "")
                + ")"
            )
        if cargo_drag is not None:
            details.append(
                f"cargo drag avg {cargo_drag['avg']:.2f} "
                f"(abs {cargo_drag['abs_avg']:.2f})"
            )
        if details:
            lines.append(
                "- Resource pressure: " + "; ".join(details) + "; "
                "favor resource-positive moves or reversible trades."
            )
    if round_end_notes:
        suffix = " (low sample)" if round_end_small_sample else ""
        lines.append(
            "- Round-end deltas: " + ", ".join(round_end_notes) + suffix
            + "; buffer survival/cargo/vitality before large expansions."
        )
        if drain_notes:
            lines.append(
                "- End-of-round drain: " + ", ".join(drain_notes)
                + "; build buffers early and avoid upkeep-heavy moves."
            )
            if drain_sig_available:
                lines.append(
                    "- Retention bias: prioritize signatures that keep action gains "
                    "through round end (small action→round-end gaps) when available."
                )
        if round_end_pressure and round_end_severity >= 0.7:
            lines.append(
                "- Safety priority: severe round-end losses; favor low-cost "
                "resource/vitality recovery, defensive positioning, and reversible "
                "coordination; keep variations close to baseline."
            )
            if critical_pressure:
                lines.append(
                    "- Critical stabilization: losses exceed emergency thresholds; "
                    "pause large transfers or expansions unless they directly improve "
                    "survival, and rebuild cargo/vitality buffers before risky moves."
                )
            if round_end_cargo is not None and round_end_cargo < 0.0:
                lines.append(
                    f"- Recovery target: offset ~{abs(round_end_cargo):.1f} avg cargo loss "
                    "before large, irreversible moves when feasible."
                )
            if round_end_vitality is not None and round_end_vitality < 0.0:
                lines.append(
                    f"- Recovery target: offset ~{abs(round_end_vitality):.1f} avg vitality loss "
                    "before large, irreversible moves when feasible."
                )
            if diversity_rescue:
                lines.append(
                    "- Diversity rescue: population convergence under severe round-end losses; "
                    "keep a baseline-adjacent variation (overlap >= 0.5) with recovery-positive "
                    "signals to preserve diversity without high risk."
                )
            if diversity_emergency:
                lines.append(
                    "- Diversity emergency: dominance is extreme; assign at least one "
                    "member to a low-risk, baseline-adjacent variation to preserve "
                    "optionality."
                )
        elif round_end_pressure:
            lines.append(
                "- Safety priority: round-end losses; bias toward "
                "resource- and vitality-positive, reversible moves and avoid large "
                "expansion/attack spikes."
            )
    if context_total:
        lines.append(
            f"- Recent context mix: {context_unique} contexts; "
            f"dominant {dominant_context_share:.2f} in {dominant_context}"
        )
        if current_key and current_key not in context_counts:
            lines.append(
                f"- Current context not in recent mix: {current_key}; "
                "favor reversible tests."
            )
    elif missing_context:
        lines.append(f"- Context tags missing for recent actions: {missing_context}")
    if context_top is not None:
        ctx_avg, ctx_weight, ctx_count, _ctx_sim, ctx_sig = context_top
        lines.append(
            f"- Context-weighted top signature: {self._format_signature(ctx_sig)} "
            f"(avg {ctx_avg:.2f}, weight {ctx_weight:.2f}, n={ctx_count})"
        )
    elif current_tags:
        lines.append(
            "- Context-weighted signals: none (no similar contexts in window)"
        )
    if member_signature_shares:
        lines.append(
            f"- Recent population signature share: avg {avg_signature_share:.2f}, "
            f"dominant rounds {dominant_share_hits}/{len(member_signature_shares)}"
        )
    if pressure_reasons:
        lines.append(
            f"- Exploration pressure: base {base_exploration_bonus:.2f} -> "
            f"{exploration_bonus:.2f} ({', '.join(pressure_reasons)})"
        )

    if baseline_sig is not None:
        base_stats = sig_perf_lookup.get(baseline_sig)
        base_avg = None
        base_std = None
        base_count = None
        baseline_recovery_score = None
        if base_stats:
            base_avg, base_std, base_count, _ = base_stats
        elif best_by_avg is not None and best_by_avg[2] == baseline_sig:
            base_avg, base_count, _ = best_by_avg
        if recovery_bias:
            baseline_recovery_score = _recovery_score(baseline_sig)
        baseline_label = "stable"
        if baseline_reason == "context":
            baseline_label = "context-aligned"
        elif baseline_reason == "average":
            baseline_label = "average"
        elif baseline_reason == "diversity":
            baseline_label = "diversity"
        detail_bits = []
        if baseline_reason == "context" and context_candidate is not None:
            ctx_avg, ctx_weight, ctx_count, _ctx_sim, ctx_sig = context_candidate
            if ctx_sig == baseline_sig:
                detail_bits.append(
                    f"ctx avg {ctx_avg:.2f}, weight {ctx_weight:.2f}, n={ctx_count}"
                )
        if baseline_recovery_score is not None and abs(baseline_recovery_score) >= 0.05:
            detail_bits.append(f"recovery {baseline_recovery_score:+.2f}")
        detail_suffix = f", {', '.join(detail_bits)}" if detail_bits else ""
        if base_avg is not None and base_count is not None:
            if base_std is not None:
                lines.append(
                    f"- Suggested baseline ({baseline_label}): "
                    f"{self._format_signature(baseline_sig)} "
                    f"(avg {base_avg:.2f}, std {base_std:.2f}, n={base_count}{detail_suffix})"
                )
            else:
                lines.append(
                    f"- Suggested baseline ({baseline_label}): "
                    f"{self._format_signature(baseline_sig)} "
                    f"(avg {base_avg:.2f}, n={base_count}{detail_suffix})"
                )
        else:
            lines.append(
                f"- Suggested baseline ({baseline_label}): "
                f"{self._format_signature(baseline_sig)}{detail_suffix}"
            )
        if diversity_override is not None:
            prev_sig = diversity_override.get("previous")
            prev_share = diversity_override.get("previous_share", 0.0)
            cand_share = diversity_override.get("candidate_share", 0.0)
            cand_avg = diversity_override.get("candidate_avg")
            base_avg = diversity_override.get("baseline_avg")
            avg_note = ""
            if base_avg is not None and cand_avg is not None:
                avg_note = f" (avg {cand_avg:.2f} vs {base_avg:.2f})"
            lines.append(
                "- Diversity adjustment: shifted off population dominant "
                f"{self._format_signature(prev_sig)} (share {prev_share:.2f}) "
                f"to {self._format_signature(baseline_sig)} "
                f"(share {cand_share:.2f}){avg_note}"
            )
    elif best_by_avg is not None:
        avg, count, sig = best_by_avg
        lines.append(
            f"- Suggested baseline: {self._format_signature(sig)} "
            f"(avg {avg:.2f}, n={count})"
        )

    recovery_candidate = None
    if recovery_bias:
        candidates = []
        for sig, perfs in perf_by_sig.items():
            count = len(perfs)
            if count < min_samples:
                continue
            recovery_score = _recovery_score(sig)
            if recovery_score <= 0.0:
                continue
            avg = sum(perfs) / count if count else 0.0
            cargo_avg = sig_cargo_avg.get(sig)
            survival_avg = sig_survival_avg.get(sig)
            candidates.append(
                (recovery_score, avg, count, sig, cargo_avg, survival_avg)
            )
        if candidates:
            candidates.sort(reverse=True)
            recovery_candidate = candidates[0]

    if recovery_candidate is not None:
        recovery_score, avg, count, sig, cargo_avg, survival_avg = recovery_candidate
        detail_bits = [f"score {recovery_score:+.2f}", f"avg {avg:.2f}", f"n={count}"]
        if cargo_avg is not None:
            detail_bits.append(f"cargo {cargo_avg:+.2f}")
        if survival_avg is not None:
            detail_bits.append(f"survival {survival_avg:+.2f}")
        vitality_avg = sig_vitality_avg.get(sig)
        if vitality_avg is not None:
            detail_bits.append(f"vitality {vitality_avg:+.2f}")
        lines.append(
            f"- Recovery candidate: {self._format_signature(sig)} "
            f"({', '.join(detail_bits)})"
        )

    if variation_candidate is not None:
        (
            adjusted,
            ucb,
            blended_avg,
            _raw_avg,
            count,
            sig,
            overlap,
            dominance_penalty,
            safety_penalty,
            stabilization_penalty,
            resource_adjustment,
            survival_adjustment,
            vitality_adjustment,
            drain_adjustment,
            recovery_adjustment,
            diversity_rescue_bonus,
            diversity_emergency_bonus,
            ctx_weight,
        ) = variation_candidate
        detail_bits = [
            f"score {adjusted:.2f}",
            f"ucb {ucb:.2f}",
            f"avg {blended_avg:.2f}",
            f"n={count}",
        ]
        if ctx_weight > 0:
            detail_bits.append(f"ctx_w {ctx_weight:.2f}")
        if baseline_sig:
            detail_bits.append(f"overlap {overlap:.2f}")
        if dominance_penalty > 0:
            detail_bits.append(f"pop-penalty {dominance_penalty:.2f}")
        if safety_penalty > 0:
            detail_bits.append(f"safety -{safety_penalty:.2f}")
        if stabilization_penalty > 0:
            detail_bits.append(f"stabilize -{stabilization_penalty:.2f}")
        if abs(resource_adjustment) > 1e-6:
            detail_bits.append(f"resource {resource_adjustment:+.2f}")
        if abs(survival_adjustment) > 1e-6:
            detail_bits.append(f"survival {survival_adjustment:+.2f}")
        if abs(vitality_adjustment) > 1e-6:
            detail_bits.append(f"vitality {vitality_adjustment:+.2f}")
        if abs(drain_adjustment) > 1e-6:
            detail_bits.append(f"drain {drain_adjustment:+.2f}")
        if abs(recovery_adjustment) > 1e-6:
            detail_bits.append(f"recovery {recovery_adjustment:+.2f}")
        if diversity_rescue_bonus > 0:
            detail_bits.append(f"rescue +{diversity_rescue_bonus:.2f}")
        if diversity_emergency_bonus > 0:
            detail_bits.append(f"emergency +{diversity_emergency_bonus:.2f}")
        lines.append(
            f"- Risk-adjusted variation: {self._format_signature(sig)} "
            f"({', '.join(detail_bits)})"
        )
    elif fallback_variation is not None:
        fallback_sig = fallback_variation["signature"]
        seed_tag = fallback_variation["seed_tag"]
        source = fallback_variation["source"]
        lines.append(
            f"- Diversity variation seed: {self._format_signature(fallback_sig)} "
            f"(add {seed_tag}, source {source}; keep overlap >= 0.5, avoid high-risk tags)"
        )
    elif sig_stats:
        ucb, blended_avg, _raw_avg, count, sig, ctx_weight = sig_stats[0]
        detail_bits = [f"score {ucb:.2f}", f"avg {blended_avg:.2f}", f"n={count}"]
        if ctx_weight > 0:
            detail_bits.append(f"ctx_w {ctx_weight:.2f}")
        lines.append(
            f"- Exploration candidate (UCB): {self._format_signature(sig)} "
            f"({', '.join(detail_bits)})"
        )

    if novelty_candidate is not None:
        avg, std, count, novelty_avg, sig = novelty_candidate
        lines.append(
            f"- Novelty candidate: {self._format_signature(sig)} "
            f"(novelty {novelty_avg:.2f}, avg {avg:.2f}, n={count})"
        )

    if dominant_sig is not None and dominant_share >= 0.6:
        if len(signature_counts) > 1:
            lines.append(
                f"- Diversity guard: {dominant_share:.2f} of recent actions share "
                f"{self._format_signature(dominant_sig)}; consider mixing tags."
            )
        else:
            lines.append(
                f"- Diversity guard: all recent actions share "
                f"{self._format_signature(dominant_sig)}; inject one low-risk tag "
                "to avoid lock-in."
            )

    if min_round is not None and pop_entries:
        lines.append(
            f"- Population window: {min_round}-{current_round} "
            f"(actions {len(pop_entries)})"
        )
        if niche_candidate is not None:
            avg, share, count, sig = niche_candidate
            lines.append(
                f"- Population niche (positive, low-share): "
                f"{self._format_signature(sig)} "
                f"(avg {avg:.2f}, share {share:.2f}, n={count})"
            )
        lines.append(
            "- Underused tags with positive population signal: "
            f"{_format_tag_list(promising_tags)}"
        )
        lines.append(
            "- Caution tags with negative population signal: "
            f"{_format_tag_list(caution_tags)}"
        )

    if negative:
        lines.append(
            "- Recent negative signatures (personal): "
            + "; ".join(
                f"{self._format_signature(sig)} (avg {avg:.2f}, n={count})"
                for avg, count, sig in negative[:max_items]
            )
        )

    return "\n".join(lines)
