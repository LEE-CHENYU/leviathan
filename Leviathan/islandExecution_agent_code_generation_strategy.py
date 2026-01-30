from typing import List, Tuple, Optional
from collections import Counter
import math
import numpy as np
from Leviathan.Member import Member




class IslandExecutionStrategyMixin:
    def _compute_action_tag_stats(
        self,
        window_rounds: int = 4,
    ) -> Optional[dict]:
        """Compute recent action-tag outcome aggregates."""
        if not self.code_memory or not self.execution_history.get('rounds'):
            return None

        current_round = len(self.execution_history['rounds'])
        end_round = max(0, current_round - 1)
        min_round = max(0, end_round - max(1, window_rounds) + 1)

        tags = ("attack", "offer", "offer_land", "bear", "expand", "message")
        stats = {
            tag: {"count": 0, "reward": 0.0, "survival": 0.0, "balanced": 0.0}
            for tag in tags
        }

        for memory in self.code_memory.values():
            for mem in memory:
                round_num = mem.get("context", {}).get("round")
                if round_num is None or round_num < min_round or round_num > end_round:
                    continue
                signature = self._get_entry_signature(
                    mem,
                    prefer_executed=True,
                    fallback_to_planned=False,
                )
                if not signature:
                    continue

                delta = mem.get("context", {}).get("delta") or {}
                perf = mem.get("performance", 0.0)
                reward_score = mem.get("reward_score")
                if reward_score is None:
                    reward_score = self._compute_reward_score(delta)
                survival_score = mem.get("survival_score")
                if survival_score is None:
                    survival_score = self._compute_survival_score(perf)
                balanced_score = mem.get("balanced_score")
                if balanced_score is None:
                    balanced_score = self._compute_balanced_score(delta, perf)

                for tag in set(signature):
                    if tag not in stats:
                        continue
                    stats[tag]["count"] += 1
                    stats[tag]["reward"] += reward_score
                    stats[tag]["survival"] += survival_score
                    stats[tag]["balanced"] += balanced_score

        return {
            "min_round": min_round,
            "end_round": end_round,
            "tags": tags,
            "stats": stats,
        }

    def _summarize_exploration_posture(
        self,
        member,
        member_id: int,
        round_context: Optional[dict] = None,
        window: int = 4,
    ) -> str:
        """Summarize exploration vs. exploitation posture without forcing choices."""
        if round_context is None:
            round_context = self._compute_round_context()

        avg_resources = float(round_context.get("avg_vitality", 0.0)) + float(
            round_context.get("avg_cargo", 0.0)
        )
        my_resources = float(getattr(member, "vitality", 0.0)) + float(
            getattr(member, "cargo", 0.0)
        )
        resource_ratio = my_resources / max(1.0, avg_resources)

        member_key = self._member_storage_key(member_id)
        perf_history = (
            self.performance_history.get(member_key, []) if member_key is not None else []
        )
        slope = None
        trend = "insufficient data"
        if len(perf_history) >= 2:
            recent = perf_history[-max(2, int(window)):]
            slope = float(recent[-1] - recent[0])
            if slope > 0.05:
                trend = "improving"
            elif slope < -0.05:
                trend = "declining"
            else:
                trend = "flat"

        streak, last_sig = self._signature_streak(
            member_id,
            prefer_executed=True,
            fallback_to_planned=False,
            include_empty=False,
        )
        idle_streak = self._idle_signature_streak(member_id, window=window)
        pop_stats = self._compute_population_diversity_stats(window_rounds=3)
        dominant_share = pop_stats.get("dominant_share", 0.0) if pop_stats else None

        lines = ["Exploration posture (soft guidance):"]
        lines.append(f"- resource buffer vs avg: {resource_ratio:.2f} (>1 means above avg)")
        if slope is None:
            lines.append("- recent performance: insufficient data")
        else:
            lines.append(f"- recent performance slope: {slope:.2f} ({trend})")
        if streak > 1:
            lines.append(
                f"- recent signature streak: {streak}x ({self._format_signature(last_sig)})"
            )
        if idle_streak >= 2:
            lines.append(f"- recent inactivity streak: {idle_streak}x (no executed actions)")
        if dominant_share is not None:
            lines.append(f"- population dominance share: {dominant_share:.2f}")

        guidance = []
        if resource_ratio < 0.85:
            guidance.append("conserve: stabilize before experimenting")
        elif resource_ratio > 1.15:
            guidance.append("explore: you can afford a safe variation")
        if slope is not None:
            if slope < -0.05:
                guidance.append("stabilize: recent outcomes declining")
            elif abs(slope) < 0.03 and streak >= 3 and resource_ratio >= 0.9:
                guidance.append("stagnation: consider a safe variation")
        if dominant_share is not None and dominant_share >= 0.6:
            guidance.append("population convergence: diversify if safe")
        if idle_streak >= 2 and resource_ratio >= 0.9:
            guidance.append("inactive: ensure at least one concrete action executes")
        if guidance:
            lines.append("- guidance: " + " | ".join(guidance))

        return "\n".join(lines)

    def _summarize_diversity_guidance(self, member, window_rounds: int = 3) -> str:
        """Generate lightweight guidance to preserve diversity without constraining actions."""
        stats = self._compute_population_diversity_stats(window_rounds=window_rounds)
        if not stats:
            return "No diversity guidance available."

        streak, last_sig = self._signature_streak(
            member.surviver_id,
            prefer_executed=True,
            fallback_to_planned=False,
            include_empty=False,
        )
        idle_streak = self._idle_signature_streak(member.surviver_id)
        eligibility = {
            "bear": bool(getattr(member, "is_qualified_to_reproduce", False)),
            "offer_land": bool(getattr(member, "land_num", 0) > 0),
            "expand": bool(getattr(member, "current_clear_list", [])),
        }
        underused = stats.get("underused_tags", [])
        eligible_underused = [tag for tag in underused if eligibility.get(tag, True)]

        lines = ["Diversity guidance:"]
        if streak > 1:
            lines.append(
                f"- recent signature streak: {streak}x ({self._format_signature(last_sig)})"
            )
        if idle_streak >= 2:
            lines.append(f"- inactivity streak: {idle_streak}x (no executed actions)")
        if stats.get("dominant_share", 0.0) >= 0.5:
            lines.append(
                f"- population dominance: {stats['dominant_share']:.2f} "
                f"({self._format_signature(stats.get('dominant_signature'))})"
            )
        if eligible_underused:
            lines.append(f"- underused tags you can try: {', '.join(eligible_underused)}")
            tag_stats_bundle = self._compute_action_tag_stats(window_rounds=window_rounds)
            if tag_stats_bundle:
                stats_map = tag_stats_bundle["stats"]
                total_count = sum(info["count"] for info in stats_map.values())
                total_balanced = sum(info["balanced"] for info in stats_map.values())
                if total_count <= 0:
                    lines.append(
                        "- no recent action impact data; exploration is purely experimental."
                    )
                    baseline = None
                    threshold = None
                else:
                    baseline = total_balanced / total_count
                    threshold = max(0.0, baseline)

                promising = []
                low_sample = []
                for tag in eligible_underused:
                    info = stats_map.get(tag, {})
                    count = info.get("count", 0)
                    if count < 2:
                        low_sample.append(tag)
                        continue
                    if threshold is not None:
                        avg_balanced = info.get("balanced", 0.0) / max(1, count)
                        if avg_balanced >= threshold:
                            promising.append(
                                f"{tag} (avg={avg_balanced:.2f}, n={count})"
                            )

                if promising:
                    lines.append(
                        "- underused tags with positive signals: "
                        + ", ".join(promising)
                    )
                elif eligible_underused and baseline is not None:
                    lines.append(
                        f"- no underused tags exceed baseline ({baseline:.2f}); "
                        "treat exploration as experiments."
                    )
                if low_sample:
                    lines.append(
                        "- underused tags with low samples: " + ", ".join(low_sample)
                    )
        elif underused:
            lines.append(f"- underused tags (not currently eligible): {', '.join(underused)}")

        if len(lines) == 1:
            return "No diversity guidance available."
        return "\n".join(lines)

    def _compute_contextual_signature_scores(
        self,
        memory_entries: list,
        current_stats: Optional[dict],
        current_round_context: Optional[dict],
        current_relation_context: Optional[dict],
        min_similarity: float = 0.35,
    ) -> dict:
        """Compute similarity-weighted scores for signatures in the current context."""
        if not memory_entries:
            return {}
        if not (current_stats or current_round_context or current_relation_context):
            return {}

        buckets = {}
        for mem in memory_entries:
            context = mem.get("context", {}) or {}
            similarity = self._context_similarity_score(
                context,
                current_stats,
                current_round_context,
                current_relation_context,
            )
            if similarity is None or similarity < min_similarity:
                continue
            sig = self._get_entry_signature(
                mem,
                prefer_executed=True,
                fallback_to_planned=True,
            )
            if not sig:
                continue
            score = mem.get("balanced_score")
            if score is None:
                delta = context.get("delta") or {}
                perf = mem.get("performance", 0.0)
                score = self._compute_balanced_score(delta, perf)
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            bucket = buckets.setdefault(
                sig,
                {"weight": 0.0, "score": 0.0, "count": 0, "sim_sum": 0.0},
            )
            bucket["weight"] += similarity
            bucket["score"] += similarity * score
            bucket["count"] += 1
            bucket["sim_sum"] += similarity

        results = {}
        for sig, bucket in buckets.items():
            weight = bucket["weight"]
            avg_score = bucket["score"] / max(1e-9, weight)
            avg_sim = bucket["sim_sum"] / max(1, bucket["count"])
            results[sig] = {
                "avg_score": avg_score,
                "avg_sim": avg_sim,
                "count": bucket["count"],
                "weight": weight,
            }
        return results

    def _summarize_strategy_recommendations(
        self,
        member,
        member_id: int,
        window: int = 8,
        min_samples: int = 2,
        exploration_bonus: float = 0.35,
        population_window_rounds: int = 3,
        max_items: int = 3,
        current_stats: Optional[dict] = None,
        current_round_context: Optional[dict] = None,
        current_relation_context: Optional[dict] = None,
    ) -> str:
        """Provide lightweight decision support without forcing convergence."""
        member_key = self._member_storage_key(member_id)
        memory = self.code_memory.get(member_key, []) if member_key is not None else []
        if not memory:
            return "No strategy recommendations available."

        recent = memory[-max(1, int(window)):]
        total_recent = len(recent)

        def _collect_signature_scores(use_executed: bool):
            counts = Counter()
            scores = {}
            for mem in recent:
                sig = (
                    self._get_executed_signature(mem)
                    if use_executed
                    else self._get_entry_signature(mem)
                )
                if not sig:
                    continue
                counts[sig] += 1
                score = mem.get("balanced_score")
                if score is None:
                    delta = mem.get("context", {}).get("delta") or {}
                    perf = mem.get("performance", 0.0)
                    score = self._compute_balanced_score(delta, perf)
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 0.0
                scores.setdefault(sig, []).append(score)
            return counts, scores

        executed_counts, executed_scores = _collect_signature_scores(True)
        used_fallback = False
        signature_counts = executed_counts
        signature_scores = executed_scores
        if not signature_counts:
            signature_counts, signature_scores = _collect_signature_scores(False)
            used_fallback = True

        if not signature_counts:
            return "No strategy recommendations available."

        executed_samples = sum(executed_counts.values())
        idle_count = max(0, total_recent - executed_samples)
        idle_ratio = idle_count / max(1, total_recent)
        planned_only_count = 0
        mismatch_count = 0
        for mem in recent:
            planned_sig = self._get_planned_signature(mem)
            executed_sig = self._get_executed_signature(mem)
            if planned_sig and not executed_sig:
                planned_only_count += 1
            elif planned_sig and executed_sig:
                if tuple(planned_sig) != tuple(executed_sig):
                    mismatch_count += 1
        planned_only_ratio = planned_only_count / max(1, total_recent)
        mismatch_ratio = mismatch_count / max(1, total_recent)
        reliability_stats = self._signature_reliability_stats(recent)

        def _signature_note(sig: tuple) -> str:
            notes = []
            ineligible = self._signature_ineligible_tags(member, sig)
            if ineligible:
                notes.append(f"ineligible now: {', '.join(ineligible)}")
            reliability = reliability_stats.get(sig)
            if reliability:
                exec_rate = reliability.get("execution_rate", 1.0)
                infeasible_rate = reliability.get("infeasible_rate", 0.0)
                if exec_rate < 0.9 or infeasible_rate > 0.0:
                    notes.append(
                        f"exec_rate {exec_rate:.2f}, infeasible {infeasible_rate:.2f}"
                    )
            if notes:
                return " [" + "; ".join(notes) + "]"
            return ""

        def _best_feasible_candidate(candidates, exclude_sigs=None):
            if not candidates:
                return None
            excluded = set(exclude_sigs or [])
            for candidate in candidates:
                sig = candidate["sig"]
                if sig in excluded:
                    continue
                if not self._signature_ineligible_tags(member, sig):
                    return candidate
            return None

        total = sum(signature_counts.values())
        dominant_sig = None
        dominant_share = 0.0
        if signature_counts:
            dominant_sig, dominant_count = signature_counts.most_common(1)[0]
            dominant_share = dominant_count / max(1, total)

        streak, last_sig = self._signature_streak(
            member_id,
            prefer_executed=True,
            fallback_to_planned=False,
            include_empty=False,
        )

        context_scores = self._compute_contextual_signature_scores(
            recent,
            current_stats,
            current_round_context,
            current_relation_context,
        )
        overall_avg = 0.0
        total_scores = sum(len(scores) for scores in signature_scores.values())
        if total_scores:
            overall_avg = (
                sum(sum(scores) for scores in signature_scores.values()) / total_scores
            )

        exploration_bonus_effective = float(exploration_bonus)
        exploration_adjustments = []
        if dominant_share >= 0.6:
            exploration_bonus_effective += 0.05
            exploration_adjustments.append("recent dominance")
        if streak >= 3:
            exploration_bonus_effective += 0.05
            exploration_adjustments.append("signature streak")
        if idle_ratio >= 0.5:
            exploration_bonus_effective += 0.05
            exploration_adjustments.append("idle window")
        resource_ratio = None
        if current_stats and current_round_context:
            avg_resources = float(current_round_context.get("avg_vitality", 0.0)) + float(
                current_round_context.get("avg_cargo", 0.0)
            )
            my_resources = float(current_stats.get("vitality", 0.0)) + float(
                current_stats.get("cargo", 0.0)
            )
            if avg_resources >= 1.0:
                resource_ratio = my_resources / avg_resources
                if resource_ratio < 0.85:
                    exploration_bonus_effective -= 0.05
                    exploration_adjustments.append("low resources")
                elif resource_ratio > 1.15:
                    exploration_bonus_effective += 0.05
                    exploration_adjustments.append("resource buffer")
        pop_stats = self._compute_population_diversity_stats(
            window_rounds=population_window_rounds
        )
        if pop_stats:
            diversity_adjustment, diversity_error = self._update_diversity_controller(
                pop_stats.get("diversity_ratio"),
                pop_stats.get("signature_entropy_norm"),
            )
            if abs(diversity_adjustment) > 1e-6:
                exploration_bonus_effective += diversity_adjustment
                alpha = float(self._diversity_controller.get("alpha", 0.0))
                exploration_adjustments.append(
                    f"diversity α={alpha:.2f} err={diversity_error:.2f}"
                )
        tag_pressure_bundle = self._compute_population_tag_pressure(
            window_rounds=population_window_rounds
        )
        tag_pressure = tag_pressure_bundle.get("pressure") or {}
        pop_signature_counts = tag_pressure_bundle.get("signature_counts") or Counter()
        pop_signature_shares = tag_pressure_bundle.get("signature_shares") or {}
        pop_unique_signatures = int(
            tag_pressure_bundle.get("unique_signatures") or len(pop_signature_counts)
        )
        pop_dominant_sig = tag_pressure_bundle.get("dominant_signature", tuple())
        pop_dominant_share = float(tag_pressure_bundle.get("dominant_share", 0.0) or 0.0)
        if pop_dominant_share >= 0.6:
            exploration_bonus_effective += 0.05
            exploration_adjustments.append("population convergence")
        exploration_bonus_effective = max(
            0.2, min(0.6, exploration_bonus_effective)
        )
        sig_stats = []
        for sig, scores in signature_scores.items():
            count = len(scores)
            avg = sum(scores) / count if count else 0.0
            ucb = avg + exploration_bonus_effective * math.sqrt(
                math.log(total + 1) / max(1, count)
            )
            diversity_bonus = self._signature_diversity_bonus(sig, tag_pressure)
            reliability = reliability_stats.get(sig, {})
            exec_rate = reliability.get("execution_rate", 1.0)
            infeasible_rate = reliability.get("infeasible_rate", 0.0)
            reliability_penalty = -0.15 * max(0.0, 1.0 - float(exec_rate))
            reliability_penalty -= 0.1 * max(0.0, min(1.0, float(infeasible_rate)))
            context_bonus = 0.0
            context_info = context_scores.get(sig)
            if context_info is not None:
                delta = context_info.get("avg_score", 0.0) - overall_avg
                context_bonus = max(-0.12, min(0.12, 0.25 * delta))
            novelty_bonus = 0.05 * (1.0 - (count / max(1.0, total)))
            feasibility_penalty = 0.0
            ineligible_tags = self._signature_ineligible_tags(member, sig)
            if ineligible_tags:
                feasibility_penalty = -0.08 * (
                    len(ineligible_tags) / max(1, len(sig))
                )
            pop_diversity_bonus = 0.0
            if pop_signature_shares:
                expected_share = 1.0 / max(
                    1, pop_unique_signatures or len(pop_signature_shares)
                )
                pop_share = float(pop_signature_shares.get(sig, 0.0) or 0.0)
                delta = (expected_share - pop_share) / max(expected_share, 1e-9)
                delta = max(-1.0, min(1.0, delta))
                pop_diversity_bonus = 0.06 * delta
                if (
                    pop_dominant_sig
                    and sig == pop_dominant_sig
                    and pop_dominant_share >= 0.6
                ):
                    pop_diversity_bonus -= 0.05 * (pop_dominant_share - 0.6) / 0.4
            dominance_penalty = 0.0
            if dominant_sig and sig == dominant_sig and dominant_share >= 0.6:
                dominance_penalty = -0.1 * (dominant_share - 0.6) / 0.4
            streak_penalty = -0.05 if (streak >= 3 and sig == last_sig) else 0.0
            adjusted = (
                ucb
                + diversity_bonus
                + reliability_penalty
                + context_bonus
                + novelty_bonus
                + pop_diversity_bonus
                + feasibility_penalty
                + dominance_penalty
                + streak_penalty
            )
            sig_stats.append(
                {
                    "adjusted": adjusted,
                    "ucb": ucb,
                    "avg": avg,
                    "count": count,
                    "sig": sig,
                    "diversity": diversity_bonus,
                    "pop_diversity": pop_diversity_bonus,
                    "reliability": reliability_penalty,
                    "feasibility": feasibility_penalty,
                    "context": context_bonus,
                    "novelty": novelty_bonus,
                    "dominance": dominance_penalty,
                    "streak": streak_penalty,
                }
            )
        sig_stats.sort(key=lambda item: item["adjusted"], reverse=True)

        avg_sorted = sorted(
            [
                (sum(scores) / len(scores), len(scores), sig)
                for sig, scores in signature_scores.items()
            ],
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        best_by_avg = None
        for avg, count, sig in avg_sorted:
            if count >= min_samples:
                best_by_avg = (avg, count, sig)
                break
        if best_by_avg is None and avg_sorted:
            best_by_avg = avg_sorted[0]

        negative = [
            (avg, count, sig)
            for avg, count, sig in avg_sorted
            if count >= min_samples and avg < 0.0
        ]
        negative.sort(key=lambda item: item[0])

        member_tag_counts = Counter()
        for mem in recent:
            sig = self._get_executed_signature(mem)
            if sig:
                member_tag_counts.update(sig)
        if not member_tag_counts:
            for mem in recent:
                sig = self._get_entry_signature(mem)
                if sig:
                    member_tag_counts.update(sig)

        known_tags = ("attack", "offer", "offer_land", "bear", "expand", "message")
        member_underuse_threshold = max(1, int(0.2 * total))
        eligibility = {
            "bear": bool(getattr(member, "is_qualified_to_reproduce", False)),
            "offer_land": bool(getattr(member, "land_num", 0) > 0),
            "expand": bool(getattr(member, "current_clear_list", [])),
        }

        tag_stats_bundle = self._compute_action_tag_stats(
            window_rounds=population_window_rounds
        )
        promising_tags = []
        caution_tags = []
        if tag_stats_bundle:
            stats_map = tag_stats_bundle.get("stats", {})
            for tag in known_tags:
                info = stats_map.get(tag, {})
                count = info.get("count", 0)
                if count < min_samples:
                    continue
                avg_balanced = info.get("balanced", 0.0) / max(1, count)
                if (
                    avg_balanced > 0.0
                    and member_tag_counts.get(tag, 0) <= member_underuse_threshold
                    and eligibility.get(tag, True)
                ):
                    promising_tags.append((avg_balanced, count, tag))
                elif avg_balanced < 0.0:
                    caution_tags.append((avg_balanced, count, tag))

        promising_tags.sort(reverse=True)
        caution_tags.sort()

        context_candidate = None
        if context_scores:
            context_sorted = sorted(
                [
                    (
                        info.get("avg_score", 0.0),
                        info.get("avg_sim", 0.0),
                        info.get("count", 0),
                        sig,
                    )
                    for sig, info in context_scores.items()
                ],
                key=lambda item: (item[0], item[1], item[2]),
                reverse=True,
            )
            if context_sorted:
                context_candidate = context_sorted[0]

        def _format_tag_list(stats):
            if not stats:
                return "none"
            return ", ".join(
                f"{tag} (avg {avg:.2f}, n={count})"
                for avg, count, tag in stats[:max_items]
            )

        lines = [
            "Strategy recommendations (soft guidance):",
            f"- Recent window: {total} signature samples (out of {total_recent} entries)",
        ]
        if exploration_adjustments or exploration_bonus_effective != exploration_bonus:
            reason_text = ", ".join(exploration_adjustments) if exploration_adjustments else "context"
            ratio_text = f"; resource_ratio {resource_ratio:.2f}" if resource_ratio is not None else ""
            lines.append(
                f"- Exploration bonus (effective): {exploration_bonus_effective:.2f} "
                f"({reason_text}{ratio_text})"
            )
        if idle_ratio >= 0.5:
            lines.append(
                f"- Inactivity rate (executed): {idle_ratio:.2f} "
                f"({idle_count}/{total_recent})"
            )
        if total_recent >= 2 and planned_only_ratio >= 0.3:
            lines.append(
                f"- Planned-only rate: {planned_only_ratio:.2f} "
                f"({planned_only_count}/{total_recent}); add a fallback action if safe."
            )
        if total_recent >= 2 and mismatch_ratio >= 0.3:
            lines.append(
                f"- Plan/execution mismatch: {mismatch_ratio:.2f} "
                f"({mismatch_count}/{total_recent}); align conditions with intended actions."
            )
        if used_fallback:
            lines.append(
                "- Note: no executed actions in window; using planned signatures for guidance."
            )

        if best_by_avg is not None:
            avg, count, sig = best_by_avg
            note = _signature_note(sig)
            lines.append(
                f"- Suggested baseline: {self._format_signature(sig)} "
                f"(avg {avg:.2f}, n={count}){note}"
            )
        if context_candidate is not None:
            avg_score, avg_sim, count, sig = context_candidate
            lines.append(
                f"- Context-matched candidate: {self._format_signature(sig)} "
                f"(avg {avg_score:.2f}, sim {avg_sim:.2f}, n={count})"
            )

        baseline_sig = best_by_avg[2] if best_by_avg else None
        explore_sig = sig_stats[0]["sig"] if sig_stats else None
        baseline_ineligible = bool(
            self._signature_ineligible_tags(member, baseline_sig)
        ) if baseline_sig else False
        explore_ineligible = bool(
            self._signature_ineligible_tags(member, explore_sig)
        ) if explore_sig else False

        if sig_stats:
            top = sig_stats[0]
            sig = top["sig"]
            note = _signature_note(sig)
            adjustment_parts = []
            for key, label in (
                ("diversity", "diversity"),
                ("pop_diversity", "pop_diversity"),
                ("reliability", "reliability"),
                ("feasibility", "feasibility"),
                ("context", "context"),
                ("novelty", "novelty"),
                ("dominance", "dominance"),
                ("streak", "streak"),
            ):
                value = top.get(key, 0.0)
                if value:
                    adjustment_parts.append(f"{label} {value:+.2f}")
            if adjustment_parts:
                lines.append(
                    "- Exploration candidate (UCB adjusted): "
                    f"{self._format_signature(sig)} "
                    f"(score {top['adjusted']:.2f}, ucb {top['ucb']:.2f}, "
                    f"{', '.join(adjustment_parts)}, avg {top['avg']:.2f}, n={top['count']}){note}"
                )
            else:
                lines.append(
                    f"- Exploration candidate (UCB): {self._format_signature(sig)} "
                    f"(score {top['ucb']:.2f}, avg {top['avg']:.2f}, n={top['count']}){note}"
                )

        shown_sigs = {sig for sig in (baseline_sig, explore_sig) if sig}
        if baseline_ineligible or explore_ineligible:
            feasible_candidate = _best_feasible_candidate(sig_stats)
            if feasible_candidate:
                sig = feasible_candidate["sig"]
                if sig not in shown_sigs:
                    note = _signature_note(sig)
                    lines.append(
                        f"- Feasible alternative: {self._format_signature(sig)} "
                        f"(score {feasible_candidate['adjusted']:.2f}, "
                        f"avg {feasible_candidate['avg']:.2f}, "
                        f"n={feasible_candidate['count']}){note}"
                    )
                    shown_sigs.add(sig)

        avoid_sigs = set()
        if dominant_sig is not None and dominant_share >= 0.6:
            avoid_sigs.add(dominant_sig)
        if pop_dominant_sig is not None and pop_dominant_share >= 0.6:
            avoid_sigs.add(pop_dominant_sig)
        if streak >= 3 and last_sig:
            avoid_sigs.add(last_sig)
        if avoid_sigs:
            diverse_candidate = _best_feasible_candidate(
                sig_stats,
                exclude_sigs=avoid_sigs,
            )
            if diverse_candidate:
                sig = diverse_candidate["sig"]
                if sig not in shown_sigs:
                    note = _signature_note(sig)
                    lines.append(
                        f"- Diversity-safe alternative: {self._format_signature(sig)} "
                        f"(score {diverse_candidate['adjusted']:.2f}, "
                        f"avg {diverse_candidate['avg']:.2f}, "
                        f"n={diverse_candidate['count']}){note}"
                    )
                    shown_sigs.add(sig)

        if dominant_sig is not None and dominant_share >= 0.6 and len(signature_counts) > 1:
            lines.append(
                f"- Diversity guard: {dominant_share:.2f} of recent actions share "
                f"{self._format_signature(dominant_sig)}; consider mixing tags."
            )
        if pop_dominant_sig is not None and pop_dominant_share >= 0.6:
            lines.append(
                f"- Population guard: {pop_dominant_share:.2f} of population actions "
                f"share {self._format_signature(pop_dominant_sig)}; diversify if safe."
            )

        if tag_stats_bundle:
            min_round = tag_stats_bundle.get("min_round")
            end_round = tag_stats_bundle.get("end_round")
            if min_round is not None and end_round is not None:
                lines.append(
                    f"- Population window: {min_round}-{end_round} "
                    f"(tag samples {sum(info.get('count', 0) for info in tag_stats_bundle.get('stats', {}).values())})"
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

        if streak >= 3:
            lines.append(
                f"- Streak alert: {streak}x repeat of "
                f"{self._format_signature(last_sig)}; a safe variation may help."
            )

        return "\n".join(lines)

    def _summarize_contextual_strategy(
        self,
        member_id: int,
        current_stats: Optional[dict],
        current_round_context: Optional[dict],
        current_relation_context: Optional[dict] = None,
        window: int = 10,
        min_similarity: float = 0.25,
        max_items: int = 3,
    ) -> str:
        """Summarize strategy signals from similar past contexts."""
        member_key = self._member_storage_key(member_id)
        memory = self.code_memory.get(member_key, []) if member_key is not None else []
        if not memory:
            return "Contextual strategy cues: no history yet."

        recent = memory[-max(1, int(window)):]
        buckets = {}
        similarity_values = []
        matched_entries = 0

        for mem in recent:
            context = mem.get("context", {}) or {}
            similarity = self._context_similarity_score(
                context,
                current_stats,
                current_round_context,
                current_relation_context,
            )
            if similarity is None:
                continue
            similarity_values.append(similarity)
            if similarity < min_similarity:
                continue
            signature = self._get_entry_signature(mem)
            if not signature:
                continue
            score = mem.get("balanced_score")
            if score is None:
                delta = context.get("delta") or {}
                perf = mem.get("performance", 0.0)
                score = self._compute_balanced_score(delta, perf)
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            bucket = buckets.setdefault(
                signature,
                {"weight": 0.0, "score": 0.0, "count": 0, "sim_sum": 0.0},
            )
            bucket["weight"] += similarity
            bucket["score"] += similarity * score
            bucket["count"] += 1
            bucket["sim_sum"] += similarity
            matched_entries += 1

        if not buckets:
            avg_sim = float(np.mean(similarity_values)) if similarity_values else 0.0
            return (
                "Contextual strategy cues: no comparable history; "
                f"avg similarity {avg_sim:.2f}."
            )

        stats = []
        total_weight = sum(bucket["weight"] for bucket in buckets.values())
        for signature, bucket in buckets.items():
            avg_score = bucket["score"] / max(1e-9, bucket["weight"])
            avg_sim = bucket["sim_sum"] / max(1, bucket["count"])
            stats.append(
                (avg_score, bucket["count"], avg_sim, signature, bucket["weight"])
            )
        stats.sort(key=lambda item: item[0], reverse=True)

        top = stats[:max(1, int(max_items))]
        avg_sim_all = float(np.mean([item[2] for item in stats])) if stats else 0.0
        lines = [
            "Contextual strategy cues (similar past contexts):",
            f"- matched samples: {matched_entries} across {len(buckets)} signatures",
            f"- avg similarity (matched): {avg_sim_all:.2f}",
            "- top candidates: "
            + ", ".join(
                f"{self._format_signature(sig)} "
                f"(avg {avg:.2f}, n={count}, sim {avg_sim:.2f})"
                for avg, count, avg_sim, sig, _ in top
            ),
        ]

        negatives = [item for item in stats if item[0] < 0.0]
        if negatives:
            avg, count, avg_sim, sig, _ = min(negatives, key=lambda item: item[0])
            lines.append(
                f"- caution: {self._format_signature(sig)} "
                f"(avg {avg:.2f}, n={count}, sim {avg_sim:.2f})"
            )

        low_sample = [sig for _, count, _, sig, _ in stats if count < 2]
        if low_sample:
            lines.append(
                "- low-sample cues: "
                + ", ".join(self._format_signature(sig) for sig in low_sample[:max_items])
            )

        if total_weight > 0:
            dominant_sig, dominant_weight = max(
                buckets.items(), key=lambda item: item[1]["weight"]
            )
            dominance = dominant_weight["weight"] / total_weight
            if dominance >= 0.6 and len(buckets) > 1:
                lines.append(
                    f"- diversity note: {dominance:.2f} of similar-context weight is "
                    f"{self._format_signature(dominant_sig)}; "
                    "consider a safe alternative to preserve diversity."
                )

        if avg_sim_all < 0.35:
            lines.append(
                "- similarity is weak; treat contextual cues as exploratory."
            )

        return "\n".join(lines)

    def _select_memory_samples(
        self,
        memory,
        max_samples: int,
        current_stats: Optional[dict] = None,
        current_round_context: Optional[dict] = None,
        current_relation_context: Optional[dict] = None,
    ):
        """Select a diversity-aware sample of memory entries."""
        if not memory:
            return []

        if len(memory) <= max_samples:
            return [("Recent", mem) for mem in memory]

        indices = list(range(len(memory)))
        signatures = self._get_memory_signatures(memory)
        signature_counts = Counter(signatures)
        entry_status = {idx: self._entry_execution_status(memory[idx]) for idx in indices}
        executed_indices = [idx for idx in indices if entry_status[idx] == "executed"]
        non_error_indices = [idx for idx in indices if entry_status[idx] != "error"]
        scoring_indices = executed_indices or non_error_indices or indices
        by_score = sorted(
            scoring_indices,
            key=lambda idx: self._memory_entry_score(memory[idx]),
        )

        recent_idx = indices[-1]
        best_idx = by_score[-1]
        worst_idx = by_score[0]
        median_idx = by_score[len(by_score) // 2]
        abs_idx = max(
            scoring_indices,
            key=lambda idx: abs(self._memory_entry_score(memory[idx])),
        )
        if executed_indices:
            executed_signature_counts = Counter(signatures[idx] for idx in executed_indices)
            rare_idx = min(
                executed_indices,
                key=lambda idx: executed_signature_counts.get(
                    signatures[idx], len(memory) + 1
                ),
            )
        else:
            rare_idx = min(
                indices,
                key=lambda idx: signature_counts.get(signatures[idx], len(memory) + 1),
            )

        context_idx = None
        if current_stats or current_round_context or current_relation_context:
            min_context_similarity = 0.25
            context_scores = {}
            context_candidates = executed_indices or indices
            for idx in context_candidates:
                score = self._context_similarity_score(
                    memory[idx].get("context", {}),
                    current_stats,
                    current_round_context,
                    current_relation_context,
                )
                if score is not None:
                    context_scores[idx] = score
            if context_scores:
                context_idx, best_score = max(
                    context_scores.items(), key=lambda item: item[1]
                )
                if best_score < min_context_similarity:
                    context_idx = None

        planned_only_idx = None
        planned_only_indices = [
            idx for idx in indices if entry_status[idx] == "planned_only"
        ]
        if planned_only_indices:
            planned_only_idx = max(
                planned_only_indices,
                key=lambda idx: memory[idx].get("context", {}).get("round", idx),
            )

        candidates = [("Recent", recent_idx)]
        if context_idx is not None:
            candidates.append(("Context", context_idx))
        candidates.extend([
            ("Best", best_idx),
            ("Worst", worst_idx),
            ("Rare", rare_idx),
            ("Volatile", abs_idx),
        ])
        if context_idx is None:
            candidates.append(("Median", median_idx))
        if planned_only_idx is not None:
            candidates.append(("Planned-only", planned_only_idx))

        selected = []
        seen = set()
        seen_signatures = set()
        for label, idx in candidates:
            if idx not in seen:
                sig = signatures[idx]
                if label not in ("Context", "Planned-only") and selected and sig in seen_signatures:
                    continue
                selected.append((label, idx))
                seen.add(idx)
                seen_signatures.add(sig)
                if len(selected) >= max_samples:
                    break

        if len(selected) < max_samples:
            remaining = [idx for idx in indices if idx not in seen]
            remaining_sorted = sorted(
                remaining,
                key=lambda idx: memory[idx].get('context', {}).get('round', idx),
                reverse=True,
            )

            for idx in remaining_sorted:
                if entry_status[idx] != "executed":
                    continue
                signature = signatures[idx]
                if signature not in seen_signatures:
                    selected.append(("Diverse", idx))
                    seen.add(idx)
                    seen_signatures.add(signature)
                    if len(selected) >= max_samples:
                        break

            if len(selected) < max_samples:
                for idx in remaining_sorted:
                    if idx in seen:
                        continue
                    selected.append(("Recent", idx))
                    seen.add(idx)
                    if len(selected) >= max_samples:
                        break

        return [(label, memory[idx]) for label, idx in selected]

    def get_code_memory_summary(
        self,
        member_id,
        current_stats: Optional[dict] = None,
        current_round_context: Optional[dict] = None,
        current_relation_context: Optional[dict] = None,
    ):
        """Generate a summary of previous code performances for the agent."""
        member_key = self._member_storage_key(member_id)
        if member_key is None or member_key not in self.code_memory:
            return "No previous code history."
            
        memory = self.code_memory[member_key]
        if not memory:
            return "No previous code history."
            
        summary = ["Previous code strategies and their outcomes (diversity-aware sample):"]

        selected = self._select_memory_samples(
            memory,
            max_samples=5,
            current_stats=current_stats,
            current_round_context=current_round_context,
            current_relation_context=current_relation_context,
        )

        for i, (label, mem) in enumerate(selected, start=1):
            perf = mem.get('performance', 0.0)
            context = mem.get('context', {})
            round_num = context.get('round')
            round_suffix = f", Round {round_num}" if round_num is not None else ""

            summary.append(f"\nStrategy {i} [{label}] (Performance: {perf:.2f}{round_suffix}):")
            delta = context.get("delta") or {}
            reward_score = mem.get("reward_score")
            survival_score = mem.get("survival_score")
            balanced_score = mem.get("balanced_score")
            if reward_score is None:
                reward_score = self._compute_reward_score(delta)
            if survival_score is None:
                survival_score = self._compute_survival_score(perf)
            if balanced_score is None:
                balanced_score = self._compute_balanced_score(delta, perf)
            summary.append(
                "Scorecard: "
                f"reward={reward_score:.2f}, "
                f"survival={survival_score:.2f}, "
                f"balanced={balanced_score:.2f}"
            )
            relative_balanced = (
                mem.get("relative_scores", {}) or {}
            ).get("balanced_score")
            if relative_balanced is not None:
                try:
                    relative_balanced = float(relative_balanced)
                except (TypeError, ValueError):
                    relative_balanced = None
            if relative_balanced is not None:
                summary.append(
                    f"Relative balanced vs population avg: {relative_balanced:+.2f}"
                )
            status = self._entry_execution_status(mem)
            status_labels = {
                "executed": "executed actions recorded",
                "planned_only": "planned-only (no executed actions recorded)",
                "idle": "idle (no planned/executed actions recorded)",
                "error": "error during execution",
            }
            if status != "unknown":
                summary.append(f"Execution status: {status_labels.get(status, status)}")
            if label == "Context":
                similarity = self._context_similarity_score(
                    context,
                    current_stats,
                    current_round_context,
                    current_relation_context,
                )
                if similarity is not None:
                    summary.append(f"Context similarity: {similarity:.2f}")
            old_stats = context.get("old_stats")
            new_stats = context.get("new_stats")
            if old_stats is not None or new_stats is not None:
                summary.append(f"Stats: old={old_stats}, new={new_stats}")

            if delta:
                summary.append(
                    "Outcome delta: "
                    f"vitality {delta.get('vitality', 0.0):.2f}, "
                    f"cargo {delta.get('cargo', 0.0):.2f}, "
                    f"land {delta.get('land', 0.0):.2f}, "
                    f"survival {delta.get('survival_chance', 0.0):.2f}"
                )

            action_summary = context.get("action_summary")
            if action_summary:
                summary.append(f"Action summary: {action_summary}")

            action_count = context.get("action_count")
            action_types = context.get("action_types")
            if action_count is not None:
                types_text = ", ".join(action_types) if action_types else "none"
                summary.append(f"Action count: {action_count}; types: {types_text}")

            round_context = context.get("round_context")
            if round_context:
                summary.append(f"Round context: {round_context}")

            relationship_context = context.get("relationship_context")
            if relationship_context:
                summary.append(f"Relationship context: {relationship_context}")

            message_summary = context.get("message_summary")
            if message_summary:
                summary.append(
                    "Messages: "
                    f"received {message_summary.get('received_count', 0)}, "
                    f"sent {message_summary.get('sent_count', 0)}"
                )
                received_senders = message_summary.get("received_senders") or {}
                if received_senders:
                    sender_text = ", ".join(
                        f"{sender}={count}" for sender, count in received_senders.items()
                    )
                    summary.append(f"Top senders: {sender_text}")
                received_intents = message_summary.get("received_intents") or {}
                if received_intents:
                    intent_text = ", ".join(
                        f"{intent}={count}" for intent, count in received_intents.items()
                    )
                    summary.append(f"Message intents: {intent_text}")
                received_sample = message_summary.get("received_sample") or []
                sent_sample = message_summary.get("sent_sample") or []
                if received_sample:
                    summary.append(f"Received sample: {received_sample}")
                if sent_sample:
                    summary.append(f"Sent sample: {sent_sample}")

            feasibility = context.get("feasibility")
            if feasibility:
                summary.append(f"Feasibility snapshot: {feasibility}")
            infeasible_tags = context.get("planned_infeasible_tags") or []
            if infeasible_tags:
                summary.append(
                    f"Planned but infeasible tags: {', '.join(infeasible_tags)}"
                )

            if not any([
                old_stats is not None,
                new_stats is not None,
                delta,
                action_summary,
                round_context,
                relationship_context,
                message_summary,
            ]) and context:
                summary.append(f"Context: {context}")

            executed_signature = self._get_executed_signature(mem)
            planned_signature = self._get_planned_signature(mem)
            if executed_signature:
                summary.append(
                    f"Action signature (executed): {', '.join(executed_signature)}"
                )
            if planned_signature and planned_signature != executed_signature:
                summary.append(
                    f"Action signature (planned): {', '.join(planned_signature)}"
                )
            if mem.get('signature_novelty') is not None:
                summary.append(f"Signature novelty: {mem['signature_novelty']:.2f}")

            strategy_notes = mem.get("strategy_notes") or []
            if strategy_notes:
                summary.append(
                    "Strategy notes: " + "; ".join(str(note) for note in strategy_notes)
                )

            summary.append("Code:")
            summary.append(mem.get('code', ''))
            if 'error' in mem:
                summary.append(f"Error encountered: {mem['error']}")
                
        return "\n".join(summary)

    # -- NEW METHOD TO REQUEST PYTHON CODE FROM GPT --
