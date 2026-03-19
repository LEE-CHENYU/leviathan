from typing import List, Tuple, Optional
from collections import Counter
import math
import numpy as np
from Leviathan.Member import Member




class IslandExecutionDiversityMixin:
    def _summarize_population_strategy_diversity(
        self,
        window_rounds: int = 3,
        top_k: int = 3,
    ) -> str:
        """Summarize population-level action signature diversity."""
        stats = self._compute_population_diversity_stats(window_rounds=window_rounds)
        if not stats:
            return "No population strategy data yet."

        min_round = stats["min_round"]
        end_round = stats["end_round"]
        total_actions = stats["total_actions"]
        total_entries = stats.get("total_entries", total_actions)
        idle_count = stats.get("idle_count", 0)
        idle_ratio = stats.get("idle_ratio", 0.0)
        planned_only_count = stats.get("planned_only_count", 0)
        active_agents = stats["active_agents"]
        signature_counts = stats["signature_counts"]
        tag_counts = stats["tag_counts"]
        unique_signatures = stats["unique_signatures"]
        diversity_ratio = stats["diversity_ratio"]
        dominant_sig = stats["dominant_signature"]
        dominant_share = stats["dominant_share"]
        signature_entropy = stats.get("signature_entropy", 0.0)
        signature_entropy_norm = stats.get("signature_entropy_norm", 0.0)
        known_tags = stats["known_tags"]
        underused_tags = stats["underused_tags"]
        coverage = sum(1 for tag in known_tags if tag_counts.get(tag, 0) > 0)

        top_signatures = signature_counts.most_common(min(top_k, len(signature_counts)))
        top_text = "; ".join(
            f"{self._format_signature(sig)} (n={count})"
            for sig, count in top_signatures
        )

        lines = [
            "Population strategy diversity snapshot:",
            f"- Window rounds: {min_round}-{end_round} "
            f"(entries {total_entries}, action sigs {total_actions}, agents {len(active_agents)})",
            f"- Idle/no-action entries: {idle_count} ({idle_ratio:.2f})",
            f"- Unique signatures: {unique_signatures} (ratio {diversity_ratio:.2f})",
            f"- Signature entropy: {signature_entropy:.2f} (norm {signature_entropy_norm:.2f})",
            f"- Tag coverage: {coverage}/{len(known_tags)}",
            f"- Dominant signature share: {dominant_share:.2f} "
            f"({self._format_signature(dominant_sig)})",
            f"- Top signatures: {top_text if top_text else 'none'}",
            f"- Underused tags: {', '.join(underused_tags) if underused_tags else 'none'}",
        ]
        if planned_only_count:
            lines.append(
                f"- Planned-only entries (no actions executed): {planned_only_count}"
            )

        if not signature_counts:
            lines.append(
                "- No executed action signatures yet; focus on at least one concrete action."
            )
            return "\n".join(lines)

        if dominant_share >= 0.6 and unique_signatures > 1:
            lines.append(
                "- Convergence risk: majority of actions share one signature; "
                "consider exploring underused tags."
            )

        return "\n".join(lines)

    def _compute_population_diversity_stats(
        self,
        window_rounds: int = 3,
    ) -> Optional[dict]:
        """Return structured diversity stats for population-level guidance."""
        if not self.code_memory or not self.execution_history.get('rounds'):
            return None

        current_round = len(self.execution_history['rounds'])
        end_round = max(0, current_round - 1)
        window_rounds = max(1, window_rounds)
        min_round = max(0, end_round - window_rounds + 1)

        entries = []
        total_entries = 0
        idle_count = 0
        planned_only_count = 0
        for member_id, memory in self.code_memory.items():
            for mem in memory:
                round_num = mem.get('context', {}).get('round')
                if round_num is None or round_num < min_round or round_num > end_round:
                    continue
                total_entries += 1
                executed_sig = self._get_executed_signature(mem)
                if executed_sig:
                    entries.append((member_id, round_num, tuple(executed_sig)))
                else:
                    idle_count += 1
                    planned_sig = self._get_planned_signature(mem)
                    if planned_sig:
                        planned_only_count += 1

        if not entries:
            if total_entries <= 0:
                return None
            return {
                "min_round": min_round,
                "end_round": end_round,
                "total_entries": total_entries,
                "total_actions": 0,
                "idle_count": idle_count,
                "idle_ratio": idle_count / max(1, total_entries),
                "planned_only_count": planned_only_count,
                "active_agents": [],
                "signature_counts": Counter(),
                "tag_counts": Counter(),
                "unique_signatures": 0,
                "diversity_ratio": 0.0,
                "dominant_signature": tuple(),
                "dominant_share": 0.0,
                "signature_entropy": 0.0,
                "signature_entropy_norm": 0.0,
                "known_tags": (
                    "attack",
                    "offer",
                    "offer_land",
                    "bear",
                    "expand",
                    "message",
                ),
                "tag_shares": {
                    "attack": 0.0,
                    "offer": 0.0,
                    "offer_land": 0.0,
                    "bear": 0.0,
                    "expand": 0.0,
                    "message": 0.0,
                },
                "underused_tags": [
                    "attack",
                    "offer",
                    "offer_land",
                    "bear",
                    "expand",
                    "message",
                ],
            }

        total_actions = len(entries)
        active_agents = sorted({member_id for member_id, _, _ in entries})

        signature_counts = Counter(sig for _, _, sig in entries)
        tag_counts = Counter()
        for _, _, sig in entries:
            tag_counts.update(set(sig))

        unique_signatures = len(signature_counts)
        diversity_ratio = unique_signatures / total_actions if total_actions else 0.0

        dominant_sig = tuple()
        dominant_share = 0.0
        if signature_counts:
            dominant_sig, dominant_count = signature_counts.most_common(1)[0]
            dominant_share = dominant_count / total_actions if total_actions else 0.0

        signature_probs = [count / total_actions for count in signature_counts.values()]
        signature_entropy = 0.0
        for prob in signature_probs:
            if prob > 0:
                signature_entropy -= prob * np.log(prob)
        max_entropy = np.log(unique_signatures) if unique_signatures > 1 else 0.0
        signature_entropy_norm = (
            signature_entropy / max_entropy if max_entropy > 0 else 0.0
        )

        known_tags = (
            "attack",
            "offer",
            "offer_land",
            "bear",
            "expand",
            "message",
        )
        tag_shares = {
            tag: (tag_counts.get(tag, 0) / total_actions if total_actions else 0.0)
            for tag in known_tags
        }
        underused_tags = []
        if total_actions <= len(known_tags):
            underused_tags = [tag for tag in known_tags if tag_counts.get(tag, 0) == 0]
        else:
            expected_share = 1.0 / len(known_tags)
            low_share = expected_share * 0.6
            for tag in known_tags:
                count = tag_counts.get(tag, 0)
                share = tag_shares[tag]
                if count == 0 or share < low_share:
                    underused_tags.append(tag)

        return {
            "min_round": min_round,
            "end_round": end_round,
            "total_entries": total_entries,
            "total_actions": total_actions,
            "idle_count": idle_count,
            "idle_ratio": idle_count / max(1, total_entries),
            "planned_only_count": planned_only_count,
            "active_agents": active_agents,
            "signature_counts": signature_counts,
            "tag_counts": tag_counts,
            "unique_signatures": unique_signatures,
            "diversity_ratio": diversity_ratio,
            "dominant_signature": dominant_sig,
            "dominant_share": dominant_share,
            "signature_entropy": signature_entropy,
            "signature_entropy_norm": signature_entropy_norm,
            "known_tags": known_tags,
            "tag_shares": tag_shares,
            "underused_tags": underused_tags,
        }

    def _update_diversity_controller(
        self,
        diversity_ratio: Optional[float],
        entropy_norm: Optional[float],
    ) -> Tuple[float, float]:
        """Adaptive controller to balance learning with diversity via EMA-like updates."""
        ctrl = getattr(self, "_diversity_controller", None) or {}
        alpha = float(ctrl.get("alpha", 0.25))
        target_diversity = float(ctrl.get("target_diversity", 0.45))
        target_entropy = float(ctrl.get("target_entropy", 0.6))
        min_adjust = float(ctrl.get("min_adjust", -0.15))
        max_adjust = float(ctrl.get("max_adjust", 0.25))

        weight = 0.0
        error_sum = 0.0
        if diversity_ratio is not None:
            error_sum += target_diversity - float(diversity_ratio)
            weight += 1.0
        if entropy_norm is not None:
            error_sum += 0.5 * (target_entropy - float(entropy_norm))
            weight += 0.5
        if weight <= 0:
            return 0.0, 0.0

        error = error_sum / weight
        adjustment = float(ctrl.get("adjustment", 0.0)) + alpha * error
        adjustment = max(min_adjust, min(max_adjust, adjustment))

        ctrl.update({"adjustment": adjustment, "last_error": error})
        self._diversity_controller = ctrl
        return adjustment, error

    def _compute_population_tag_pressure(
        self,
        window_rounds: int = 3,
        min_total: int = 8,
        max_bonus: float = 0.15,
    ) -> dict:
        """Compute soft tag-level pressure signals to preserve diversity."""
        stats = self._compute_population_diversity_stats(window_rounds=window_rounds)
        if not stats:
            return {
                "pressure": {},
                "dominant_signature": tuple(),
                "dominant_share": 0.0,
                "total_actions": 0,
                "signature_counts": Counter(),
                "signature_shares": {},
                "unique_signatures": 0,
            }

        total_actions = stats.get("total_actions", 0) or 0
        if total_actions <= 0:
            return {
                "pressure": {},
                "dominant_signature": stats.get("dominant_signature", tuple()),
                "dominant_share": stats.get("dominant_share", 0.0),
                "total_actions": 0,
                "signature_counts": Counter(),
                "signature_shares": {},
                "unique_signatures": 0,
            }

        known_tags = stats.get("known_tags") or ()
        if not known_tags:
            return {
                "pressure": {},
                "dominant_signature": stats.get("dominant_signature", tuple()),
                "dominant_share": stats.get("dominant_share", 0.0),
                "total_actions": total_actions,
                "signature_counts": stats.get("signature_counts") or Counter(),
                "signature_shares": {},
                "unique_signatures": int(stats.get("unique_signatures") or 0),
            }

        expected = 1.0 / max(1, len(known_tags))
        tag_shares = stats.get("tag_shares") or {}
        reliability = min(1.0, total_actions / max(1.0, float(min_total)))
        pressure = {}
        for tag in known_tags:
            share = float(tag_shares.get(tag, 0.0) or 0.0)
            delta = (expected - share) / max(expected, 1e-9)
            delta = max(-1.0, min(1.0, delta))
            pressure[tag] = max_bonus * reliability * delta

        signature_counts = stats.get("signature_counts") or Counter()
        unique_signatures = int(stats.get("unique_signatures") or len(signature_counts))
        signature_shares = {
            sig: (count / total_actions if total_actions else 0.0)
            for sig, count in signature_counts.items()
        }

        return {
            "pressure": pressure,
            "dominant_signature": stats.get("dominant_signature", tuple()),
            "dominant_share": stats.get("dominant_share", 0.0),
            "total_actions": total_actions,
            "signature_counts": signature_counts,
            "signature_shares": signature_shares,
            "unique_signatures": unique_signatures,
        }

    def _signature_diversity_bonus(self, signature: tuple, tag_pressure: dict) -> float:
        """Compute a soft bonus/penalty for a signature based on tag pressure."""
        if not signature or not tag_pressure:
            return 0.0
        values = [tag_pressure.get(tag, 0.0) for tag in signature if tag in tag_pressure]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _compute_reward_score(self, delta: Optional[dict]) -> float:
        """Compute a bounded reward score aligned with Member learning weights."""
        if not delta:
            return 0.0
        try:
            vitality = float(delta.get("vitality", 0.0))
        except (TypeError, ValueError):
            vitality = 0.0
        try:
            cargo = float(delta.get("cargo", 0.0))
        except (TypeError, ValueError):
            cargo = 0.0
        try:
            land = float(delta.get("land", 0.0))
        except (TypeError, ValueError):
            land = 0.0

        reward = (
            vitality * Member._REWARD_WEIGHTS[0]
            + cargo * Member._REWARD_WEIGHTS[1]
            + land * Member._REWARD_WEIGHTS[2]
        )
        scale = float(getattr(Member, "_REWARD_SCALE", 50.0))
        return float(np.tanh(reward / max(1.0, scale)))

    def _compute_survival_score(self, survival_delta: float) -> float:
        """Normalize survival change to a bounded score for comparisons."""
        try:
            delta = float(survival_delta)
        except (TypeError, ValueError):
            delta = 0.0
        scale = float(getattr(Member, "_REWARD_SCALE", 50.0))
        return float(np.tanh(delta / max(1.0, scale)))

    def _compute_balanced_score(self, delta: Optional[dict], survival_delta: float) -> float:
        """Blend reward- and survival-based signals to avoid single-metric overfit."""
        reward_score = self._compute_reward_score(delta)
        survival_score = self._compute_survival_score(survival_delta)
        return 0.6 * reward_score + 0.4 * survival_score

    def _apply_population_baseline(self, round_entries: list, round_data: dict) -> None:
        """Attach population-relative scores to memory entries and round logs."""
        if not round_entries:
            return

        metrics = ("reward_score", "survival_score", "balanced_score")
        baseline = {}
        for metric in metrics:
            values = [
                float(entry.get(metric))
                for entry in round_entries
                if entry.get(metric) is not None
            ]
            if not values:
                continue
            avg = float(np.mean(values))
            std = float(np.std(values)) if len(values) > 1 else 0.0
            baseline[metric] = {"avg": avg, "std": std, "count": len(values)}

        if not baseline:
            return

        round_data["population_baseline"] = baseline
        round_data.setdefault("relative_scores", {})

        for entry in round_entries:
            member_id = entry.get("member_id")
            memory_entry = entry.get("memory_entry")
            if memory_entry is None:
                continue
            relative = {}
            for metric, stats in baseline.items():
                value = entry.get(metric)
                if value is None:
                    continue
                relative[metric] = float(value) - stats["avg"]
            if relative:
                memory_entry["relative_scores"] = relative
                if member_id is not None:
                    round_data["relative_scores"][member_id] = relative

    def _memory_entry_score(self, memory_entry: dict) -> float:
        """Return a balanced score for memory sampling when available."""
        if not memory_entry:
            return 0.0
        balanced = None
        cached = memory_entry.get("balanced_score")
        if cached is not None:
            try:
                balanced = float(cached)
            except (TypeError, ValueError):
                balanced = None
        if balanced is None:
            delta = memory_entry.get("context", {}).get("delta") or {}
            perf = memory_entry.get("performance", 0.0)
            balanced = self._compute_balanced_score(delta, perf)
        relative = memory_entry.get("relative_scores", {}).get("balanced_score")
        if relative is None:
            return balanced
        try:
            relative = float(relative)
        except (TypeError, ValueError):
            return balanced
        relative_norm = math.tanh(relative)
        return 0.75 * balanced + 0.25 * relative_norm

    def _summarize_learning_trend(self, member_id: int, window: int = 5) -> str:
        """Summarize recent performance trend and novelty for adaptive learning."""
        member_key = self._member_storage_key(member_id)
        perf_history = (
            self.performance_history.get(member_key, []) if member_key is not None else []
        )
        if len(perf_history) < 2:
            return "Learning trend: insufficient data."

        recent = perf_history[-max(2, window):]
        avg_perf = float(np.mean(recent)) if recent else 0.0
        slope = float(recent[-1] - recent[0]) if len(recent) >= 2 else 0.0

        trend = "flat"
        if slope > 0.05:
            trend = "improving"
        elif slope < -0.05:
            trend = "declining"

        memory = (
            self.code_memory.get(member_key, []) if member_key is not None else []
        )
        novelty_values = [
            mem.get("signature_novelty")
            for mem in memory[-max(2, window):]
            if mem.get("signature_novelty") is not None
        ]
        novelty_avg = float(np.mean(novelty_values)) if novelty_values else None

        lines = [
            "Learning trend:",
            f"- recent avg performance: {avg_perf:.2f}; slope: {slope:.2f} ({trend})",
        ]
        if novelty_avg is not None:
            lines.append(f"- recent signature novelty avg: {novelty_avg:.2f}")

        streak, last_sig = self._signature_streak(
            member_id,
            prefer_executed=True,
            fallback_to_planned=False,
            include_empty=False,
        )
        idle_streak = self._idle_signature_streak(member_id)
        if trend == "flat" and streak >= 3:
            sig_text = self._format_signature(last_sig)
            lines.append(
                f"- stagnation signal: {streak}x repeat of ({sig_text}); "
                "consider a safe variation if situation allows."
            )
        if idle_streak >= 2:
            lines.append(
                f"- inactivity streak: {idle_streak}x rounds with no executed actions."
            )
        return "\n".join(lines)

    def _summarize_execution_reliability(self, member_id: int, window: int = 6) -> str:
        """Summarize how often planned actions actually execute."""
        member_key = self._member_storage_key(member_id)
        memory = self.code_memory.get(member_key, []) if member_key is not None else []
        if not memory:
            return "Execution reliability: no history yet."

        recent = memory[-max(1, int(window)):]
        counts = Counter()
        mismatch_count = 0
        match_count = 0

        for mem in recent:
            status = self._entry_execution_status(mem)
            counts[status] += 1
            planned_sig = self._get_planned_signature(mem)
            executed_sig = self._get_executed_signature(mem)
            if planned_sig and executed_sig:
                if tuple(planned_sig) == tuple(executed_sig):
                    match_count += 1
                else:
                    mismatch_count += 1

        total = len(recent)
        executed_ratio = counts.get("executed", 0) / max(1, total)
        planned_ratio = counts.get("planned_only", 0) / max(1, total)
        idle_ratio = counts.get("idle", 0) / max(1, total)
        error_ratio = counts.get("error", 0) / max(1, total)

        lines = [
            "Execution reliability (recent window):",
            f"- executed: {counts.get('executed', 0)}/{total} ({executed_ratio:.2f}); "
            f"planned-only: {counts.get('planned_only', 0)}/{total} ({planned_ratio:.2f}); "
            f"idle: {counts.get('idle', 0)}/{total} ({idle_ratio:.2f}); "
            f"errors: {counts.get('error', 0)}/{total} ({error_ratio:.2f})",
        ]

        paired_total = match_count + mismatch_count
        infeasible_only = 0
        for mem in recent:
            planned_sig = self._get_planned_signature(mem)
            if not planned_sig:
                continue
            infeasible = (
                mem.get("context", {}).get("planned_infeasible_tags") or []
            )
            if infeasible and set(infeasible) >= set(planned_sig):
                infeasible_only += 1
        infeasible_ratio = infeasible_only / max(1, total)
        if paired_total > 0:
            alignment = match_count / max(1, paired_total)
            lines.append(
                f"- plan/execution alignment: {match_count}/{paired_total} ({alignment:.2f}); "
                f"mismatches {mismatch_count}"
            )
        else:
            lines.append("- plan/execution alignment: n/a (no paired signatures)")

        guidance = []
        if executed_ratio < 0.5:
            guidance.append("low follow-through: add a low-constraint fallback action")
        if planned_ratio >= 0.3:
            guidance.append("many planned-only rounds: ensure at least one action triggers")
        if infeasible_ratio >= 0.3:
            guidance.append("planned actions often infeasible: check eligibility/targets")
        if paired_total > 0 and mismatch_count / max(1, paired_total) >= 0.3:
            guidance.append("mismatch risk: align triggers with intended actions")
        if error_ratio > 0:
            guidance.append("recent errors: harden guard clauses and index checks")
        if guidance:
            lines.append("- guidance: " + " | ".join(guidance))

        return "\n".join(lines)

    def _summarize_action_impact(
        self,
        window_rounds: int = 4,
        min_samples: int = 2,
    ) -> str:
        """Summarize recent action-tag correlations with outcomes."""
        stats_bundle = self._compute_action_tag_stats(window_rounds=window_rounds)
        if not stats_bundle:
            return "Action impact snapshot: no data yet."

        tags = stats_bundle["tags"]
        stats = stats_bundle["stats"]

        if not any(info["count"] > 0 for info in stats.values()):
            return "Action impact snapshot: no recent samples."

        lines = [
            "Action impact snapshot (recent rounds, correlation not causation):"
        ]
        low_sample = []
        for tag in tags:
            info = stats[tag]
            if info["count"] <= 0:
                lines.append(f"- {tag}: no recent samples")
                continue
            avg_reward = info["reward"] / info["count"]
            avg_survival = info["survival"] / info["count"]
            avg_balanced = info["balanced"] / info["count"]
            lines.append(
                f"- {tag}: n={info['count']}, reward={avg_reward:.2f}, "
                f"survival={avg_survival:.2f}, balanced={avg_balanced:.2f}"
            )
            if info["count"] < min_samples:
                low_sample.append(tag)

        if low_sample:
            lines.append(
                f"- low-sample tags (treat cautiously): {', '.join(low_sample)}"
            )

        return "\n".join(lines)

    def _summarize_personal_action_impact(
        self,
        member_id: int,
        window_rounds: int = 6,
        min_samples: int = 2,
    ) -> str:
        """Summarize member-specific action-tag outcome aggregates."""
        member_key = self._member_storage_key(member_id)
        memory = self.code_memory.get(member_key, []) if member_key is not None else []
        if not memory:
            return "Personal action impact: no history yet."

        current_round = len(self.execution_history.get("rounds", []))
        end_round = max(0, current_round - 1)
        min_round = max(0, end_round - max(1, window_rounds) + 1)

        tags = ("attack", "offer", "offer_land", "bear", "expand", "message")
        stats = {
            tag: {"count": 0, "reward": 0.0, "survival": 0.0, "balanced": 0.0}
            for tag in tags
        }

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

        if not any(info["count"] > 0 for info in stats.values()):
            return "Personal action impact: no recent samples."

        lines = ["Personal action impact (recent, member-specific):"]
        best_candidates = []
        low_sample = []
        for tag in tags:
            info = stats[tag]
            if info["count"] <= 0:
                continue
            count = info["count"]
            avg_reward = info["reward"] / count
            avg_survival = info["survival"] / count
            avg_balanced = info["balanced"] / count
            lines.append(
                f"- {tag}: n={count}, reward={avg_reward:.2f}, "
                f"survival={avg_survival:.2f}, balanced={avg_balanced:.2f}"
            )
            if count >= min_samples:
                best_candidates.append((avg_balanced, tag, count))
            else:
                low_sample.append(tag)

        if best_candidates:
            best = max(best_candidates, key=lambda item: item[0])
            worst = min(best_candidates, key=lambda item: item[0])
            lines.append(
                f"- strongest tag (balanced score): {best[1]} "
                f"(avg={best[0]:.2f}, n={best[2]})"
            )
            if worst[1] != best[1]:
                lines.append(
                    f"- weakest tag (balanced score): {worst[1]} "
                    f"(avg={worst[0]:.2f}, n={worst[2]})"
                )
        if low_sample:
            lines.append(f"- low-sample tags (treat cautiously): {', '.join(low_sample)}")

        return "\n".join(lines)

    def _summarize_experiment_log(
        self,
        member_id: int,
        window: int = 8,
        max_items: int = 3,
        novelty_threshold: float = 0.6,
    ) -> str:
        """Summarize recent strategy variations and their outcomes."""
        member_key = self._member_storage_key(member_id)
        memory = self.code_memory.get(member_key, []) if member_key is not None else []
        if not memory:
            return "Experiment log: no history yet."

        recent = memory[-max(1, int(window)):]
        experiments = []
        prev_sig = None
        for mem in recent:
            sig = self._get_entry_signature(
                mem,
                prefer_executed=True,
                fallback_to_planned=True,
            )
            status = self._entry_execution_status(mem)

            novelty_val = None
            if mem.get("signature_novelty") is not None:
                try:
                    novelty_val = float(mem.get("signature_novelty"))
                except (TypeError, ValueError):
                    novelty_val = None

            changed = bool(sig) and prev_sig is not None and tuple(sig) != tuple(prev_sig)
            is_experiment = False
            if novelty_val is not None and novelty_val >= novelty_threshold:
                is_experiment = True
            if changed:
                is_experiment = True

            if is_experiment:
                context = mem.get("context", {}) or {}
                delta = context.get("delta") or {}
                perf = mem.get("performance", 0.0)
                balanced = mem.get("balanced_score")
                if balanced is None:
                    balanced = self._compute_balanced_score(delta, perf)
                try:
                    balanced_val = float(balanced)
                except (TypeError, ValueError):
                    balanced_val = 0.0
                experiments.append(
                    {
                        "round": context.get("round"),
                        "sig": sig,
                        "status": status,
                        "novelty": novelty_val,
                        "balanced": balanced_val,
                        "delta": delta,
                    }
                )

            if sig:
                prev_sig = sig

        if not experiments:
            return "Experiment log: no recent variations."

        experiments = experiments[-max(1, int(max_items)):]
        positives = sum(1 for item in experiments if item.get("balanced", 0.0) > 0)
        avg_balanced = float(
            np.mean([item.get("balanced", 0.0) for item in experiments])
        )

        lines = ["Experiment log (recent variations):"]
        for item in experiments:
            round_num = item.get("round")
            round_label = f"r{round_num}" if round_num is not None else "r?"
            sig_text = self._format_signature(tuple(item.get("sig") or tuple()))
            status = item.get("status", "unknown")
            novelty_val = item.get("novelty")
            novelty_text = f"{novelty_val:.2f}" if novelty_val is not None else "n/a"
            delta = item.get("delta") or {}
            lines.append(
                f"- {round_label} sig={sig_text} ({status}) "
                f"novelty={novelty_text} balanced={item.get('balanced', 0.0):+.2f} "
                f"d_vit={delta.get('vitality', 0.0):+.1f} "
                f"d_cargo={delta.get('cargo', 0.0):+.1f} "
                f"d_land={delta.get('land', 0.0):+.1f}"
            )
        lines.append(
            f"- recent experiment avg balanced: {avg_balanced:+.2f} "
            f"({positives}/{len(experiments)} positive)"
        )
        return "\n".join(lines)

