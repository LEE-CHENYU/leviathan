from typing import Optional
import math
import numpy as np
from collections import defaultdict, Counter
from MetaIsland.strategy_recommendations import build_strategy_recommendations

class IslandExecutionPopulationMixin:
    def _get_latest_total(self, series) -> float:
        """Return the latest numeric total from a list-like series."""
        if isinstance(series, list):
            if not series:
                return 0.0
            try:
                return float(series[-1])
            except (TypeError, ValueError):
                return 0.0
        try:
            return float(series)
        except (TypeError, ValueError):
            return 0.0

    def _get_record_total(self, key: str) -> float:
        """Fetch the latest total for a record_total_dict key."""
        totals = getattr(self, "record_total_dict", {})
        if not isinstance(totals, dict):
            return 0.0
        return self._get_latest_total(totals.get(key, 0.0))

    def _get_latest_round_end_metrics(self) -> Optional[dict]:
        """Return the most recent round_end_metrics entry, if any."""
        rounds = []
        if hasattr(self, "execution_history"):
            rounds = self.execution_history.get("rounds", []) or []
        if not rounds:
            return None
        for record in reversed(rounds):
            metrics = record.get("round_end_metrics")
            if isinstance(metrics, dict) and metrics:
                return metrics
        return None

    def _get_latest_round_metrics(self) -> Optional[dict]:
        """Return the most recent round_metrics entry, if any."""
        rounds = []
        if hasattr(self, "execution_history"):
            rounds = self.execution_history.get("rounds", []) or []
        if not rounds:
            return None
        for record in reversed(rounds):
            metrics = record.get("round_metrics")
            if isinstance(metrics, dict) and metrics:
                return metrics
        return None

    def _compute_gini(self, values: list) -> float:
        """Compute Gini coefficient for a list of values."""
        if not values:
            return 0.0
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        min_val = float(np.min(arr))
        if min_val < 0:
            arr = arr - min_val
        total = float(np.sum(arr))
        if total <= 0:
            return 0.0
        arr = np.sort(arr)
        n = arr.size
        cum = np.cumsum(arr)
        gini = (n + 1 - 2 * float(np.sum(cum)) / total) / n
        return float(max(0.0, min(1.0, gini)))

    def get_population_state_summary(self, top_k: int = 3) -> str:
        """Summarize current population state for strategic grounding."""
        members = list(self.current_members) if hasattr(self, "current_members") else []
        if not members:
            return "No population data available."

        member_count = len(members)
        cargo_vals = [float(m.cargo) for m in members]
        land_vals = [float(m.land_num) for m in members]
        vitality_vals = [float(m.vitality) for m in members]
        survival_vals = [float(self.compute_survival_chance(m)) for m in members]
        relation_vals = [float(self.compute_relation_balance(m)) for m in members]

        def _mean_std(values):
            if not values:
                return 0.0, 0.0
            arr = np.array(values, dtype=float)
            return float(np.mean(arr)), float(np.std(arr))

        cargo_mean, cargo_std = _mean_std(cargo_vals)
        land_mean, land_std = _mean_std(land_vals)
        vitality_mean, vitality_std = _mean_std(vitality_vals)
        relation_mean, relation_std = _mean_std(relation_vals)

        survival_min = float(np.min(survival_vals)) if survival_vals else 0.0
        survival_mean = float(np.mean(survival_vals)) if survival_vals else 0.0
        survival_max = float(np.max(survival_vals)) if survival_vals else 0.0

        land_total = float(np.prod(self.land.shape)) if hasattr(self, "land") else 0.0
        land_owned = float(sum(land_vals))
        land_scarcity = land_owned / land_total if land_total else 0.0

        gini_cargo = self._compute_gini(cargo_vals)
        gini_land = self._compute_gini(land_vals)
        wealth_vals = [c + l for c, l in zip(cargo_vals, land_vals)]
        gini_wealth = self._compute_gini(wealth_vals)

        action_totals = None
        if self.execution_history.get('rounds'):
            action_totals = self.execution_history['rounds'][-1].get('action_totals')

        if action_totals:
            attack_total = float(action_totals.get("attack", 0.0))
            benefit_total = float(action_totals.get("benefit", 0.0))
            benefit_land_total = float(action_totals.get("benefit_land", 0.0))
        else:
            attack_total = self._get_record_total("attack")
            benefit_total = self._get_record_total("benefit")
            benefit_land_total = self._get_record_total("benefit_land")

        attack_rate = attack_total / max(1, member_count)
        benefit_rate = benefit_total / max(1, member_count)
        benefit_land_rate = benefit_land_total / max(1, member_count)

        production = self._get_latest_total(getattr(self, "record_total_production", 0.0))
        consumption = self._get_latest_total(getattr(self, "record_total_consumption", 0.0))
        net_production = production - consumption

        births = len(getattr(self, "record_born", []))
        deaths = len(getattr(self, "record_death", []))

        top_k = max(0, int(top_k))
        top_holders = sorted(
            members,
            key=lambda m: (m.cargo + m.land_num),
            reverse=True
        )[:top_k]
        if top_holders:
            top_text = ", ".join(
                f"member_{m.id}({m.cargo:.1f}+{m.land_num})"
                for m in top_holders
            )
        else:
            top_text = "none"

        sent_total = 0
        received_total = 0
        senders = set()
        recipients = set()
        if self.execution_history.get('rounds'):
            last_round = self.execution_history['rounds'][-1]
            for comm in (last_round.get('agent_messages', {}) or {}).values():
                for recipient_id, _msg in comm.get('sent', []) or []:
                    sent_total += 1
                    if recipient_id is not None:
                        recipients.add(recipient_id)
                for msg in comm.get('received', []) or []:
                    received_total += 1
                    sender_id = self._parse_message_sender(msg)
                    if sender_id is not None:
                        senders.add(sender_id)

        if sent_total or received_total:
            communication_line = (
                f"- Communication last round: sent {sent_total} "
                f"(to {len(recipients)}), received {received_total} "
                f"(from {len(senders)})"
            )
        else:
            communication_line = "- Communication last round: no messages"

        contract_line = "- Contracts: unavailable"
        if hasattr(self, "contracts"):
            stats = self.contracts.get_statistics()
            contract_line = (
                "- Contracts (total/pending/active/completed/failed): "
                f"{stats.get('total_contracts', 0)}/"
                f"{stats.get('pending', 0)}/"
                f"{stats.get('active', 0)}/"
                f"{stats.get('completed', 0)}/"
                f"{stats.get('failed', 0)}"
            )

        physics_line = "- Physics: unavailable"
        if hasattr(self, "physics"):
            stats = self.physics.get_statistics()
            domains = stats.get("domains", [])
            domain_text = ", ".join(sorted(domains)) if domains else "none"
            physics_line = (
                f"- Physics constraints: active {stats.get('active_constraints', 0)} "
                f"(domains: {domain_text})"
            )

        lines = [
            "Population state snapshot:",
            f"- Members: {member_count}; land scarcity: "
            f"{land_scarcity:.2f} ({land_owned:.0f}/{land_total:.0f})",
            f"- Survival chance (min/avg/max): "
            f"{survival_min:.2f}/{survival_mean:.2f}/{survival_max:.2f}",
            f"- Vitality avg/std: {vitality_mean:.2f}/{vitality_std:.2f}; "
            f"Cargo avg/std: {cargo_mean:.2f}/{cargo_std:.2f}; "
            f"Land avg/std: {land_mean:.2f}/{land_std:.2f}",
            f"- Relation balance avg/std: {relation_mean:.2f}/{relation_std:.2f}",
            f"- Inequality (Gini): cargo {gini_cargo:.2f}, land {gini_land:.2f}, "
            f"wealth {gini_wealth:.2f}",
            f"- Interaction rates per member (recent): attack {attack_rate:.2f}, "
            f"benefit {benefit_rate:.2f}, land benefit {benefit_land_rate:.2f}",
            f"- Production/consumption: {production:.1f}/{consumption:.1f} "
            f"(net {net_production:.1f}); births {births}, deaths {deaths}",
            f"- Top holders (cargo+land): {top_text}",
            contract_line,
            physics_line,
            communication_line,
        ]

        return "\n".join(lines)

    def get_strategy_profile_summary(self, member_id: int, window: int = 5) -> str:
        """Summarize action signature coverage to encourage strategy diversity."""
        _, memory = self._get_member_history(self.code_memory, member_id)
        if not memory:
            return "No strategy profile available."
        signatures = self._get_memory_signatures(memory)

        tag_counts = Counter()
        for sig in signatures:
            tag_counts.update(sig)

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

        total_entries = len(memory)
        recent_window = min(max(window, 1), total_entries)
        recent_signatures = signatures[-recent_window:]
        recent_tags = sorted({tag for sig in recent_signatures for tag in sig})
        recent_signature_counts = Counter(recent_signatures)
        recent_unique = len(recent_signature_counts)
        novelty_ratio = (recent_unique / recent_window) if recent_window else 0.0
        recent_tag_set = set(recent_tags)

        entropy = 0.0
        if recent_signature_counts and recent_window:
            for count in recent_signature_counts.values():
                p = count / float(recent_window)
                entropy -= p * math.log2(p)

        normalized_entropy = 0.0
        if recent_unique > 1:
            normalized_entropy = entropy / math.log2(recent_unique)

        streak = 0
        last_sig = tuple()
        if recent_signatures:
            last_sig = recent_signatures[-1]
            streak = 1
            for sig in reversed(recent_signatures[:-1]):
                if sig == last_sig:
                    streak += 1
                else:
                    break

        recent_tag_counts = Counter()
        for sig in recent_signatures:
            recent_tag_counts.update(set(sig))
        overused_tags = [
            tag for tag in known_tags
            if recent_tag_counts.get(tag, 0) >= max(1, int(0.6 * recent_window))
        ]

        coverage = sum(1 for tag in known_tags if tag_counts.get(tag, 0) > 0)
        underused = [tag for tag in known_tags if tag_counts.get(tag, 0) == 0]
        if not underused:
            underused = [tag for tag in known_tags if tag_counts.get(tag, 0) <= 1]
        historical_tags = {tag for tag in known_tags if tag_counts.get(tag, 0) > 0}
        stale_tags = sorted(historical_tags - recent_tag_set)

        recent_context_keys = []
        missing_context = 0
        for mem in memory[-recent_window:]:
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

        lines = [
            "Strategy profile (action signature coverage):",
            f"- Total entries: {total_entries}",
            f"- Tag coverage: {coverage}/{len(known_tags)}",
            f"- Recent tags (last {recent_window}): {', '.join(recent_tags) if recent_tags else 'none'}",
            f"- Stale tags (used before, absent recently): {', '.join(stale_tags) if stale_tags else 'none'}",
            f"- Recent signature diversity: {recent_unique}/{recent_window} (ratio {novelty_ratio:.2f}, entropy {entropy:.2f}, norm {normalized_entropy:.2f})",
            (
                f"- Recent context coverage: {context_unique}/{context_total} "
                f"(dominant {dominant_context_share:.2f} in {dominant_context})"
                if context_total else
                (f"- Recent context coverage: none (missing {missing_context} tags)"
                 if missing_context else "- Recent context coverage: none")
            ),
            f"- Recent signature streak: {streak}x ({', '.join(last_sig) if last_sig else 'none'})",
            f"- Underused tags: {', '.join(underused) if underused else 'none'}",
            f"- Overused tags (recent): {', '.join(overused_tags) if overused_tags else 'none'}",
            "- Tag counts: " + ", ".join(f"{tag}={tag_counts.get(tag, 0)}" for tag in known_tags),
        ]

        if context_counts and context_unique > 1 and dominant_context_share >= 0.7:
            lines.append(
                "- Context concentration: recent actions cluster in one context; "
                "consider testing a different context signature when safe."
            )

        return "\n".join(lines)

    def get_population_strategy_summary(
        self,
        window_rounds: int = 3,
        top_k: int = 3
    ) -> str:
        """Summarize population-level action signature diversity."""
        if not self.code_memory or not self.execution_history.get('rounds'):
            return "No population strategy data yet."

        current_round = len(self.execution_history['rounds'])
        window_rounds = max(1, window_rounds)
        min_round = max(1, current_round - window_rounds + 1)

        entries = []
        context_keys = []
        missing_context = 0
        for member_id, memory in self.code_memory.items():
            for mem in memory:
                round_num = mem.get('context', {}).get('round')
                if round_num is None or round_num < min_round:
                    continue
                sig = mem.get('signature')
                if sig is None:
                    sig = self._extract_action_signature(mem.get('code', ''))
                sig = tuple(sig) if sig else tuple()
                entries.append((member_id, round_num, sig))
                context_key = self._get_memory_context_key(mem)
                if context_key:
                    context_keys.append(context_key)
                else:
                    missing_context += 1

        if not entries:
            return "No recent population strategy data."

        total_actions = len(entries)
        active_agents = sorted({member_id for member_id, _, _ in entries})

        signature_counts = Counter(sig for _, _, sig in entries)
        tag_counts = Counter()
        for _, _, sig in entries:
            tag_counts.update(sig)

        unique_signatures = len(signature_counts)
        diversity_ratio = unique_signatures / total_actions if total_actions else 0.0

        dominant_sig, dominant_count = signature_counts.most_common(1)[0]
        dominant_share = dominant_count / total_actions if total_actions else 0.0

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

        coverage = sum(1 for tag in known_tags if tag_counts.get(tag, 0) > 0)
        underused_tags = [tag for tag in known_tags if tag_counts.get(tag, 0) == 0]

        top_signatures = signature_counts.most_common(min(top_k, len(signature_counts)))
        top_text = "; ".join(
            f"{self._format_signature(sig)} (n={count})"
            for sig, count in top_signatures
        )

        context_counts = Counter(context_keys)
        context_total = len(context_keys)
        context_unique = len(context_counts)
        dominant_context = None
        dominant_context_share = 0.0
        if context_counts:
            dominant_context, dominant_count = context_counts.most_common(1)[0]
            dominant_context_share = dominant_count / context_total if context_total else 0.0

        lines = [
            "Population strategy diversity snapshot:",
            f"- Window rounds: {min_round}-{current_round} "
            f"(actions {total_actions}, agents {len(active_agents)})",
            f"- Unique signatures: {unique_signatures} (ratio {diversity_ratio:.2f})",
            f"- Tag coverage: {coverage}/{len(known_tags)}",
            f"- Dominant signature share: {dominant_share:.2f} "
            f"({self._format_signature(dominant_sig)})",
            f"- Top signatures: {top_text if top_text else 'none'}",
            f"- Underused tags: {', '.join(underused_tags) if underused_tags else 'none'}",
            (
                f"- Context coverage: {context_unique} contexts "
                f"(dominant {dominant_context_share:.2f} in {dominant_context})"
                if context_total else
                (f"- Context coverage: none (missing {missing_context} tags)"
                 if missing_context else "- Context coverage: none")
            ),
        ]

        if dominant_share >= 0.6 and unique_signatures > 1:
            lines.append(
                "- Convergence risk: majority of actions share one signature; "
                "consider exploring underused tags."
            )
        if context_counts and context_unique > 1 and dominant_context_share >= 0.7:
            lines.append(
                "- Context concentration risk: population actions cluster in one context; "
                "maintain fallback plans for shifts."
            )

        return "\n".join(lines)

    def get_population_exploration_summary(
        self,
        window_rounds: int = 3,
        min_samples: int = 2,
        underuse_share: float = 0.15,
        max_items: int = 4
    ) -> str:
        """Summarize population tag performance to encourage exploration."""
        if not self.code_memory or not self.execution_history.get('rounds'):
            return "No population exploration data yet."

        current_round = len(self.execution_history['rounds'])
        window_rounds = max(1, window_rounds)
        min_round = max(1, current_round - window_rounds + 1)

        entries = []
        for member_id, memory in self.code_memory.items():
            for mem in memory:
                round_num = mem.get('context', {}).get('round')
                if round_num is None or round_num < min_round:
                    continue
                sig = mem.get('signature')
                if sig is None:
                    sig = self._extract_action_signature(mem.get('code', ''))
                sig = tuple(sig) if sig else tuple()
                perf = self._get_memory_performance(mem)
                entries.append((sig, perf))

        if not entries:
            return "No recent population exploration data."

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

        tag_perf = {tag: [] for tag in known_tags}
        for sig, perf in entries:
            for tag in sig:
                if tag not in tag_perf:
                    tag_perf[tag] = []
                tag_perf[tag].append(perf)

        total_actions = len(entries)
        low_threshold = max(1, int(round(underuse_share * total_actions)))

        tag_stats = []
        for tag in known_tags:
            perfs = tag_perf.get(tag, [])
            count = len(perfs)
            avg = sum(perfs) / count if count else 0.0
            tag_stats.append((tag, count, avg))

        def _format_tag_stats(stats):
            if not stats:
                return "none"
            return ", ".join(
                f"{tag} (avg {avg:.2f}, n={count})"
                for avg, count, tag in stats[:max_items]
            )

        underused = [
            (tag, count) for tag, count, _ in tag_stats if count < low_threshold
        ]
        untried = [tag for tag, count in underused if count == 0]
        underused_text = ", ".join(
            f"{tag} (n={count})" for tag, count in underused[:max_items]
        ) if underused else "none"
        untried_text = ", ".join(untried[:max_items]) if untried else "none"

        positive = sorted(
            [(avg, count, tag) for tag, count, avg in tag_stats
             if count >= min_samples and avg > 0.0],
            reverse=True
        )
        negative = sorted(
            [(avg, count, tag) for tag, count, avg in tag_stats
             if count >= min_samples and avg < 0.0]
        )

        lines = [
            "Population exploration signals (coarse):",
            f"- Window rounds: {min_round}-{current_round} (actions {total_actions})",
            f"- Underused tags (<{low_threshold} uses): {underused_text}",
            f"- Untried tags: {untried_text}",
            f"- Positive tags (avg delta_survival > 0, n >= {min_samples}): "
            f"{_format_tag_stats(positive)}",
            f"- Negative tags (avg delta_survival < 0, n >= {min_samples}): "
            f"{_format_tag_stats(negative)}",
        ]

        return "\n".join(lines)

    def get_contextual_strategy_summary(
        self,
        member_id: int,
        window: int = 12,
        min_samples: int = 2,
        max_contexts: int = 3,
        max_signatures: int = 2
    ) -> str:
        """Summarize recent performance by coarse context tags."""
        _, memory = self._get_member_history(self.code_memory, member_id)
        if not memory:
            return "No contextual strategy data yet."
        window = max(1, window)
        recent = memory[-window:]
        if not recent:
            return "No contextual strategy data yet."

        current_tags = self._get_member_context_tags(member_id)
        current_key = self._context_key_from_tags(current_tags)

        context_entries = []
        for mem in recent:
            context_key = mem.get('context_key')
            if not context_key:
                context_key = self._context_key_from_tags(mem.get('context_tags', {}))
            if not context_key:
                continue
            context_entries.append((context_key, mem))

        if not context_entries:
            return "No contextual strategy data yet."

        by_context = defaultdict(list)
        for context_key, mem in context_entries:
            by_context[context_key].append(mem)

        total = len(context_entries)
        lines = [
            "Contextual strategy cues (recent window):",
            f"- Window size: {total} actions, contexts {len(by_context)}",
        ]
        if current_tags:
            if current_key:
                lines.append(
                    f"- Current context: {current_key} "
                    f"({self._format_context_tags(current_tags)})"
                )
            else:
                lines.append(
                    f"- Current context tags: {self._format_context_tags(current_tags)}"
                )

        dominant_ctx, dominant_mems = max(
            by_context.items(),
            key=lambda item: len(item[1])
        )
        dominant_share = len(dominant_mems) / total if total else 0.0
        if dominant_share >= 0.6 and len(by_context) > 1:
            lines.append(
                f"- Context concentration: {dominant_share:.2f} in {dominant_ctx}; "
                "prepare fallback plans for shifts."
            )

        for context_key, mems in sorted(
            by_context.items(),
            key=lambda item: len(item[1]),
            reverse=True
        )[:max_contexts]:
            sig_perf = defaultdict(list)
            for mem in mems:
                sig = mem.get('signature')
                if sig is None:
                    sig = self._extract_action_signature(mem.get('code', ''))
                sig = tuple(sig) if sig else tuple()
                sig_perf[sig].append(self._get_memory_performance(mem))

            sig_stats = []
            for sig, perfs in sig_perf.items():
                count = len(perfs)
                avg = sum(perfs) / count if count else 0.0
                sig_stats.append((avg, count, sig))
            sig_stats.sort(key=lambda x: (x[0], x[1]), reverse=True)

            top = sig_stats[:max_signatures]
            if top:
                top_text = "; ".join(
                    f"{self._format_signature(sig)} (avg {avg:.2f}, n={count})"
                    for avg, count, sig in top
                )
            else:
                top_text = "none"

            sample_note = " low-sample" if len(mems) < min_samples else ""
            context_line = f"- {context_key} (n={len(mems)}): {top_text}{sample_note}"
            if current_key and context_key == current_key:
                context_line += " [current]"
            lines.append(context_line)

        if current_key and current_key not in by_context and current_tags:
            match_idx, match_score = self._find_contextual_memory_match(memory, current_tags)
            if match_idx is not None and match_score > 0:
                match_mem = memory[match_idx]
                match_context = match_mem.get('context_key') or self._context_key_from_tags(
                    match_mem.get('context_tags', {})
                )
                if match_context:
                    lines.append(
                        f"- Nearest observed context: {match_context} "
                        f"(match {match_score:.2f})"
                    )

        return "\n".join(lines)

    def get_strategy_recommendations(
        self,
        member_id: int,
        window: int = 8,
        min_samples: int = 2,
        exploration_bonus: float = 0.6,
        population_window_rounds: int = 3,
        max_items: int = 3
    ) -> str:
        """Provide lightweight decision support without forcing convergence."""
        return build_strategy_recommendations(
            self,
            member_id,
            window=window,
            min_samples=min_samples,
            exploration_bonus=exploration_bonus,
            population_window_rounds=population_window_rounds,
            max_items=max_items,
        )
