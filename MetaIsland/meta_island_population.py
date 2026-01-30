from typing import Optional
import math
import numpy as np
from collections import defaultdict, Counter

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
        _, memory = self._get_member_history(self.code_memory, member_id)
        if not memory:
            return "No strategy recommendations available."
        window = max(1, window)
        recent = memory[-window:]
        if not recent:
            return "No strategy recommendations available."

        signature_counts = Counter()
        perf_by_sig = {}
        novelty_by_sig = {}
        for mem in recent:
            sig = mem.get('signature')
            if sig is None:
                sig = self._extract_action_signature(mem.get('code', ''))
            sig = tuple(sig) if sig else tuple()
            signature_counts[sig] += 1
            perf_by_sig.setdefault(sig, []).append(self._get_memory_performance(mem))
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
        if survival_tag == "fragile":
            risk_budget = 0.45
        elif survival_tag == "dominant":
            risk_budget = 0.95
        if relation_tag == "hostile":
            risk_budget -= 0.1
        elif relation_tag == "friendly":
            risk_budget += 0.05
        risk_budget = min(1.0, max(0.35, risk_budget))
        if risk_budget >= 0.85:
            risk_label = "high"
        elif risk_budget <= 0.55:
            risk_label = "low"
        else:
            risk_label = "medium"

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
        stable_candidates = [
            stat for stat in sig_perf_stats
            if stat[2] >= min_samples and stat[1] <= std_cutoff
        ]
        if stable_candidates:
            baseline_candidate = max(
                stable_candidates,
                key=lambda x: (x[0], -x[1], x[2])
            )
        elif sig_perf_stats:
            baseline_candidate = max(
                sig_perf_stats,
                key=lambda x: (x[0], -x[1], x[2])
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

        diversity_override = None
        if (
            baseline_sig is not None
            and pop_total
            and pop_dominant_sig is not None
            and baseline_sig == pop_dominant_sig
            and pop_dominant_share >= 0.6
            and risk_budget >= 0.6
        ):
            base_stats = sig_perf_lookup.get(baseline_sig)
            baseline_avg = base_stats[0] if base_stats else None
            perf_floor = baseline_avg - 0.03 if baseline_avg is not None else None
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
                dominance_penalty = 0.0
                if pop_dominant_sig is not None and pop_dominant_share >= 0.6:
                    if sig == pop_dominant_sig:
                        dominance_penalty = 0.15 + 0.15 * pop_dominant_share
                adjusted = ucb - distance_penalty - dominance_penalty
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
            if base_stats:
                base_avg, base_std, base_count, _ = base_stats
            elif best_by_avg is not None and best_by_avg[2] == baseline_sig:
                base_avg, base_count, _ = best_by_avg
            baseline_label = "stable"
            if baseline_reason == "context":
                baseline_label = "context-aligned"
            elif baseline_reason == "average":
                baseline_label = "average"
            elif baseline_reason == "diversity":
                baseline_label = "diversity"
            detail_suffix = ""
            if baseline_reason == "context" and context_candidate is not None:
                ctx_avg, ctx_weight, ctx_count, _ctx_sim, ctx_sig = context_candidate
                if ctx_sig == baseline_sig:
                    detail_suffix = (
                        f", ctx avg {ctx_avg:.2f}, weight {ctx_weight:.2f}, n={ctx_count}"
                    )
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
            lines.append(
                f"- Risk-adjusted variation: {self._format_signature(sig)} "
                f"({', '.join(detail_bits)})"
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

        if dominant_sig is not None and dominant_share >= 0.6 and len(signature_counts) > 1:
            lines.append(
                f"- Diversity guard: {dominant_share:.2f} of recent actions share "
                f"{self._format_signature(dominant_sig)}; consider mixing tags."
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


