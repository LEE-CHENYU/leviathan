from typing import List, Optional, Tuple
import ast
import math
import re
import numpy as np
from collections import defaultdict, Counter

from utils.error_tags import classify_error_tag

class IslandExecutionSignatureMixin:
    def _format_signature(self, signature: tuple) -> str:
        """Format a signature tuple for readable summaries."""
        if not signature:
            return "none"
        return ", ".join(signature)

    def _normalize_action_tags(self, tags: List[str]) -> tuple:
        """Normalize a list of action tags into a canonical signature tuple."""
        if not tags:
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
        aliases = {
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
        offer_land_pattern = re.compile(r"\boffer(?:[_\-\s]+)land\b")

        def _extract_from_text(text: str) -> set:
            if not text:
                return set()
            lowered = text.lower()
            found = set()
            offer_land_matches = list(offer_land_pattern.finditer(lowered))
            if offer_land_matches:
                found.add("offer_land")
            tokens = re.findall(r"[a-z_]+", lowered)
            for token in tokens:
                token = token.strip("_")
                if not token:
                    continue
                token = aliases.get(token, token)
                if token in tag_order:
                    found.add(token)
            if "offer_land" in found and "offer" in found and offer_land_matches:
                offer_matches = list(re.finditer(r"\boffer\b", lowered))
                if offer_matches:
                    inside = sum(
                        1
                        for match in offer_matches
                        if any(
                            land_match.start() <= match.start() < land_match.end()
                            for land_match in offer_land_matches
                        )
                    )
                    if inside == len(offer_matches):
                        found.discard("offer")
            return found

        seen = set()
        for raw in tags:
            if raw is None:
                continue
            raw_text = str(raw).strip()
            if not raw_text:
                continue
            normalized = raw_text.lower().replace(" ", "_")
            normalized = aliases.get(normalized, normalized)
            if normalized in tag_order:
                seen.add(normalized)
                continue
            extracted = _extract_from_text(raw_text)
            if extracted:
                seen.update(extracted)
        if not seen:
            return tuple()
        return tuple([tag for tag in tag_order if tag in seen])

    def _score_signature_match(self, signature: tuple, target: tuple) -> float:
        """Score how closely an action signature matches a target tag list."""
        if not signature or not target:
            return 0.0
        sig_set = set(signature)
        target_set = set(target)
        if not sig_set or not target_set:
            return 0.0
        if sig_set == target_set:
            return 1.0
        coverage = len(sig_set & target_set) / float(len(target_set))
        penalty = len(sig_set - target_set) / float(len(sig_set)) if sig_set else 0.0
        score = coverage - 0.25 * penalty
        return float(max(0.0, min(1.0, score)))

    def _get_latest_analysis_card(
        self,
        member_id: int,
        max_back: int = 3,
    ) -> Tuple[Optional[int], Optional[dict]]:
        """Return the most recent analysis card for a member."""
        rounds = self.execution_history.get("rounds", [])
        if not rounds:
            return None, None
        max_back = max(1, int(max_back))
        for round_data in reversed(rounds[-max_back:]):
            card = self._get_round_member_entry(round_data, "analysis_cards", member_id)
            if card:
                round_num = round_data.get("round_number")
                return round_num, card
        return None, None

    def _record_experiment_outcome(
        self,
        member_id: int,
        round_num: int,
        signature: tuple,
        metrics: dict,
        context_key: str = "",
    ) -> Optional[dict]:
        """Record how the executed signature aligns with the latest analysis plan."""
        plan_round, card = self._get_latest_analysis_card(member_id)
        if not card:
            return None

        baseline_sig = self._normalize_action_tags(card.get("baseline_signature"))
        variation_sig = self._normalize_action_tags(card.get("variation_signature"))
        baseline_score = self._score_signature_match(signature, baseline_sig)
        variation_score = self._score_signature_match(signature, variation_sig)

        label = "unmatched"
        match_score = 0.0
        if baseline_score or variation_score:
            if baseline_score >= variation_score:
                label = "baseline"
                match_score = baseline_score
            else:
                label = "variation"
                match_score = variation_score

        experiment = {
            "plan_round": plan_round,
            "label": label,
            "match_score": match_score,
            "baseline_signature": baseline_sig,
            "variation_signature": variation_sig,
            "success_metrics": card.get("success_metrics", []),
            "guardrails": card.get("guardrails", []),
            "confidence": card.get("confidence"),
            "context_key": context_key or "",
        }

        try:
            member_index = self._resolve_member_index(member_id)
            member = (
                self.current_members[member_index]
                if member_index is not None
                else None
            )
        except Exception:
            member = None

        delta_survival = metrics.get("delta_survival", 0.0)
        note = (
            f"exp r{round_num} plan_r{plan_round or '?'} "
            f"{label} match={match_score:.2f} "
            f"d_surv={float(delta_survival):.2f} "
            f"base={self._format_signature(baseline_sig)} "
            f"var={self._format_signature(variation_sig)}"
        )
        note = self._truncate_message(note, limit=180)
        self._append_strategy_report(member, note, key="experiments")

        return experiment

    def _extract_action_signature(self, code_str: str) -> tuple:
        """Extract a coarse action signature from generated code for diversity sampling."""
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
                            if (isinstance(value.value, ast.Name)
                                and value.value.id in engine_aliases
                                and value.attr == "send_message"):
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
                            if (isinstance(func.value.value, ast.Name)
                                and func.value.value.id in engine_aliases):
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

    def _get_memory_signatures(self, memory: list) -> list:
        """Normalize action signatures from memory entries."""
        signatures = []
        for mem in memory:
            sig = mem.get('signature')
            if sig is None:
                sig = self._extract_action_signature(mem.get('code', ''))
            sig = tuple(sig) if sig else tuple()
            signatures.append(sig)
        return signatures

    def _collect_round_signature_stats(self, round_num: int) -> dict:
        """Collect population signature stats for a given round."""
        round_signatures = {}
        for member_id, mem_list in self.code_memory.items():
            if not mem_list:
                continue
            for mem in reversed(mem_list):
                if mem.get('context', {}).get('round') != round_num:
                    continue
                sig = mem.get('signature')
                if sig is None:
                    sig = self._extract_action_signature(mem.get('code', ''))
                sig = tuple(sig) if sig else tuple()
                round_signatures[member_id] = sig
                break

        signature_counts = Counter(round_signatures.values())
        total = len(round_signatures)
        unique = len(signature_counts)
        diversity_ratio = unique / total if total else 0.0
        dominant_share = 0.0
        if signature_counts and total:
            dominant_share = signature_counts.most_common(1)[0][1] / total

        entropy = 0.0
        if total:
            for count in signature_counts.values():
                if count <= 0:
                    continue
                p = count / total
                entropy -= p * math.log2(p)

        return {
            "signatures": round_signatures,
            "counts": signature_counts,
            "total": total,
            "unique": unique,
            "diversity_ratio": diversity_ratio,
            "entropy": entropy,
            "dominant_share": dominant_share,
        }

    def _collect_plan_alignment_stats(self, round_num: int) -> dict:
        """Collect plan alignment metrics based on experiment labels for a round."""
        stats = {
            "total_actions": 0,
            "plan_samples": 0,
            "baseline": 0,
            "variation": 0,
            "unmatched": 0,
            "missing": 0,
            "avg_match_score": None,
        }
        scores = []
        for mem_list in self.code_memory.values():
            for mem in mem_list:
                if mem.get('context', {}).get('round') != round_num:
                    continue
                stats["total_actions"] += 1
                experiment = mem.get("experiment")
                if not experiment:
                    stats["missing"] += 1
                    continue
                label = experiment.get("label", "unmatched")
                if label not in ("baseline", "variation", "unmatched"):
                    label = "unmatched"
                stats[label] += 1
                stats["plan_samples"] += 1
                score = experiment.get("match_score")
                if score is not None:
                    try:
                        scores.append(float(score))
                    except (TypeError, ValueError):
                        pass
        if scores:
            stats["avg_match_score"] = sum(scores) / len(scores)
        return stats

    def _identify_ineligible_plan_tags(
        self,
        plan_tags: tuple,
        member_stats: Optional[dict],
        member_count: int,
    ) -> set:
        """Return plan tags that are infeasible given member stats and system availability."""
        if not plan_tags:
            return set()

        land = 0.0
        if member_stats:
            try:
                land = float(member_stats.get("land", 0.0))
            except (TypeError, ValueError):
                land = 0.0

        ineligible = set()
        for tag in plan_tags:
            if tag in ("bear", "offer_land"):
                if land <= 0.0 or member_count < 2:
                    ineligible.add(tag)
            elif tag == "contracts":
                if not hasattr(self, "contracts"):
                    ineligible.add(tag)
            elif tag in ("market", "resources", "businesses"):
                if not hasattr(self, tag):
                    ineligible.add(tag)
            elif tag == "attack":
                if member_count < 2:
                    ineligible.add(tag)
        return ineligible

    def _collect_plan_feasibility_stats(self, round_num: int) -> dict:
        """Collect feasibility stats for planned action tags in a round."""
        stats = {
            "plan_samples": 0,
            "plan_missing": 0,
            "plan_tag_total": 0,
            "plan_feasible_tag_total": 0,
            "plan_ineligible_tag_count": 0,
            "plan_only_tag_count": 0,
            "plan_missing_reason_counts": {
                "missing_experiment": 0,
                "missing_label": 0,
                "missing_signature": 0,
            },
        }

        round_record = None
        if self.execution_history.get("rounds"):
            round_record = self.execution_history["rounds"][-1]
        start_snapshot = {}
        if round_record:
            start_snapshot = round_record.get("round_start_snapshot") or {}
        member_count = len(start_snapshot) if start_snapshot else len(self.current_members)

        for member_id, mem_list in self.code_memory.items():
            for mem in mem_list:
                if mem.get("context", {}).get("round") != round_num:
                    continue
                experiment = mem.get("experiment")
                if not experiment:
                    stats["plan_missing"] += 1
                    stats["plan_missing_reason_counts"]["missing_experiment"] += 1
                    continue
                label = experiment.get("label")
                if label not in ("baseline", "variation"):
                    stats["plan_missing"] += 1
                    stats["plan_missing_reason_counts"]["missing_label"] += 1
                    continue
                if label == "baseline":
                    plan_sig = experiment.get("baseline_signature")
                else:
                    plan_sig = experiment.get("variation_signature")

                plan_tags = self._normalize_action_tags(plan_sig or [])
                if not plan_tags:
                    stats["plan_missing"] += 1
                    stats["plan_missing_reason_counts"]["missing_signature"] += 1
                    continue

                stats["plan_samples"] += 1
                stats["plan_tag_total"] += len(plan_tags)

                context = mem.get("context", {}) or {}
                member_stats = context.get("old_stats") or start_snapshot.get(member_id, {})
                ineligible = self._identify_ineligible_plan_tags(
                    plan_tags,
                    member_stats,
                    member_count,
                )
                stats["plan_ineligible_tag_count"] += len(ineligible)

                feasible_tags = [tag for tag in plan_tags if tag not in ineligible]
                stats["plan_feasible_tag_total"] += len(feasible_tags)

                executed_sig = mem.get("signature")
                if executed_sig is None:
                    executed_sig = self._extract_action_signature(mem.get("code", ""))
                executed_sig = tuple(executed_sig) if executed_sig else tuple()
                plan_only = [tag for tag in feasible_tags if tag not in executed_sig]
                stats["plan_only_tag_count"] += len(plan_only)

        if stats["plan_tag_total"]:
            stats["plan_ineligible_tag_rate"] = (
                stats["plan_ineligible_tag_count"] / stats["plan_tag_total"]
            )
        else:
            stats["plan_ineligible_tag_rate"] = None

        if stats["plan_feasible_tag_total"]:
            stats["plan_only_tag_rate"] = (
                stats["plan_only_tag_count"] / stats["plan_feasible_tag_total"]
            )
        else:
            stats["plan_only_tag_rate"] = None

        return stats

    def _collect_agent_error_stats(self, round_num: int) -> dict:
        """Collect agent code execution error stats for a round."""
        stats = {
            "agent_code_error_count": 0,
            "agent_code_error_tag_counts": {},
            "agent_code_error_type_counts": {},
        }
        round_record = None
        if self.execution_history.get("rounds"):
            round_record = self.execution_history["rounds"][-1]
        if not round_record:
            return stats

        errors = round_record.get("errors", {}).get("agent_code_errors", [])
        if not isinstance(errors, list) or not errors:
            return stats

        tag_counts = Counter()
        type_counts = Counter()
        for error_info in errors:
            code_str = error_info.get("code", "") if isinstance(error_info, dict) else ""
            error_category = None
            if isinstance(error_info, dict):
                error_category = error_info.get("error_category")
            if error_category:
                type_counts[error_category] += 1
            else:
                type_counts["unknown"] += 1
            signature = self._extract_action_signature(code_str)
            signature = tuple(signature) if signature else tuple()
            if signature:
                for tag in signature:
                    tag_counts[tag] += 1
                continue
            error_tag = classify_error_tag(error_info)
            tag_counts[error_tag or "unknown"] += 1

        stats["agent_code_error_count"] = len(errors)
        stats["agent_code_error_tag_counts"] = dict(tag_counts)
        stats["agent_code_error_type_counts"] = dict(type_counts)
        return stats

    def _classify_agent_execution_error(self, error: Exception) -> str:
        """Classify agent execution errors into coarse categories."""
        if isinstance(error, IndentationError):
            return "agent_execution_indent"
        if isinstance(error, SyntaxError):
            return "agent_execution_syntax"
        return "agent_execution"

    def _describe_execution_error(self, error: Exception) -> dict:
        """Extract lightweight error details for diagnostics."""
        details = {"error_type": type(error).__name__ if error is not None else None}
        if isinstance(error, SyntaxError):
            details["error_line"] = error.lineno
            details["error_offset"] = error.offset
            if error.text:
                details["error_text"] = error.text.strip()
        return {key: value for key, value in details.items() if value is not None}

    def _signature_overlap(self, sig_a: tuple, sig_b: tuple) -> float:
        """Compute Jaccard overlap between two action signatures."""
        if not sig_a or not sig_b:
            return 0.0
        set_a = set(sig_a)
        set_b = set(sig_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / float(len(set_a | set_b))

    def _compute_signature_novelty(self, member_id: int, signature: tuple) -> float:
        """Compute a simple novelty score based on prior signature frequency."""
        if not signature:
            return 0.0
        _, history = self._get_member_history(self.code_memory, member_id)
        if not history:
            return 1.0
        count = 0
        for mem in history:
            sig = mem.get('signature')
            if sig is None:
                sig = self._extract_action_signature(mem.get('code', ''))
            if tuple(sig) == signature:
                count += 1
        return 1.0 / (1.0 + count)

    def _compute_population_context_cutoffs(self, snapshot: dict) -> dict:
        """Compute quantile cutoffs for context tagging from a snapshot."""
        if not snapshot:
            return {}

        cutoffs = {}
        fields = ("survival_chance", "relation_balance", "cargo", "land")
        for field in fields:
            values = []
            for stats in snapshot.values():
                value = stats.get(field)
                if value is None:
                    continue
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(val):
                    continue
                values.append(val)

            if not values:
                cutoffs[field] = (0.0, 0.0)
                continue

            if len(values) < 3:
                mean = float(np.mean(values))
                cutoffs[field] = (mean, mean)
                continue

            q_low, q_high = np.nanquantile(values, [0.33, 0.67])
            cutoffs[field] = (float(q_low), float(q_high))

        return cutoffs

    def _assign_band(self, value: float, low: float, high: float, labels: tuple) -> str:
        """Assign a low/mid/high band label for a value."""
        if not labels or len(labels) != 3:
            return "mid"
        try:
            val = float(value)
        except (TypeError, ValueError):
            return labels[1]
        if math.isnan(val):
            return labels[1]
        if low == high:
            return labels[1]
        if val <= low:
            return labels[0]
        if val >= high:
            return labels[2]
        return labels[1]

    def _classify_context_tags(self, stats: dict, cutoffs: dict) -> dict:
        """Classify member context into coarse tags for strategy summaries."""
        if not stats or not cutoffs:
            return {}

        survival_low, survival_high = cutoffs.get('survival_chance', (0.0, 0.0))
        relation_low, relation_high = cutoffs.get('relation_balance', (0.0, 0.0))
        cargo_low, cargo_high = cutoffs.get('cargo', (0.0, 0.0))
        land_low, land_high = cutoffs.get('land', (0.0, 0.0))

        return {
            'survival': self._assign_band(
                stats.get('survival_chance', 0.0),
                survival_low,
                survival_high,
                ("fragile", "steady", "dominant")
            ),
            'relations': self._assign_band(
                stats.get('relation_balance', 0.0),
                relation_low,
                relation_high,
                ("hostile", "neutral", "friendly")
            ),
            'cargo': self._assign_band(
                stats.get('cargo', 0.0),
                cargo_low,
                cargo_high,
                ("cargo_low", "cargo_mid", "cargo_high")
            ),
            'land': self._assign_band(
                stats.get('land', 0.0),
                land_low,
                land_high,
                ("land_low", "land_mid", "land_high")
            ),
        }

    def _context_key_from_tags(self, context_tags: dict) -> str:
        """Build a compact context key from tags for grouping."""
        if not context_tags:
            return ""
        survival = context_tags.get('survival')
        relations = context_tags.get('relations')
        cargo = context_tags.get('cargo')
        land = context_tags.get('land')

        base = ""
        if survival and relations:
            base = f"{survival}/{relations}"
        elif survival:
            base = f"survival={survival}"
        elif relations:
            base = f"relations={relations}"

        qualifiers = []
        if cargo and cargo not in ("cargo_mid", "mid"):
            qualifiers.append(cargo)
        if land and land not in ("land_mid", "mid"):
            qualifiers.append(land)

        if base and qualifiers:
            return base + "|" + "|".join(qualifiers)
        if base:
            return base
        if qualifiers:
            return "|".join(qualifiers)
        return ""

    def _get_memory_context_key(self, mem: dict) -> str:
        """Return a normalized context key for a memory entry."""
        if not mem:
            return ""
        context_key = mem.get('context_key')
        if not context_key:
            context_key = self._context_key_from_tags(mem.get('context_tags', {}))
        return context_key or ""

    def _format_context_tags(self, context_tags: dict) -> str:
        """Format context tags for prompt summaries."""
        if not context_tags:
            return "none"
        return ", ".join(
            f"{key}={value}" for key, value in sorted(context_tags.items())
        )

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

    def _context_similarity_score(self, current_tags: dict, past_tags: dict) -> float:
        """Compute a coarse similarity score between two context tag sets."""
        if not current_tags or not past_tags:
            return 0.0
        shared_keys = set(current_tags) & set(past_tags)
        if not shared_keys:
            return 0.0
        matches = sum(
            1 for key in shared_keys
            if current_tags.get(key) == past_tags.get(key)
        )
        return matches / float(len(shared_keys))

    def _collect_context_weighted_signature_stats(
        self,
        memory: list,
        current_tags: dict,
        min_similarity: float = 0.35,
    ) -> dict:
        """Aggregate signature performance weighted by context similarity."""
        if not memory or not current_tags:
            return {}

        stats = {}
        for mem in memory:
            past_tags = mem.get("context_tags") or {}
            if not past_tags:
                continue
            similarity = self._context_similarity_score(current_tags, past_tags)
            if similarity < min_similarity:
                continue
            sig = mem.get("signature")
            if sig is None:
                sig = self._extract_action_signature(mem.get("code", ""))
            sig = tuple(sig) if sig else tuple()
            perf = self._get_memory_performance(mem)
            record = stats.setdefault(
                sig,
                {
                    "weight_sum": 0.0,
                    "perf_sum": 0.0,
                    "count": 0,
                    "similarity_sum": 0.0,
                },
            )
            record["weight_sum"] += similarity
            record["perf_sum"] += similarity * perf
            record["count"] += 1
            record["similarity_sum"] += similarity

        for record in stats.values():
            weight_sum = record["weight_sum"]
            record["avg"] = record["perf_sum"] / weight_sum if weight_sum else 0.0
            count = record["count"]
            record["similarity_avg"] = (
                record["similarity_sum"] / count if count else 0.0
            )
        return stats

    def _get_memory_performance(self, mem: dict, prefer_round: bool = True) -> float:
        """Return the best available performance metric for a memory entry."""
        if not mem:
            return 0.0
        metrics = mem.get('metrics') or {}
        if prefer_round:
            for key in ("round_delta_survival", "round_relative_survival"):
                if key in metrics:
                    try:
                        return float(metrics[key])
                    except (TypeError, ValueError):
                        pass
        if 'performance' in mem:
            try:
                return float(mem.get('performance', 0.0))
            except (TypeError, ValueError):
                return 0.0
        if 'delta_survival' in metrics:
            try:
                return float(metrics.get('delta_survival', 0.0))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _get_member_context_tags(self, member_id: int) -> dict:
        """Compute current context tags for a member."""
        if member_id is None:
            return {}
        snapshot = self._collect_member_snapshot()
        if not snapshot:
            return {}
        stable_id = self._resolve_member_stable_id(member_id)
        if stable_id is None:
            return {}
        stats = snapshot.get(stable_id)
        if not stats:
            return {}
        cutoffs = self._compute_population_context_cutoffs(snapshot)
        return self._classify_context_tags(stats, cutoffs)

    def _find_contextual_memory_match(self, memory: list, current_tags: dict):
        """Return (index, similarity_score) for the closest context match."""
        if not memory or not current_tags:
            return None, 0.0
        best_idx = None
        best_score = 0.0
        best_perf = float("-inf")
        best_round = -1
        for idx, mem in enumerate(memory):
            past_tags = mem.get('context_tags') or {}
            score = self._context_similarity_score(current_tags, past_tags)
            if score <= 0:
                continue
            perf = self._get_memory_performance(mem)
            round_num = mem.get('context', {}).get('round')
            if round_num is None:
                round_num = idx
            candidate = (score, perf, round_num)
            if candidate > (best_score, best_perf, best_round):
                best_idx = idx
                best_score, best_perf, best_round = candidate
        return best_idx, best_score

    def _summarize_signature_performance(
        self,
        memory: list,
        window: int = 8,
        top_k: int = 3
    ) -> str:
        """Summarize recent performance by action signature."""
        if not memory:
            return "Signature performance (recent window): no history."

        window = max(1, window)
        recent = memory[-window:]

        signature_counts = Counter()
        perf_by_sig = {}
        for mem in recent:
            sig = mem.get('signature')
            if sig is None:
                sig = self._extract_action_signature(mem.get('code', ''))
            sig = tuple(sig) if sig else tuple()
            signature_counts[sig] += 1
            perf_by_sig.setdefault(sig, []).append(self._get_memory_performance(mem))

        total = len(recent)
        stats = []
        for sig, perfs in perf_by_sig.items():
            avg_perf = sum(perfs) / len(perfs)
            std_perf = float(np.std(perfs)) if len(perfs) > 1 else 0.0
            stats.append((avg_perf, std_perf, len(perfs), sig))

        stats_sorted = sorted(stats, key=lambda x: (x[0], x[2]), reverse=True)
        top = stats_sorted[:min(top_k, len(stats_sorted))]
        bottom = sorted(stats, key=lambda x: (x[0], -x[2]))[:min(2, len(stats_sorted))]

        dominant_sig = None
        dominant_share = 0.0
        if signature_counts:
            dominant_sig, dominant_count = signature_counts.most_common(1)[0]
            dominant_share = dominant_count / total if total else 0.0

        lines = [
            "Signature performance (recent window):",
            f"- Window size: {total}",
        ]

        if top:
            lines.append(
                "- Top signatures by avg performance: "
                + "; ".join(
                    f"{self._format_signature(sig)} "
                    f"(avg {avg:.2f}, std {std:.2f}, n={count})"
                    for avg, std, count, sig in top
                )
            )

        if len(stats_sorted) > 1:
            lines.append(
                "- Lowest signatures by avg performance: "
                + "; ".join(
                    f"{self._format_signature(sig)} "
                    f"(avg {avg:.2f}, std {std:.2f}, n={count})"
                    for avg, std, count, sig in bottom
                )
            )

        if dominant_sig is not None and dominant_share >= 0.6 and len(signature_counts) > 1:
            lines.append(
                f"- Concentration: {dominant_share:.2f} of recent actions use "
                f"{self._format_signature(dominant_sig)}."
            )

        if len(signature_counts) <= 1 and total >= 3:
            lines.append(
                "- Diversity note: only one signature used recently; consider testing "
                "underused tags if performance stalls."
            )

        return "\n".join(lines)
