from typing import List, Tuple, Optional
from collections import Counter
import json
import numpy as np
from Leviathan.Member import Member


class IslandExecutionAnalysisMixin:
    def clean_code_string(self, code_str: str) -> str:
        """Remove markdown code block markers and clean up the code string."""
        # Remove ```python and ``` markers
        code_str = code_str.replace('```python', '').replace('```', '').strip()
        
        # Remove any leading or trailing whitespace
        lines = code_str.split('\n')
        lines = [line.rstrip() for line in lines]
        
        # Remove any empty lines at start/end while preserving internal empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
            
        return '\n'.join(lines)

    def _truncate_message(self, message: str, limit: int = 120) -> str:
        """Truncate messages for concise logging/memory."""
        if message is None:
            return ""
        text = str(message)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _collect_strategy_notes(
        self,
        member,
        max_items: int = 3,
        limit: int = 160,
    ) -> List[str]:
        """Collect compact strategy_memory notes from the member for logging."""
        if member is None or not hasattr(member, "strategy_memory"):
            return []

        raw = getattr(member, "strategy_memory")
        notes: List[str] = []

        def _stringify(value) -> str:
            if isinstance(value, str):
                return value
            try:
                return json.dumps(value, ensure_ascii=True, default=str)
            except (TypeError, ValueError):
                return str(value)

        def _add(prefix: Optional[str], value) -> None:
            text = self._truncate_message(_stringify(value), limit=limit)
            if prefix:
                notes.append(f"{prefix}: {text}")
            else:
                notes.append(text)

        if isinstance(raw, dict):
            preferred_keys = (
                "notes",
                "tactics",
                "experiments",
                "hypotheses",
                "outcomes",
                "reports",
                "auto_notes",
            )
            for key in preferred_keys:
                if key not in raw:
                    continue
                value = raw.get(key)
                if isinstance(value, list) and value:
                    _add(key, value[-1])
                elif isinstance(value, dict) and value:
                    last_key = next(reversed(value))
                    _add(f"{key}.{last_key}", value[last_key])
                elif value is not None:
                    _add(key, value)
                if len(notes) >= max_items:
                    break
            if not notes:
                _add("strategy_memory", raw)
        elif isinstance(raw, list):
            for item in raw[-max_items:]:
                _add(None, item)
        elif raw is not None:
            _add(None, raw)

        return notes[:max_items]

    def _format_context_tags(self, context_tags: Optional[dict]) -> str:
        """Format context tags for compact strategy notes."""
        if not context_tags:
            return "none"
        parts = []
        for key in sorted(context_tags.keys()):
            value = context_tags.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts) if parts else "none"

    def _auto_update_strategy_memory(
        self,
        member,
        round_num: int,
        signature: tuple,
        metrics: Optional[dict],
        context_tags: Optional[dict] = None,
        max_items: int = 12,
    ) -> None:
        """Append a compact auto note to strategy_memory for learning continuity."""
        if member is None:
            return

        def _fmt(value, decimals: int) -> str:
            try:
                return f"{float(value):.{decimals}f}"
            except (TypeError, ValueError):
                return f"{0.0:.{decimals}f}"

        sig_text = self._format_signature(signature) if signature else "none"
        ctx_text = self._format_context_tags(context_tags)
        parts = [
            f"auto r{round_num}",
            f"sig={sig_text}",
        ]
        if metrics:
            if "delta_survival" in metrics:
                parts.append(f"d_surv={_fmt(metrics.get('delta_survival'), 2)}")
            if "delta_vitality" in metrics:
                parts.append(f"d_vit={_fmt(metrics.get('delta_vitality'), 1)}")
            if "delta_cargo" in metrics:
                parts.append(f"d_cargo={_fmt(metrics.get('delta_cargo'), 1)}")
            if "delta_land" in metrics:
                parts.append(f"d_land={_fmt(metrics.get('delta_land'), 1)}")
            if "balanced_score" in metrics:
                parts.append(f"score={_fmt(metrics.get('balanced_score'), 2)}")
        if ctx_text != "none":
            parts.append(f"ctx={ctx_text}")
        note = " ".join(parts)

        if not hasattr(member, "strategy_memory") or member.strategy_memory is None:
            member.strategy_memory = {"auto_notes": []}

        mem = member.strategy_memory
        if isinstance(mem, dict):
            bucket = mem.get("auto_notes")
            if not isinstance(bucket, list):
                bucket = []
                mem["auto_notes"] = bucket
            bucket.append(note)
            if len(bucket) > max_items:
                del bucket[:-max_items]
        elif isinstance(mem, list):
            mem.append(note)
            if len(mem) > max_items:
                del mem[:-max_items]
        else:
            member.strategy_memory = {"auto_notes": [str(mem), note]}

    def _extract_message_sender(self, message: str) -> Optional[str]:
        """Extract sender label from a stored message string."""
        if not message:
            return None
        prefix = "From member_"
        if not message.startswith(prefix):
            return None
        remainder = message[len(prefix):]
        parts = remainder.split(":", 1)
        if not parts:
            return None
        try:
            sender_id = int(parts[0].strip())
        except ValueError:
            return None
        return f"member_{sender_id}"

    def _extract_message_stats(self, received_messages: List[str], recent_limit: int = 3) -> dict:
        """Extract structured stats from received messages for logging and memory."""
        stats = {
            "received_count": len(received_messages),
            "sender_counts": {},
            "sender_total": 0,
            "intent_counts": {},
            "intent_total": 0,
            "recent": [],
        }
        if not received_messages:
            return stats

        sender_counts = Counter()
        intent_counts = Counter()
        intents = {
            "attack": ("attack", "raid", "strike", "hit"),
            "offer": ("offer", "give", "gift", "share", "help"),
            "offer_land": ("land", "tile", "territory", "plot"),
            "bear": ("reproduce", "bear", "offspring", "child"),
            "expand": ("expand", "claim", "clear"),
            "alliance": ("ally", "alliance", "coalition", "team"),
            "trade": ("trade", "exchange", "deal", "swap"),
            "defense": ("defend", "protect", "guard"),
        }

        for msg in received_messages:
            sender = self._extract_message_sender(msg)
            if sender:
                sender_counts[sender] += 1
            msg_lower = str(msg).lower()
            for intent, keywords in intents.items():
                if any(keyword in msg_lower for keyword in keywords):
                    intent_counts[intent] += 1

        stats["sender_total"] = len(sender_counts)
        stats["sender_counts"] = {
            sender: count for sender, count in sender_counts.most_common(3)
        }
        stats["intent_total"] = len(intent_counts)
        stats["intent_counts"] = {
            intent: count for intent, count in intent_counts.most_common(4)
        }
        stats["recent"] = [
            self._truncate_message(msg, limit=160)
            for msg in received_messages[-max(1, recent_limit):]
        ]
        return stats

    def _summarize_received_messages(self, received_messages: List[str]) -> str:
        """Summarize received messages for coordination cues."""
        if not received_messages:
            return "No messages received."

        stats = self._extract_message_stats(received_messages)
        sender_counts = stats.get("sender_counts") or {}
        intent_counts = stats.get("intent_counts") or {}
        sender_total = stats.get("sender_total") or 0

        sender_text = (
            ", ".join(f"{sender}={count}" for sender, count in sender_counts.items())
            if sender_counts else "unknown"
        )
        intent_text = (
            ", ".join(f"{intent}={count}" for intent, count in intent_counts.items())
            if intent_counts else "none"
        )
        recent_text = " | ".join(stats.get("recent") or []) if stats.get("recent") else "none"

        lines = [
            "Message digest:",
            f"- received_count: {stats.get('received_count', 0)} "
            f"from {sender_total if sender_total else 'unknown'} senders",
            f"- senders: {sender_text}",
            f"- intents: {intent_text}",
            f"- recent: {recent_text}",
        ]
        return "\n".join(lines)

    def _safe_relation_value(self, matrix, row: int, col: int) -> float:
        if matrix is None:
            return 0.0
        try:
            if row < 0 or col < 0:
                return 0.0
            if row >= matrix.shape[0] or col >= matrix.shape[1]:
                return 0.0
            value = matrix[row, col]
            if np.isnan(value):
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    def _safe_row_sum(self, matrix, row: int) -> float:
        if matrix is None:
            return 0.0
        try:
            if row < 0 or row >= matrix.shape[0]:
                return 0.0
            return float(np.nansum(matrix[row, :]))
        except Exception:
            return 0.0

    def _safe_col_sum(self, matrix, col: int) -> float:
        if matrix is None:
            return 0.0
        try:
            if col < 0 or col >= matrix.shape[1]:
                return 0.0
            return float(np.nansum(matrix[:, col]))
        except Exception:
            return 0.0

    def _compute_relationship_context(self, member_id: int) -> dict:
        """Aggregate numeric relationship signals for similarity matching."""
        victim = self.relationship_dict.get("victim")
        benefit = self.relationship_dict.get("benefit")
        benefit_land = self.relationship_dict.get("benefit_land")

        hostility_in = self._safe_row_sum(victim, member_id)
        hostility_out = self._safe_col_sum(victim, member_id)
        benefit_out = self._safe_row_sum(benefit, member_id)
        benefit_in = self._safe_col_sum(benefit, member_id)
        land_out = self._safe_row_sum(benefit_land, member_id)
        land_in = self._safe_col_sum(benefit_land, member_id)

        support_in = benefit_in + land_in
        support_out = benefit_out + land_out

        return {
            "hostility_in": hostility_in,
            "hostility_out": hostility_out,
            "benefit_in": benefit_in,
            "benefit_out": benefit_out,
            "land_in": land_in,
            "land_out": land_out,
            "support_in": support_in,
            "support_out": support_out,
            "net_support": support_in - hostility_in,
            "net_exchange": support_in - support_out,
        }

    def _summarize_relationship_signals(self, member_id: int, top_k: int = 3) -> str:
        """Summarize aggregated relationship signals for strategy context."""
        victim = self.relationship_dict.get("victim")
        benefit = self.relationship_dict.get("benefit")
        benefit_land = self.relationship_dict.get("benefit_land")
        if victim is None and benefit is None and benefit_land is None:
            return "No relationship signals yet."

        member_count = len(self.current_members)
        threats = []
        allies = []

        total_signal = 0.0
        for other_id in range(member_count):
            if other_id == member_id:
                continue
            attacked_me = self._safe_relation_value(victim, member_id, other_id)
            support_from = (
                self._safe_relation_value(benefit, other_id, member_id)
                + self._safe_relation_value(benefit_land, other_id, member_id)
            )
            if attacked_me or support_from:
                total_signal += abs(attacked_me) + abs(support_from)
            threat_score = attacked_me - 0.5 * support_from
            ally_score = support_from - 0.5 * attacked_me
            threats.append((threat_score, attacked_me, support_from, other_id))
            allies.append((ally_score, support_from, attacked_me, other_id))

        if total_signal <= 0.0:
            return "No relationship signals yet."

        threats = [t for t in threats if t[0] > 0.0 or t[1] > 0.0]
        allies = [a for a in allies if a[0] > 0.0 or a[1] > 0.0]
        threats.sort(reverse=True, key=lambda item: (item[0], item[1]))
        allies.sort(reverse=True, key=lambda item: (item[0], item[1]))

        aggressors = []
        benefactors = []
        if victim is not None:
            for other_id in range(member_count):
                if other_id == member_id:
                    continue
                aggression = self._safe_col_sum(victim, other_id)
                if aggression > 0.0:
                    aggressors.append((aggression, other_id))
        if benefit is not None or benefit_land is not None:
            for other_id in range(member_count):
                if other_id == member_id:
                    continue
                support = self._safe_col_sum(benefit, other_id) + self._safe_col_sum(
                    benefit_land, other_id
                )
                if support > 0.0:
                    benefactors.append((support, other_id))

        aggressors.sort(reverse=True, key=lambda item: item[0])
        benefactors.sort(reverse=True, key=lambda item: item[0])

        lines = ["Relationship signals (soft guidance):"]
        rel_context = self._compute_relationship_context(member_id)
        lines.append(
            "- net_support: "
            f"{rel_context.get('net_support', 0.0):.2f} "
            f"(support_in {rel_context.get('support_in', 0.0):.2f} "
            f"- hostility_in {rel_context.get('hostility_in', 0.0):.2f})"
        )
        lines.append(
            "- net_exchange: "
            f"{rel_context.get('net_exchange', 0.0):.2f} "
            f"(support_in {rel_context.get('support_in', 0.0):.2f} "
            f"- support_out {rel_context.get('support_out', 0.0):.2f})"
        )

        if threats:
            threats_text = ", ".join(
                f"member_{other} (attacked_me {attacked:.2f}, support_from {support:.2f})"
                for _, attacked, support, other in threats[:max(1, int(top_k))]
            )
            lines.append(f"- top threats: {threats_text}")
        else:
            lines.append("- top threats: none")

        if allies:
            allies_text = ", ".join(
                f"member_{other} (support_from {support:.2f}, attacked_me {attacked:.2f})"
                for _, support, attacked, other in allies[:max(1, int(top_k))]
            )
            lines.append(f"- top allies: {allies_text}")
        else:
            lines.append("- top allies: none")

        if aggressors:
            aggressor_text = ", ".join(
                f"member_{other} ({score:.2f})"
                for score, other in aggressors[:max(1, int(top_k))]
            )
            lines.append(f"- population aggressors: {aggressor_text}")
        if benefactors:
            benefactor_text = ", ".join(
                f"member_{other} ({score:.2f})"
                for score, other in benefactors[:max(1, int(top_k))]
            )
            lines.append(f"- population benefactors: {benefactor_text}")

        return "\n".join(lines)

    def _average_similarity(self, memory_block: dict, current_block: dict, keys: Tuple[str, ...]) -> Optional[float]:
        if not memory_block or not current_block:
            return None
        scores = []
        for key in keys:
            if key not in memory_block or key not in current_block:
                continue
            try:
                mem_val = float(memory_block[key])
                cur_val = float(current_block[key])
            except (TypeError, ValueError):
                continue
            denom = abs(mem_val) + abs(cur_val) + 1.0
            diff = abs(mem_val - cur_val) / denom
            similarity = max(0.0, 1.0 - diff)
            scores.append(similarity)
        if not scores:
            return None
        return float(np.mean(scores))

    def _context_similarity_score(
        self,
        memory_context: dict,
        current_stats: Optional[dict],
        current_round_context: Optional[dict],
        current_relation_context: Optional[dict] = None,
    ) -> Optional[float]:
        """Compute similarity between a memory context and current conditions."""
        if not memory_context:
            return None
        stats_score = None
        round_score = None
        relation_score = None
        if current_stats:
            stats_score = self._average_similarity(
                memory_context.get("old_stats", {}),
                current_stats,
                ("vitality", "cargo", "land", "survival_chance"),
            )
        if current_round_context:
            round_score = self._average_similarity(
                memory_context.get("round_context", {}),
                current_round_context,
                (
                    "attack_rate",
                    "benefit_rate",
                    "benefit_land_rate",
                    "land_scarcity",
                    "resource_pressure",
                    "avg_vitality",
                    "avg_cargo",
                    "gini_wealth",
                    "action_entropy",
                    "dominant_action_share",
                ),
            )
        if current_relation_context:
            relation_score = self._average_similarity(
                memory_context.get("relationship_context", {}),
                current_relation_context,
                (
                    "hostility_in",
                    "hostility_out",
                    "support_in",
                    "support_out",
                    "net_support",
                    "net_exchange",
                ),
            )
        if stats_score is None and round_score is None and relation_score is None:
            return None
        components = []
        if stats_score is not None:
            components.append((0.5, stats_score))
        if round_score is not None:
            components.append((0.3, round_score))
        if relation_score is not None:
            components.append((0.2, relation_score))
        total_weight = sum(weight for weight, _ in components)
        if total_weight <= 0:
            return None
        return sum(weight * score for weight, score in components) / total_weight

    def _extract_action_signature(self, code_str: str) -> tuple:
        """Extract a coarse action signature from generated code for diversity sampling."""
        if not code_str:
            return tuple()

        patterns = {
            "attack": ("execution_engine.attack(",),
            "offer": ("execution_engine.offer(",),
            "offer_land": ("execution_engine.offer_land(",),
            "bear": ("execution_engine.bear(",),
            "expand": ("execution_engine.expand(",),
            "message": (
                "execution_engine.send_message(",
                "execution_engine.send_message_by_id(",
                "send_message(",
                "send_message_by_id(",
            ),
        }

        signature = []
        for tag, needles in patterns.items():
            if any(needle in code_str for needle in needles):
                signature.append(tag)

        return tuple(signature)

    def _extract_executed_signature(
        self,
        action_summary: Optional[dict],
        message_summary: Optional[dict] = None,
    ) -> tuple:
        """Extract a signature based on executed actions/messages."""
        if not action_summary and not message_summary:
            return tuple()

        tags = []
        if action_summary:
            for action, summary in action_summary.items():
                try:
                    outgoing = int(summary.get("outgoing_count", 0))
                except (TypeError, ValueError, AttributeError):
                    outgoing = 0
                if outgoing > 0:
                    tags.append(action)

        if message_summary and message_summary.get("sent_count", 0):
            tags.append("message")

        if not tags:
            return tuple()

        order = ("attack", "offer", "offer_land", "bear", "expand", "message")
        ordered = [tag for tag in order if tag in tags]
        extras = sorted(tag for tag in set(tags) if tag not in order)
        return tuple(ordered + extras)

    def _get_planned_signature(self, memory_entry: dict) -> tuple:
        """Return the signature implied by generated code."""
        if not memory_entry:
            return tuple()
        sig = memory_entry.get("signature")
        if sig is None:
            sig = self._extract_action_signature(memory_entry.get("code", ""))
        return tuple(sig) if sig else tuple()

    def _get_executed_signature(self, memory_entry: dict) -> tuple:
        """Return the signature based on executed actions, if recorded."""
        if not memory_entry:
            return tuple()
        sig = memory_entry.get("executed_signature")
        return tuple(sig) if sig else tuple()

    def _get_entry_signature(
        self,
        memory_entry: dict,
        prefer_executed: bool = True,
        fallback_to_planned: bool = True,
    ) -> tuple:
        """Select an action signature with optional executed-first preference."""
        if not memory_entry:
            return tuple()
        if prefer_executed:
            sig = self._get_executed_signature(memory_entry)
            if sig or not fallback_to_planned:
                return sig
        return self._get_planned_signature(memory_entry)

    def _entry_execution_status(self, memory_entry: dict) -> str:
        """Classify whether a memory entry reflects executed actions or plans only."""
        if not memory_entry:
            return "unknown"
        if memory_entry.get("error"):
            return "error"
        executed_sig = self._get_executed_signature(memory_entry)
        if executed_sig:
            return "executed"
        planned_sig = self._get_planned_signature(memory_entry)
        if planned_sig:
            return "planned_only"
        return "idle"

    def _format_signature(self, signature: tuple) -> str:
        """Format a signature tuple for readable summaries."""
        if not signature:
            return "none"
        return ", ".join(signature)

    def _compute_action_feasibility(self, member) -> dict:
        """Return coarse feasibility for common actions based on current state."""
        if member is None:
            return {}
        member_count = len(getattr(self, "current_members", []) or [])
        has_other = member_count > 1
        has_neighbors = bool(getattr(member, "current_clear_list", []))
        return {
            "attack": has_other,
            "offer": has_other,
            "offer_land": has_other and bool(getattr(member, "land_num", 0) > 0),
            "bear": bool(getattr(member, "is_qualified_to_reproduce", False)) and has_neighbors,
            "expand": has_neighbors,
            "message": has_other,
        }

    def _summarize_action_feasibility(self, member_id: int) -> str:
        """Summarize which actions are currently feasible and why."""
        try:
            member = self.current_members[member_id]
        except (TypeError, IndexError):
            return "Action feasibility: unavailable (invalid member index)."

        member_count = len(getattr(self, "current_members", []) or [])
        has_other = member_count > 1
        has_neighbors = bool(getattr(member, "current_clear_list", []))
        has_land = bool(getattr(member, "land_num", 0) > 0)
        qualified_reproduce = bool(getattr(member, "is_qualified_to_reproduce", False))

        feasibility = {
            "attack": has_other,
            "offer": has_other,
            "offer_land": has_other and has_land,
            "bear": qualified_reproduce and has_neighbors,
            "expand": has_neighbors,
            "message": has_other,
        }

        reasons = {}
        if not has_other:
            reasons["attack"] = ["no other members alive"]
            reasons["offer"] = ["no other members alive"]
            reasons["message"] = ["no other members alive"]
        if not has_land:
            reasons.setdefault("offer_land", []).append("no land to offer")
        if not qualified_reproduce:
            reasons.setdefault("bear", []).append("not qualified to reproduce")
        if not has_neighbors:
            reasons.setdefault("bear", []).append("no neighbors in current_clear_list")
            reasons.setdefault("expand", []).append("no neighbors in current_clear_list")

        eligible = [tag for tag, ok in feasibility.items() if ok]
        ineligible = [tag for tag, ok in feasibility.items() if not ok]

        lines = ["Action feasibility (current):"]
        lines.append(f"- eligible: {', '.join(eligible) if eligible else 'none'}")
        if ineligible:
            details = []
            for tag in ineligible:
                note = "; ".join(reasons.get(tag, []))
                if note:
                    details.append(f"{tag} ({note})")
                else:
                    details.append(tag)
            lines.append(f"- ineligible: {', '.join(details)}")
        return "\n".join(lines)

    def _summarize_neighbor_index_map(
        self,
        member,
        max_items: int = 6,
    ) -> str:
        """Map stable neighbor ids to current indices for safe targeting."""
        if member is None:
            return "Neighbor index map: unavailable."
        neighbors = list(getattr(member, "current_clear_list", []) or [])
        if not neighbors:
            return "Neighbor index map: none (current_clear_list empty)."

        mapping = []
        for stable_id in neighbors:
            idx = self.resolve_member_index_by_id(stable_id)
            if idx is None:
                continue
            mapping.append((stable_id, idx))

        if not mapping:
            return "Neighbor index map: none (no live neighbors resolved)."

        shown = mapping[: max(1, int(max_items))]
        extra = len(mapping) - len(shown)
        text = ", ".join(f"{stable_id}->{idx}" for stable_id, idx in shown)
        if extra > 0:
            text += f" (+{extra} more)"
        return f"Neighbor index map (stable id -> current index): {text}"

    def _member_storage_key(self, member_ref) -> Optional[int]:
        """Resolve current member index to stable member.id for memory tracking."""
        if isinstance(member_ref, Member):
            return getattr(member_ref, "id", None)
        try:
            idx = int(member_ref)
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(self.current_members):
            return getattr(self.current_members[idx], "id", None)
        return None

    def _signature_ineligible_tags(self, member, signature: tuple) -> list:
        """Return tags in a signature that are currently ineligible for the member."""
        if not signature or member is None:
            return []
        feasibility = self._compute_action_feasibility(member)
        return [tag for tag in signature if tag in feasibility and not feasibility[tag]]

    def _get_memory_signatures(self, memory: list) -> list:
        """Normalize action signatures from memory entries."""
        signatures = []
        for mem in memory:
            sig = self._get_entry_signature(mem)
            signatures.append(sig)
        return signatures

    def _compute_signature_novelty(self, member_id: int, signature: tuple) -> float:
        """Compute a simple novelty score based on prior signature frequency."""
        if not signature:
            return 0.0
        member_key = self._member_storage_key(member_id)
        history = self.code_memory.get(member_key, []) if member_key is not None else []
        if not history:
            return 1.0
        count = 0
        for mem in history:
            sig = self._get_entry_signature(mem)
            if sig == signature:
                count += 1
        return 1.0 / (1.0 + count)

    def _signature_reliability_stats(self, memory_entries: list) -> dict:
        """Compute per-signature execution reliability and infeasibility stats."""
        stats = {}
        for mem in memory_entries:
            planned_sig = self._get_planned_signature(mem)
            executed_sig = self._get_executed_signature(mem)

            if planned_sig:
                sig = tuple(planned_sig)
                record = stats.setdefault(
                    sig,
                    {"planned": 0, "executed": 0, "infeasible_sum": 0.0},
                )
                record["planned"] += 1
                infeasible_tags = (
                    mem.get("context", {}).get("planned_infeasible_tags") or []
                )
                if infeasible_tags:
                    sig_tags = set(sig)
                    if sig_tags:
                        infeasible_count = sum(
                            1 for tag in sig_tags if tag in infeasible_tags
                        )
                        record["infeasible_sum"] += infeasible_count / len(sig_tags)

            if executed_sig:
                sig = tuple(executed_sig)
                record = stats.setdefault(
                    sig,
                    {"planned": 0, "executed": 0, "infeasible_sum": 0.0},
                )
                record["executed"] += 1

        results = {}
        for sig, record in stats.items():
            planned = record["planned"]
            executed = record["executed"]
            if planned <= 0:
                exec_rate = 1.0 if executed > 0 else 0.0
                infeasible_rate = 0.0
            else:
                exec_rate = executed / planned
                infeasible_rate = record["infeasible_sum"] / planned
            results[sig] = {
                "planned": planned,
                "executed": executed,
                "execution_rate": exec_rate,
                "infeasible_rate": infeasible_rate,
            }
        return results

    def _signature_streak(
        self,
        member_id: int,
        window: int = 4,
        prefer_executed: bool = True,
        fallback_to_planned: bool = True,
        include_empty: bool = True,
    ) -> Tuple[int, tuple]:
        """Return (streak_length, last_signature) for recent code history."""
        member_key = self._member_storage_key(member_id)
        memory = self.code_memory.get(member_key, []) if member_key is not None else []
        if not memory:
            return 0, tuple()

        signatures = []
        for mem in memory:
            if prefer_executed and not fallback_to_planned:
                sig = self._get_executed_signature(mem)
            else:
                sig = self._get_entry_signature(
                    mem,
                    prefer_executed=prefer_executed,
                    fallback_to_planned=fallback_to_planned,
                )
            sig = tuple(sig) if sig else tuple()
            if not include_empty and not sig:
                continue
            signatures.append(sig)

        if not signatures:
            return 0, tuple()

        recent = signatures[-max(1, window):]
        last_sig = recent[-1] if recent else tuple()
        streak = 1 if recent else 0
        for sig in reversed(recent[:-1]):
            if sig == last_sig:
                streak += 1
            else:
                break
        return streak, last_sig

    def _idle_signature_streak(self, member_id: int, window: int = 4) -> int:
        """Count consecutive recent rounds with no executed actions/messages."""
        member_key = self._member_storage_key(member_id)
        memory = self.code_memory.get(member_key, []) if member_key is not None else []
        if not memory:
            return 0
        recent = memory[-max(1, window):]
        streak = 0
        for mem in reversed(recent):
            sig = self._get_executed_signature(mem)
            if sig:
                break
            streak += 1
        return streak

    def _rename_action_keys(self, mapping: Optional[dict]) -> dict:
        if not mapping:
            return {}
        rename = {"reproduce": "bear", "clear": "expand"}
        result = {}
        for key, value in mapping.items():
            mapped = rename.get(key, key)
            try:
                result[mapped] = float(value)
            except (TypeError, ValueError):
                result[mapped] = value
        return result

    def _format_action_stats(self, action_map: dict, action_order: Tuple[str, ...]) -> str:
        if not action_map:
            return "none"
        parts = []
        for action in action_order:
            if action in action_map:
                parts.append(f"{action}={action_map[action]:.2f}")
        return ", ".join(parts) if parts else "none"

    def _summarize_strategy_state(self, member) -> str:
        """Summarize member-level strategy memory for prompt guidance."""
        if not hasattr(member, "strategy_state"):
            return "No strategy profile available."

        state = member.strategy_state or {}
        profile = getattr(member, "strategy_profile", "unknown")
        action_order = ("attack", "offer", "offer_land", "bear", "expand")

        profile_weights = self._rename_action_keys(
            Member._STRATEGY_PROFILES.get(profile, {})
        )
        profile_bias = self._format_action_stats(profile_weights, action_order)

        decision_scale = self._rename_action_keys(state.get("decision_scale", {}))
        adaptive_scale = self._rename_action_keys(state.get("adaptive_scale", {}))
        action_memory = self._rename_action_keys(state.get("action_memory", {}))
        action_efficacy = self._rename_action_keys(state.get("action_efficacy", {}))

        recent_rewards = list(state.get("recent_rewards", []))
        reward_avg = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        reward_last = float(recent_rewards[-1]) if recent_rewards else 0.0

        total_usage = sum(action_memory.values()) if action_memory else 0.0
        underused = []
        overused = []
        if total_usage > 0:
            expected = 1.0 / max(1, len(action_memory))
            for action, count in action_memory.items():
                share = count / total_usage
                if share < expected * 0.7:
                    underused.append(action)
                elif share > expected * 1.3:
                    overused.append(action)
        elif action_memory:
            underused = list(action_memory.keys())

        eligibility = {
            "bear": bool(getattr(member, "is_qualified_to_reproduce", False)),
            "offer_land": bool(getattr(member, "land_num", 0) > 0),
            "expand": bool(getattr(member, "current_clear_list", [])),
        }
        other_members = max(0, len(getattr(self, "current_members", []) or []) - 1)
        neighbor_count = len(getattr(member, "current_clear_list", []) or [])
        if underused:
            underused = [action for action in underused if eligibility.get(action, True)]

        streak, last_sig = self._signature_streak(
            member.surviver_id,
            prefer_executed=True,
            fallback_to_planned=False,
            include_empty=False,
        )
        idle_streak = self._idle_signature_streak(member.surviver_id)

        lines = [
            "Strategy state (member memory):",
            f"- profile: {profile}",
            f"- profile_bias (soft anchor): {profile_bias}",
            f"- decision_scale: {self._format_action_stats(decision_scale, action_order)}",
            f"- adaptive_scale: {self._format_action_stats(adaptive_scale, action_order)}",
            f"- action_memory: {self._format_action_stats(action_memory, action_order)}",
            f"- action_efficacy: {self._format_action_stats(action_efficacy, action_order)}",
            f"- recent_reward_avg: {reward_avg:.2f}; last: {reward_last:.2f}",
            f"- eligibility: bear={eligibility['bear']}, offer_land={eligibility['offer_land']}, expand={eligibility['expand']}",
            f"- target availability: other_members={other_members}, neighbor_count={neighbor_count}",
        ]
        if underused:
            lines.append(f"- underused actions (eligible): {', '.join(underused)}")
        if overused:
            lines.append(f"- overused actions: {', '.join(overused)}")
        if streak > 1:
            sig_text = ", ".join(last_sig) if last_sig else "none"
            lines.append(f"- recent action signature streak: {streak}x ({sig_text})")
        if idle_streak >= 2:
            lines.append(f"- recent inactivity streak: {idle_streak}x (no executed actions)")

        return "\n".join(lines)

    def _summarize_population_action_mix(self) -> str:
        """Summarize last-round action mix to discourage population collapse."""
        if not self.execution_history.get('rounds'):
            return "No previous action mix available."

        last_round = self.execution_history['rounds'][-1]
        counts = {"attack": 0, "offer": 0, "offer_land": 0, "bear": 0, "expand": 0}
        alias = {
            "attack": "attack",
            "benefit": "offer",
            "benefit_land": "offer_land",
            "reproduce": "bear",
            "clear": "expand",
            "offer": "offer",
            "offer_land": "offer_land",
            "bear": "bear",
            "expand": "expand",
        }

        for entry in last_round.get('actions', []):
            action_summary = entry.get('action_summary', {}) or {}
            for action_name, summary in action_summary.items():
                mapped = alias.get(action_name)
                if not mapped:
                    continue
                counts[mapped] += int(summary.get('outgoing_count', 0))

        total = sum(counts.values())
        if total <= 0:
            return "No previous action mix available."

        parts = [
            f"{action}={counts[action]} ({counts[action] / total:.2f})"
            for action in counts
        ]
        dominant = max(counts, key=counts.get)
        dominance = counts[dominant] / total
        dominance_note = f"; dominant={dominant} ({dominance:.2f})" if dominance >= 0.5 else ""
        return "Population action mix (last round): " + ", ".join(parts) + dominance_note

    def _summarize_population_state(self, top_k: int = 3) -> str:
        """Summarize current population state for strategic grounding."""
        members = list(self.current_members) if hasattr(self, "current_members") else []
        if not members:
            return "Population state snapshot: no members."

        member_count = len(members)
        cargo_vals = [float(m.cargo) for m in members]
        land_vals = [float(m.land_num) for m in members]
        vitality_vals = [float(m.vitality) for m in members]
        survival_vals = [float(self.compute_survival_chance(m)) for m in members]

        def _mean_std(values):
            if not values:
                return 0.0, 0.0
            arr = np.array(values, dtype=float)
            return float(np.mean(arr)), float(np.std(arr))

        cargo_mean, cargo_std = _mean_std(cargo_vals)
        land_mean, land_std = _mean_std(land_vals)
        vitality_mean, vitality_std = _mean_std(vitality_vals)

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

        round_context = self._compute_round_context()
        attack_rate = float(round_context.get("attack_rate", 0.0))
        benefit_rate = float(round_context.get("benefit_rate", 0.0))
        benefit_land_rate = float(round_context.get("benefit_land_rate", 0.0))
        resource_pressure = float(round_context.get("resource_pressure", 0.0))
        action_entropy = float(round_context.get("action_entropy", 0.0))
        dominant_action_share = float(round_context.get("dominant_action_share", 0.0))

        top_k = max(0, int(top_k))
        top_holders = sorted(
            members,
            key=lambda m: (m.cargo + m.land_num),
            reverse=True,
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
            for comm in (last_round.get('messages', {}) or {}).values():
                for recipient_id, _msg in comm.get('sent', []) or []:
                    sent_total += 1
                    if recipient_id is not None:
                        recipients.add(recipient_id)
                for msg in comm.get('received', []) or []:
                    received_total += 1
                    sender_label = self._extract_message_sender(msg)
                    if sender_label is not None:
                        senders.add(sender_label)

        if sent_total or received_total:
            communication_line = (
                f"- Communication last round: sent {sent_total} "
                f"(to {len(recipients)}), received {received_total} "
                f"(from {len(senders)})"
            )
        else:
            communication_line = "- Communication last round: no messages"

        lines = [
            "Population state snapshot:",
            f"- Members: {member_count}; land scarcity: "
            f"{land_scarcity:.2f} ({land_owned:.0f}/{land_total:.0f})",
            f"- Survival chance (min/avg/max): "
            f"{survival_min:.2f}/{survival_mean:.2f}/{survival_max:.2f}",
            f"- Vitality avg/std: {vitality_mean:.2f}/{vitality_std:.2f}; "
            f"Cargo avg/std: {cargo_mean:.2f}/{cargo_std:.2f}; "
            f"Land avg/std: {land_mean:.2f}/{land_std:.2f}",
            f"- Inequality (Gini): cargo {gini_cargo:.2f}, land {gini_land:.2f}, "
            f"wealth {gini_wealth:.2f}",
            f"- Interaction rates per member (current round): attack {attack_rate:.2f}, "
            f"benefit {benefit_rate:.2f}, land benefit {benefit_land_rate:.2f}",
            f"- Action entropy/dominance: entropy {action_entropy:.2f}, "
            f"dominant_share {dominant_action_share:.2f}",
            f"- Resource pressure: {resource_pressure:.2f}",
            f"- Top holders (cargo+land): {top_text}",
            communication_line,
        ]

        return "\n".join(lines)

    def _summarize_member_action_mix(self, member_id: int, window: int = 5) -> str:
        """Summarize a member's recent action mix to encourage adaptive diversity."""
        member_key = self._member_storage_key(member_id)
        memory = self.code_memory.get(member_key, []) if member_key is not None else []
        if not memory:
            return "no history"

        window = max(1, int(window))
        recent = memory[-window:]
        action_order = ("attack", "offer", "offer_land", "bear", "expand", "message")
        counts = {action: 0 for action in action_order}

        for mem in recent:
            context = mem.get("context", {}) or {}
            action_summary = context.get("action_summary") or {}
            for action, summary in action_summary.items():
                if action in counts:
                    counts[action] += int(summary.get("outgoing_count", 0))
            message_summary = context.get("message_summary") or {}
            sent_count = message_summary.get("sent_count", 0)
            if sent_count:
                counts["message"] += int(sent_count)

        total = sum(counts.values())
        if total <= 0:
            return "no actions recorded"

        parts = []
        for action in action_order:
            if counts[action] > 0:
                parts.append(f"{action}={counts[action]} ({counts[action] / total:.2f})")

        if not parts:
            return "no actions recorded"

        dominant = max(counts, key=counts.get)
        dominance = counts[dominant] / total if total > 0 else 0.0
        dominance_note = f"; dominant={dominant} ({dominance:.2f})" if dominance >= 0.5 else ""
        return ", ".join(parts) + dominance_note

