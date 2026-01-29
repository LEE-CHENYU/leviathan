from Leviathan.Island import Island
import Leviathan.api_key
from Leviathan.Member import Member
from typing import List, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd
import openai
import json
from datetime import datetime
import traceback
import os
import math

class IslandExecution(Island):
    def __init__(self, 
        init_member_number: int,
        land_shape: Tuple[int, int],
        save_path: str,
        random_seed: Optional[int] = None,
        action_board: List[List[Tuple[str, int, int]]] = None # change
    ):
        super().__init__(
            init_member_number,
            land_shape,
            save_path,
            random_seed
        )
        
        # Remove example action board and related attributes since we're not using them
        self.performance_history = {}  # {member_id (stable id): [list_of_performance_metrics]}
        
        # Add code memory tracking
        self.code_memory = {}  # {member_id (stable id): [{'code': str, 'performance': float, 'context': dict}]}
        
        # Add execution history tracking
        self.execution_history = {
            'rounds': [],
            'generated_code': {},
            'performance_metrics': {},
            'errors': [],
            'relationships': {},
            'survival_tracking': {}
        }

        # Add message storage
        self.messages = {}  # {member_id: [list_of_messages]}
        # Cache messages consumed during decision for later logging/evaluation
        self._decision_messages = {}

    def _resolve_member_ref(self, member_ref, prefer_index: bool = True) -> Optional[Member]:
        """Resolve a member reference (Member or index/id) to a live Member object."""
        if isinstance(member_ref, Member):
            return member_ref
        idx = self.resolve_member_index(member_ref, prefer_index=prefer_index)
        if idx is None:
            return None
        if 0 <= idx < len(self.current_members):
            return self.current_members[idx]
        return None

    def _require_member(self, member_ref, label: str, prefer_index: bool = True) -> Member:
        member = self._resolve_member_ref(member_ref, prefer_index=prefer_index)
        if member is None:
            raise ValueError(f"{label} reference {member_ref} could not be resolved")
        return member

    def offer(self, member_1, member_2, parameter_influence):
        giver = self._require_member(member_1, "offer member_1")
        receiver = self._require_member(member_2, "offer member_2")
        super()._offer(giver, receiver, parameter_influence)

    def offer_land(self, member_1, member_2, parameter_influence: bool = True):
        giver = self._require_member(member_1, "offer_land member_1")
        receiver = self._require_member(member_2, "offer_land member_2")
        super()._offer_land(giver, receiver, parameter_influence)
        
    def attack(self, member_1, member_2):
        attacker = self._require_member(member_1, "attack member_1")
        target = self._require_member(member_2, "attack member_2")
        super()._attack(attacker, target)

    def bear(self, member_1, member_2):
        parent = self._require_member(member_1, "bear member_1")
        partner = self._require_member(member_2, "bear member_2")
        super()._bear(parent, partner)

    def expand(self, member_1, member_2=None):
        actor = self._require_member(member_1, "expand member_1")
        super()._expand(actor)

    def parse_relationship_matrix(self, relationship_dict=None):
        """
        Parse and return a human-readable summary of the relationship matrices.
        
        :param relationship_dict: A dictionary with keys like 'victim', 'benefit', 'benefit_land'
                                 each containing a NxN numpy array of relationships.
        :return: A list of strings describing the relationships.
        """
        if relationship_dict is None:
            relationship_dict = self.relationship_dict
        summary = []
        rel_map = {
            'victim':      "member_{i} was attacked by member_{j}",
            'benefit':     "member_{i} gave a benefit to member_{j}",
            'benefit_land':"member_{i} gave land to member_{j}"
        }
        
        for relation_type, matrix in relationship_dict.items():
            if relation_type not in rel_map:
                continue
            
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    # Filter out invalid or zero entries
                    if not np.isnan(val) and val != 0:
                        # Construct a description
                        statement = (f"{rel_map[relation_type]} "
                                     f"(value={val:.2f})")
                        # Replace {i} with actual index+1 (or keep zero-based)
                        # Same for {j}
                        statement = statement.format(i=i, j=j)
                        summary.append(statement)
        
        return summary
    
    def get_current_member_features(self) -> pd.DataFrame:
        """Collect features for all current members"""
        feature_rows = []
        
        for idx, member in enumerate(self.current_members):
            # Get self attributes
            feature_row = {
                "member_index": getattr(member, "surviver_id", idx),
                "self_productivity": member.overall_productivity,
                "self_vitality": member.vitality, 
                "self_cargo": member.cargo,
                "self_land": member.land_num,
                "self_age": member.age,
                "self_neighbor": len(member.current_clear_list),
                "member_id": member.id
            }
            feature_rows.append(feature_row)
                
        return pd.DataFrame(feature_rows)

    def _summarize_member_round_actions(self, member_id: int) -> dict:
        """Summarize round actions involving a member for memory/evaluation."""
        summary = {}
        if not hasattr(self, "round_action_dict"):
            return summary
        rename = {
            "benefit": "offer",
            "benefit_land": "offer_land",
            "reproduce": "bear",
            "clear": "expand",
        }
        for action_name, action_map in self.round_action_dict.items():
            if not action_map:
                continue
            outgoing_total = 0.0
            incoming_total = 0.0
            outgoing_count = 0
            incoming_count = 0
            for (actor_id, target_id), value in action_map.items():
                if actor_id == member_id:
                    outgoing_total += float(value)
                    outgoing_count += 1
                if target_id == member_id:
                    incoming_total += float(value)
                    incoming_count += 1
            if outgoing_count or incoming_count:
                label = rename.get(action_name, action_name)
                if label in summary:
                    summary[label]["outgoing_total"] += outgoing_total
                    summary[label]["outgoing_count"] += outgoing_count
                    summary[label]["incoming_total"] += incoming_total
                    summary[label]["incoming_count"] += incoming_count
                else:
                    summary[label] = {
                        "outgoing_total": outgoing_total,
                        "outgoing_count": outgoing_count,
                        "incoming_total": incoming_total,
                        "incoming_count": incoming_count,
                    }
        return summary
    
    # Remove decision() method since we're only using agent_code_decision

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
            }

        total_actions = stats.get("total_actions", 0) or 0
        if total_actions <= 0:
            return {
                "pressure": {},
                "dominant_signature": stats.get("dominant_signature", tuple()),
                "dominant_share": stats.get("dominant_share", 0.0),
                "total_actions": 0,
            }

        known_tags = stats.get("known_tags") or ()
        if not known_tags:
            return {
                "pressure": {},
                "dominant_signature": stats.get("dominant_signature", tuple()),
                "dominant_share": stats.get("dominant_share", 0.0),
                "total_actions": total_actions,
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

        return {
            "pressure": pressure,
            "dominant_signature": stats.get("dominant_signature", tuple()),
            "dominant_share": stats.get("dominant_share", 0.0),
            "total_actions": total_actions,
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

    def _summarize_strategy_recommendations(
        self,
        member,
        member_id: int,
        window: int = 8,
        min_samples: int = 2,
        exploration_bonus: float = 0.35,
        population_window_rounds: int = 3,
        max_items: int = 3,
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
                sig = candidate[4]
                if sig in excluded:
                    continue
                if not self._signature_ineligible_tags(member, sig):
                    return candidate
            return None

        total = sum(signature_counts.values())
        tag_pressure_bundle = self._compute_population_tag_pressure(
            window_rounds=population_window_rounds
        )
        tag_pressure = tag_pressure_bundle.get("pressure") or {}
        sig_stats = []
        for sig, scores in signature_scores.items():
            count = len(scores)
            avg = sum(scores) / count if count else 0.0
            ucb = avg + exploration_bonus * math.sqrt(
                math.log(total + 1) / max(1, count)
            )
            diversity_bonus = self._signature_diversity_bonus(sig, tag_pressure)
            reliability = reliability_stats.get(sig, {})
            exec_rate = reliability.get("execution_rate", 1.0)
            infeasible_rate = reliability.get("infeasible_rate", 0.0)
            reliability_penalty = -0.15 * max(0.0, 1.0 - float(exec_rate))
            reliability_penalty -= 0.1 * max(0.0, min(1.0, float(infeasible_rate)))
            adjusted = ucb + diversity_bonus + reliability_penalty
            sig_stats.append(
                (adjusted, ucb, avg, count, sig, diversity_bonus, reliability_penalty)
            )
        sig_stats.sort(key=lambda item: item[0], reverse=True)

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

        baseline_sig = best_by_avg[2] if best_by_avg else None
        explore_sig = sig_stats[0][4] if sig_stats else None
        baseline_ineligible = bool(
            self._signature_ineligible_tags(member, baseline_sig)
        ) if baseline_sig else False
        explore_ineligible = bool(
            self._signature_ineligible_tags(member, explore_sig)
        ) if explore_sig else False

        if sig_stats:
            adjusted, ucb, avg, count, sig, diversity_bonus, reliability_penalty = sig_stats[0]
            note = _signature_note(sig)
            adjustment_parts = []
            if diversity_bonus:
                adjustment_parts.append(f"diversity {diversity_bonus:+.2f}")
            if reliability_penalty:
                adjustment_parts.append(f"reliability {reliability_penalty:+.2f}")
            if adjustment_parts:
                lines.append(
                    "- Exploration candidate (UCB adjusted): "
                    f"{self._format_signature(sig)} "
                    f"(score {adjusted:.2f}, ucb {ucb:.2f}, "
                    f"{', '.join(adjustment_parts)}, avg {avg:.2f}, n={count}){note}"
                )
            else:
                lines.append(
                    f"- Exploration candidate (UCB): {self._format_signature(sig)} "
                    f"(score {ucb:.2f}, avg {avg:.2f}, n={count}){note}"
                )

        shown_sigs = {sig for sig in (baseline_sig, explore_sig) if sig}
        if baseline_ineligible or explore_ineligible:
            feasible_candidate = _best_feasible_candidate(sig_stats)
            if feasible_candidate:
                adjusted, _, avg, count, sig, _, _ = feasible_candidate
                if sig not in shown_sigs:
                    note = _signature_note(sig)
                    lines.append(
                        f"- Feasible alternative: {self._format_signature(sig)} "
                        f"(score {adjusted:.2f}, avg {avg:.2f}, n={count}){note}"
                    )
                    shown_sigs.add(sig)

        avoid_sigs = set()
        if dominant_sig is not None and dominant_share >= 0.6:
            avoid_sigs.add(dominant_sig)
        if streak >= 3 and last_sig:
            avoid_sigs.add(last_sig)
        if avoid_sigs:
            diverse_candidate = _best_feasible_candidate(
                sig_stats,
                exclude_sigs=avoid_sigs,
            )
            if diverse_candidate:
                adjusted, _, avg, count, sig, _, _ = diverse_candidate
                if sig not in shown_sigs:
                    note = _signature_note(sig)
                    lines.append(
                        f"- Diversity-safe alternative: {self._format_signature(sig)} "
                        f"(score {adjusted:.2f}, avg {avg:.2f}, n={count}){note}"
                    )
                    shown_sigs.add(sig)

        if dominant_sig is not None and dominant_share >= 0.6 and len(signature_counts) > 1:
            lines.append(
                f"- Diversity guard: {dominant_share:.2f} of recent actions share "
                f"{self._format_signature(dominant_sig)}; consider mixing tags."
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
    def agent_code_decision(self, member_id) -> None:
        """
        Asks GPT for directly executable Python code, stores it in a dictionary keyed by member_id.
        The code will define a function agent_action(execution_engine, member_id), 
        which references attributes that actually exist.
        """
        member = self.current_members[member_id]
        member_key = member.id
        if not hasattr(self, "_decision_messages"):
            self._decision_messages = {}
        # Gather relationship info
        relations = self.parse_relationship_matrix(self.relationship_dict)
        features = self.get_current_member_features()
        round_context = self._compute_round_context()
        strategy_state = self._summarize_strategy_state(member)
        population_action_mix = self._summarize_population_action_mix()
        population_strategy_profile = self._summarize_population_strategy_diversity()

        current_stats = {
            "vitality": member.vitality,
            "cargo": member.cargo,
            "land": member.land_num,
            "survival_chance": self.compute_survival_chance(member),
        }
        relationship_context = self._compute_relationship_context(member_id)
        relationship_signals = self._summarize_relationship_signals(member_id)

        # Summaries of past code
        code_memory = self.get_code_memory_summary(
            member_id,
            current_stats=current_stats,
            current_round_context=round_context,
            current_relation_context=relationship_context,
        )
        action_feasibility = self._summarize_action_feasibility(member_id)
        neighbor_index_map = self._summarize_neighbor_index_map(member)
        experiment_log = self._summarize_experiment_log(member_id)

        # Track relationships for logging
        self.execution_history['relationships'][
            f'round_{len(self.execution_history["rounds"])}_{member_id}'
        ] = relations

        past_performance = "No previous actions"
        perf_list = self.performance_history.get(member_key, [])
        if perf_list:
            avg_perf = sum(perf_list) / len(perf_list)
            recent_perf = perf_list[-3:]
            past_performance = (
                f"Average performance change: {avg_perf:.2f}. "
                f"Recent changes (last {len(recent_perf)}): {recent_perf}"
            )

        # Get previous errors for this member
        previous_errors = [
            e for e in self.execution_history['errors'] 
            if e['member_id'] == member_id
        ]
        error_context = "No previous execution errors"
        if previous_errors:
            last_error = previous_errors[-1]
            error_context = (
                f"Last execution error (Round {last_error['round']}):\n"
                f"Error type: {last_error['error']}\n"
                f"Code that caused error:\n{last_error.get('code', '')}"
            )

        # Get any received messages (and clear them)
        received_messages = self.messages.pop(member_id, [])
        self._decision_messages[member_id] = list(received_messages)
        message_context = self._summarize_received_messages(received_messages)
        self_action_mix = self._summarize_member_action_mix(member_id)
        diversity_guidance = self._summarize_diversity_guidance(member)
        learning_trend = self._summarize_learning_trend(member_id)
        execution_reliability = self._summarize_execution_reliability(member_id)
        action_impact = self._summarize_action_impact()
        personal_action_impact = self._summarize_personal_action_impact(member_id)
        exploration_posture = self._summarize_exploration_posture(
            member,
            member_id,
            round_context=round_context,
        )
        contextual_strategy_summary = self._summarize_contextual_strategy(
            member_id,
            current_stats=current_stats,
            current_round_context=round_context,
            current_relation_context=relationship_context,
        )
        strategy_recommendations = self._summarize_strategy_recommendations(
            member,
            member_id,
        )
        
        # Build a clarifying prompt to reduce hallucinations
        prompt = f"""
        [Communication Strategy]
        You can communicate with other members using:
        execution_engine.send_message(your_index, recipient_index, "message")
        execution_engine.send_message_by_id(your_id, recipient_id, "message")
        Example usage:
        - Coordinate attacks: "Let's attack member_3 together"
        - Negotiate trades: "I'll give you cargo if you protect me"
        - Share information: "Member_2 is low on vitality"

        [Received Messages]
        {message_context}

        [Previous code execution context]
        {error_context}

        [Current task]
        You are member_{member.id} in a survival game. 
        Write a Python function named agent_action(execution_engine, member_id) 
        that uses execution_engine instead of self to carry out your plan of survival.

        [Critical constraints]
        - Carefully analyze previous errors shown above and avoid repeating them
        - Never target yourself (member_{member.id}) in any action
        - Verify member has land before using bear() or offer_land()
        - Check member IDs exist before referencing them
        - Ensure all matrix indices are valid
        - member_id passed into agent_action is the current_members index (surviver_id), not necessarily member.id
        - member.current_clear_list contains stable member IDs; map them before indexing current_members
        - current_members is a LIST accessed by index, not a dictionary
        - Access members using execution_engine.current_members[index]
        - Check if index exists: if index < len(execution_engine.current_members)
        
        IMPORTANT: Here are the attributes and methods actually available:

        1) Each member object has:
            • member.id (int): Stable unique ID for the member (does not change)
            • member.surviver_id (int): Current index in current_members (changes after deaths)
            • member.vitality (float)
            • member.cargo (float)
            • member.land_num (int): Number of land tiles owned
            • member.overall_productivity (float)
            • member.age (float)
            • member.current_clear_list (List[int]) - stable member IDs of neighbors or cleared adjacents
        2) The relationships are stored in execution_engine.relationship_dict, NOT in "relationship_history".
            Use the arrays in relationship_dict, or rely on the summary below (the 'relations' variable).
            The keys are: 'victim', 'benefit', 'benefit_land'.
            Example usage:
                matrix = execution_engine.relationship_dict['victim']
                # matrix[i, j] indicates how many times member_i was attacked by member_j (if > 0).
        3) The parse_relationship_matrix method is used to produce a summary of relationships as a list of strings.
            For example, 'member_2 was attacked by member_1 (value=3.00)'.
        4) You can use these methods on execution_engine:
            • execution_engine.attack(member1, member2)
            • execution_engine.offer(member1, member2, True)
            • execution_engine.offer_land(member1, member2, True)
            • execution_engine.bear(member1, member2)
            • execution_engine.expand(member1)
            • execution_engine.send_message(sender_index, recipient_index, "message")
            • execution_engine.send_message_by_id(sender_id, recipient_id, "message")
            • execution_engine.get_member_by_id(member_id) -> Member or None
            • execution_engine.resolve_member_index_by_id(member_id) -> index or None
        5) The members are accessed by execution_engine.current_members[index].
            Example: execution_engine.current_members[2] returns the member with surviver_id=2.
        6) DO NOT reference 'member.member_id' or 'member.self_vitality'. Use member.id, member.vitality, etc.

        Current status (features of all members):
        {features}

        Relationship summary (parsed from relationship_dict):
        {relations}

        Relationship signals (aggregated):
        {relationship_signals}

        Code Memory and Previous Performance:
        {code_memory}

        Performance history:
        {past_performance}

        Round context (global signals):
        {round_context}

        {action_feasibility}

        {neighbor_index_map}

        Strategy memory (member-level):
        {strategy_state}

        Self action mix (recent decisions):
        {self_action_mix}

        Population action mix (last round):
        {population_action_mix}

        Population strategy diversity snapshot:
        {population_strategy_profile}

        Diversity guidance:
        {diversity_guidance}

        Exploration posture:
        {exploration_posture}

        Contextual strategy cues:
        {contextual_strategy_summary}

        {experiment_log}

        {learning_trend}

        Execution reliability:
        {execution_reliability}

        Personal action impact (recent, member-specific):
        {personal_action_impact}

        {strategy_recommendations}

        {action_impact}

        Based on the previous code performance, adapt and improve the strategy.
        If a previous strategy worked well (high performance), consider building upon it.
        If it failed, try a different approach.
        Balance short-term survival with longer-term vitality/cargo/land outcomes rather than optimizing a single metric.
        Use your strategy profile as a soft identity anchor (a bias, not a rule); adapt when evidence suggests.
        Preserve agent-level diversity: avoid repeating identical action signatures if you are on a streak; if a single action dominates the population mix, consider alternative actions that still fit your situation.

        Example minimal code:

        def agent_action(execution_engine, member_id):
            # Access your own data
            if member_id < len(execution_engine.current_members):
                me = execution_engine.current_members[member_id]
            my_vitality = me.vitality
            # Possibly parse victim or benefit info from relationship_dict
            total_times_I_was_attacked = execution_engine.relationship_dict['victim'][me.surviver_id].sum()

            # Make a decision
            if my_vitality < 50.0:
                execution_engine.offer(me, execution_engine.current_members[0], True)
            else:
                execution_engine.attack(me, execution_engine.current_members[1])

        Return only the code, no extra text or explanation.
        """

        try:
            completion = openai.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            code_result = completion.choices[0].message.content.strip()

            # Clean and store the code
            code_result = self.clean_code_string(code_result)

            # Log the generated code
            round_num = len(self.execution_history['rounds'])
            if round_num not in self.execution_history['generated_code']:
                self.execution_history['generated_code'][round_num] = {}

            self.execution_history['generated_code'][round_num][member_id] = {
                'code': code_result,
                'features_at_generation': features.to_dict('records'),
                'relationships_at_generation': relations
            }

            print(f"\nGenerated code for Member {member_id}:")
            print(code_result)

            if not hasattr(self, 'agent_code_by_member'):
                self.agent_code_by_member = {}
            self.agent_code_by_member[member_id] = code_result

        except Exception as e:
            error_info = {
                'round': len(self.execution_history['rounds']),
                'member_id': member_id,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.execution_history['errors'].append(error_info)
            print(f"Error generating code for member {member_id}:")
            print(traceback.format_exc())
            self._logger.error(f"GPT Code Generation Error (member {member_id}): {e}")

    # -- NEW METHOD TO EXECUTE AGENT CODE SAFELY --
    def execute_code_actions(self) -> None:
        """Executes all code that the agents wrote (if any) using a restricted namespace."""
        if not hasattr(self, 'agent_code_by_member'):
            self._logger.warning("No agent code to execute.")
            return
        if not hasattr(self, "_decision_messages"):
            self._decision_messages = {}

        round_num = len(self.execution_history['rounds'])
        round_data = {
            'timestamp': datetime.now().isoformat(),
            'actions': [],
            'performance_changes': {},
            'survival_changes': {},
            'messages': {}  # Add message tracking
        }
        round_entries = []

        for member_id, code_str in self.agent_code_by_member.items():
            if not code_str:
                continue

            print(f"\nExecuting code for Member {member_id}:")
            print(code_str)

            # Track old stats before executing
            old_survival = self.compute_survival_chance(self.current_members[member_id])
            old_stats = {
                'vitality': self.current_members[member_id].vitality,
                'cargo': self.current_members[member_id].cargo,
                'land': self.current_members[member_id].land_num,
                'survival_chance': old_survival
            }
            relationship_context = self._compute_relationship_context(member_id)

            error_occurred = None
            cleaned_code = self.clean_code_string(code_str)
            planned_signature = self._extract_action_signature(cleaned_code)
            feasibility = self._compute_action_feasibility(
                self.current_members[member_id]
            )
            planned_infeasible = [
                tag for tag in planned_signature if not feasibility.get(tag, True)
            ]
            try:
                # Track messages for this member
                messages_sent = []
                # Get received messages for this member BEFORE clearing
                received_messages = self._decision_messages.get(
                    member_id, self.messages.get(member_id, [])
                )
                
                # Modified exec environment with message tracking
                had_instance_override = 'send_message' in self.__dict__
                original_send_message = self.send_message

                def tracked_send_message(sender, recipient, msg):
                    nonlocal messages_sent
                    recipient_label = self.resolve_member_id(recipient)
                    if recipient_label is None:
                        recipient_label = recipient
                    messages_sent.append((recipient_label, msg))
                    return original_send_message(sender, recipient, msg)

                self.send_message = tracked_send_message
                    
                local_env = {
                    'execution_engine': self,
                    'send_message': tracked_send_message
                }
                
                # Execute the code in a way that makes the function accessible
                exec(cleaned_code, local_env)

                if 'agent_action' in local_env and callable(local_env['agent_action']):
                    print(f"Executing agent_action() for Member {member_id}")
                    # Pass self as execution_engine and member_id
                    local_env['agent_action'](self, member_id)
                else:
                    error_occurred = "No valid agent_action() found"
                    print(f"No valid agent_action() found for Member {member_id}")
                    self._logger.warning(f"No valid agent_action() found for member {member_id}.")

            except Exception as e:
                error_occurred = str(e)
                error_info = {
                    'round': round_num,
                    'member_id': member_id,
                    'code': code_str,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.execution_history['errors'].append(error_info)
                print(f"Error executing code for member {member_id}:")
                print(traceback.format_exc())
                self._logger.error(f"Error executing code for member {member_id}: {e}")
            finally:
                if had_instance_override:
                    self.send_message = original_send_message
                else:
                    if 'send_message' in self.__dict__:
                        del self.send_message

            # Track changes
            new_survival = self.compute_survival_chance(self.current_members[member_id])
            new_stats = {
                'vitality': self.current_members[member_id].vitality,
                'cargo': self.current_members[member_id].cargo,
                'land': self.current_members[member_id].land_num,
                'survival_chance': new_survival
            }
            member_key = self._member_storage_key(member_id)
            if member_key is None:
                member_key = member_id
            
            performance_change = new_survival - old_survival
            delta = {
                'vitality': new_stats['vitality'] - old_stats['vitality'],
                'cargo': new_stats['cargo'] - old_stats['cargo'],
                'land': new_stats['land'] - old_stats['land'],
                'survival_chance': new_stats['survival_chance'] - old_stats['survival_chance'],
            }
            reward_score = self._compute_reward_score(delta)
            survival_score = self._compute_survival_score(performance_change)
            balanced_score = self._compute_balanced_score(delta, performance_change)
            action_summary = self._summarize_member_round_actions(member_id)
            round_context = self._compute_round_context()
            message_summary = None
            if received_messages or messages_sent:
                message_stats = self._extract_message_stats(received_messages)
                message_summary = {
                    'received_count': message_stats.get('received_count', 0),
                    'sent_count': len(messages_sent),
                    'received_senders': message_stats.get('sender_counts', {}),
                    'received_intents': message_stats.get('intent_counts', {}),
                    'received_sample': [
                        self._truncate_message(msg) for msg in received_messages[-2:]
                    ],
                    'sent_sample': [
                        (recipient, self._truncate_message(msg))
                        for recipient, msg in messages_sent[-2:]
                    ],
                }
            executed_signature = self._extract_executed_signature(
                action_summary,
                message_summary,
            )
            signature_basis = executed_signature or planned_signature
            auto_metrics = {
                "delta_survival": performance_change,
                "delta_vitality": delta.get("vitality", 0.0),
                "delta_cargo": delta.get("cargo", 0.0),
                "delta_land": delta.get("land", 0.0),
                "balanced_score": balanced_score,
            }
            auto_context_tags = {
                "land_scarcity": round_context.get("land_scarcity"),
                "resource_pressure": round_context.get("resource_pressure"),
                "dominant_action_share": round_context.get("dominant_action_share"),
                "net_support": relationship_context.get("net_support"),
            }
            self._auto_update_strategy_memory(
                self.current_members[member_id],
                round_num,
                signature_basis,
                auto_metrics,
                context_tags=auto_context_tags,
            )
            strategy_notes = self._collect_strategy_notes(
                self.current_members[member_id]
            )
            signature_novelty = self._compute_signature_novelty(
                member_id,
                signature_basis,
            )
            action_count = 0
            action_types = []
            if action_summary:
                action_count = sum(
                    summary.get("outgoing_count", 0)
                    for summary in action_summary.values()
                )
                action_types = [
                    action for action, summary in action_summary.items()
                    if summary.get("outgoing_count", 0) > 0
                ]

            # Store in code memory
            if member_key not in self.code_memory:
                self.code_memory[member_key] = []
                
            memory_entry = {
                'code': code_str,
                'performance': performance_change,
                'signature': planned_signature,
                'executed_signature': executed_signature,
                'signature_novelty': signature_novelty,
                'reward_score': reward_score,
                'survival_score': survival_score,
                'balanced_score': balanced_score,
                'context': {
                    'old_stats': old_stats,
                    'new_stats': new_stats,
                    'delta': delta,
                    'action_summary': action_summary,
                    'action_count': action_count,
                    'action_types': action_types,
                    'round_context': round_context,
                    'relationship_context': relationship_context,
                    'message_summary': message_summary,
                    'feasibility': feasibility,
                    'planned_infeasible_tags': planned_infeasible,
                    'round': round_num
                }
            }
            if strategy_notes:
                memory_entry['strategy_notes'] = strategy_notes
            if error_occurred:
                memory_entry['error'] = error_occurred
                
            self.code_memory[member_key].append(memory_entry)

            # Store performance in history
            if member_key not in self.performance_history:
                self.performance_history[member_key] = []
            self.performance_history[member_key].append(performance_change)

            # Log changes for this round
            round_data['performance_changes'][member_id] = performance_change
            round_data['survival_changes'][member_id] = new_survival
            round_data['actions'].append({
                'member_id': member_id,
                'code_executed': code_str,
                'old_stats': old_stats,
                'new_stats': new_stats,
                'delta': delta,
                'action_summary': action_summary,
                'action_count': action_count,
                'action_types': action_types,
                'round_context': round_context,
                'performance_change': performance_change,
                'messages_sent': messages_sent
            })

            # Log messages in round data
            round_data['messages'][member_id] = {
                'received': received_messages,
                'sent': messages_sent
            }
            round_entries.append(
                {
                    "member_id": member_id,
                    "memory_entry": memory_entry,
                    "reward_score": reward_score,
                    "survival_score": survival_score,
                    "balanced_score": balanced_score,
                }
            )

        self._apply_population_baseline(round_entries, round_data)
        self.execution_history['rounds'].append(round_data)
        
        # Save execution history after each round
        self.save_execution_history()

        # Clear cached decision messages after logging
        self._decision_messages = {}
        
        # Clear code after execution
        self.agent_code_by_member = {}

        # Add this line to preserve messages until next decision phase
        self.messages = {k:v for k,v in self.messages.items() if v}  # Only keep non-empty lists

    def save_execution_history(self):
        """Save the execution history to a JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('execution_histories', exist_ok=True)
        filename = os.path.join('execution_histories', f'execution_history_{timestamp}.json')
        
        try:
            # Convert any non-serializable objects to strings
            history_copy = json.loads(json.dumps(self.execution_history, default=str))
            
            with open(filename, 'w') as f:
                json.dump(history_copy, f, indent=2)
            print(f"\nExecution history saved to {filename}")
        except Exception as e:
            print(f"Error saving execution history: {e}")
            print(traceback.format_exc())

    def print_agent_performance(self):
        """
        Print or log detailed performance metrics for each agent
        including relationship changes and survival probability changes.
        """
        for member_id, performance_list in self.performance_history.items():
            member_idx = self.resolve_member_index(member_id, prefer_index=False)
            if member_idx is None:
                continue  # Skip dead members or missing indices

            member = self.current_members[member_idx]
            current_survival = self.compute_survival_chance(member)
            avg_perf = sum(performance_list) / len(performance_list) if performance_list else 0
            
            # Get relationship summary
            relations = self.parse_relationship_matrix(self.relationship_dict)
            relation_summary = [r for r in relations if f"member_{member_idx}" in r]
            
            self._logger.info(
                f"\nMember {member_id} Status Report:"
                f"\n- Current survival chance: {current_survival:.2f}"
                f"\n- Performance changes: {performance_list}"
                f"\n- Average performance: {avg_perf:.2f}"
                f"\n- Recent relationships:\n  " + "\n  ".join(relation_summary[-3:])  # Last 3 relationships
            )

    def compute_survival_chance(self, member):
        """
        Compute a more sophisticated survival chance based on:
        - Member's own attributes (vitality, cargo)
        - Relationships with others (beneficial/hostile connections)
        - Current neighborhood situation
        Returns a float representing survival probability
        """
        # Base survival from own attributes
        base_survival = member.vitality + member.cargo
        
        # Get relationship bonuses/penalties
        relationship_modifier = 0
        for relation_type, matrix in self.relationship_dict.items():
            if member.id >= matrix.shape[0]:  # Prevent index out of bounds
                continue
            if relation_type == 'benefit':
                # Use nansum to handle potential NaN values
                relationship_modifier += np.nansum(matrix[member.id, :]) * 0.2
            elif relation_type == 'victim':
                # Use nansum to handle potential NaN values
                relationship_modifier -= np.nansum(matrix[member.id, :]) * 0.3
        
        # Neighborhood safety (more neighbors = more risk/opportunity)
        # Add type conversion to ensure numerical operation
        neighbor_count = float(len(member.current_clear_list))
        neighborhood_modifier = -0.1 * neighbor_count
        
        # Ensure final value is finite
        survival_score = base_survival + relationship_modifier + neighborhood_modifier
        return np.nan_to_num(survival_score, nan=50.0)  # Default to 50 if calculation fails

    def send_message(self, sender_id: int, recipient_id: int, message: str):
        """Allow agents to send messages to each other"""
        sender_index = self.resolve_member_index(sender_id, prefer_index=True)
        recipient_index = self.resolve_member_index(recipient_id, prefer_index=True)

        sender_label = sender_id
        if isinstance(sender_id, Member):
            sender_label = sender_id.id
        elif sender_index is not None and 0 <= sender_index < len(self.current_members):
            sender_label = self.current_members[sender_index].id

        recipient_label = recipient_id
        if isinstance(recipient_id, Member):
            recipient_label = recipient_id.id
        elif recipient_index is not None and 0 <= recipient_index < len(self.current_members):
            recipient_label = self.current_members[recipient_index].id

        print(f"[MSG] Member {sender_label} -> Member {recipient_label}: {message!r}")
        # Add validation check
        if recipient_index is not None and 0 <= recipient_index < len(self.current_members):
            if recipient_index not in self.messages:
                self.messages[recipient_index] = []
            self.messages[recipient_index].append(
                f"From member_{sender_label}: {message}"
            )
        else:
            print(f"Invalid message recipient {recipient_id} from member {sender_id}")

    def send_message_by_id(self, sender_id: int, recipient_id: int, message: str):
        """Send a message using stable member.id values instead of indices."""
        recipient_index = self.resolve_member_index_by_id(recipient_id)
        if recipient_index is None:
            print(f"Invalid message recipient {recipient_id} from member {sender_id}")
            return
        sender_index = self.resolve_member_index_by_id(sender_id)
        sender_ref = sender_index if sender_index is not None else sender_id
        self.send_message(sender_ref, recipient_index, message)

    def print_agent_messages(self):
        """Print message communication between agents"""
        print("\n=== Agent Message History ===")
        for round_idx, round_data in enumerate(self.execution_history['rounds']):
            print(f"\nRound {round_idx + 1} ({round_data['timestamp']}):")
            if not round_data['messages']:
                print("  No messages exchanged")
                continue
            
            for member_id, comm in round_data['messages'].items():
                print(f"Member {member_id}:")
                if comm['received']:
                    print("  Received:")
                    for msg in comm['received']:
                        print(f"    - {msg}")
                if comm['sent']:
                    print("  Sent:")
                    for recipient, msg in comm['sent']:
                        print(f"    -> Member {recipient}: {msg}")

def main():
    from Leviathan.Island import Island
    from Leviathan.Member import Member
    from Leviathan.Analyzer import Analyzer
    from time import time
    from Leviathan.Land import Land
    from utils import save
    import os

    rng = np.random.default_rng()
    path = save.datetime_dir("../data")
    exec = IslandExecution(5, (5, 5), path, 2023)
    IslandExecution._RECORD_PERIOD = 1
    Member._DECISION_BACKEND = 'inner product'
    Member._PARAMETER_INFLUENCE = 0

    action_prob = 0.5
    round_num = 10
    
    for round_i in range(round_num):
        print(f"\n{'='*50}")
        print(f"=== Round {round_i + 1} ===")
        print(f"{'='*50}")
        
        exec.new_round()
        exec.get_neighbors()
        exec.produce()
        
        print("\nGenerating agent decisions...")
        for i in range(len(exec.current_members)):
            exec.agent_code_decision(i)
        
        print("\nExecuting agent actions...")
        exec.execute_code_actions()
        exec.consume()
        
        print("\nPerformance Report:")
        exec.print_agent_performance()
        
        # Add this line to show messages
        exec.print_agent_messages()
        
        exec.log_status(action=True, log_instead_of_print=False)
        
        print(f"\nSurviving members at end of round: {len(exec.current_members)}")
        for member in exec.current_members:
            survival_chance = exec.compute_survival_chance(member)
            print(f"Member {member.id}: Vitality={member.vitality:.2f}, "
                  f"Cargo={member.cargo:.2f}, Survival={survival_chance:.2f}")

if __name__ == "__main__":
    main()
