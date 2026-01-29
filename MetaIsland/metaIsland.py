from typing import List, Tuple, Optional
import ast
import re
import math
import numpy as np
import pandas as pd
import json
from datetime import datetime
import traceback
import os
from collections import defaultdict, Counter
import inspect
import openai
import asyncio

from MetaIsland.base_island import Island

from MetaIsland.agent_code_decision import _agent_code_decision
from MetaIsland.agent_mechanism_proposal import _agent_mechanism_proposal
from MetaIsland.analyze import _analyze

# Import new systems
from MetaIsland.graph_engine import ExecutionGraph
from MetaIsland.contracts import ContractEngine
from MetaIsland.physics import PhysicsEngine
from MetaIsland.judge import Judge
from MetaIsland.nodes import (
    NewRoundNode, ProduceNode, ConsumeNode, LogStatusNode,
    AnalyzeNode, ProposeMechanismNode, AgentDecisionNode, ExecuteActionsNode,
    JudgeNode, ExecuteMechanismsNode, ContractNode, EnvironmentNode
)

from dotenv import load_dotenv
load_dotenv()

import aisuite as ai

client = ai.Client()

from MetaIsland.model_router import model_router

provider, model_id = model_router("default")
class IslandExecution(Island):
    def __init__(self, 
        init_member_number: int,
        land_shape: Tuple[int, int],
        save_path: str,
        random_seed: Optional[int] = None,
        action_board: List[List[Tuple[str, int, int]]] = None,
        agent_modifications: dict = None
    ):
        # Create directories for saving generated code
        self.code_save_path = os.path.join(save_path, 'generated_code')
        os.makedirs(self.code_save_path, exist_ok=True)
        
        # Create subdirectories for different types of code
        self.agent_code_path = os.path.join(self.code_save_path, 'agent_actions')
        self.mechanism_code_path = os.path.join(self.code_save_path, 'mechanisms')
        self.analysis_code_path = os.path.join(self.code_save_path, 'analysis')
        self.execution_history_path = os.path.join(self.code_save_path, 'execution_histories')
        os.makedirs(self.agent_code_path, exist_ok=True)
        os.makedirs(self.mechanism_code_path, exist_ok=True)
        os.makedirs(self.analysis_code_path, exist_ok=True)
        os.makedirs(self.execution_history_path, exist_ok=True)
        
        # Add version tracking
        self._VERSION = "2.1"
        
        # Add base class code storage
        self.base_class_code = self._load_base_class_code()
        
        # Add agent modification tracking
        self.agent_modifications = {
            'pre_init': [],
            'post_init': [],

        }
            
        super().__init__(
            init_member_number,
            land_shape,
            save_path,
            random_seed
        )
        
        self.performance_history = {}  # {member_id: [list_of_performance_metrics]}
        self.round_performance_history = {}  # {member_id: [per-round survival deltas]}
        
        # Add code memory tracking
        self.code_memory = {}  # {member_id: [{'code': str, 'performance': float, 'context': dict}]}
        
        # Add execution history tracking
        self.execution_history = {
            'rounds': []
        }

        # Add message storage
        self.messages = {}  # {member_id: [list_of_messages]}
        # Track per-round message snapshots so multiple prompts can share context
        self._message_snapshot_round = {}
        self._message_snapshot_len = {}

        # Initialize code storage
        self.agent_code_by_member = {}

        self.island_ideology = ""

        self.voting_box = {}

        # Initialize new systems
        self.graph = ExecutionGraph()
        self.contracts = ContractEngine()
        self.physics = PhysicsEngine()
        self.judge = Judge(model_name="default")

        # Setup default execution graph
        self._setup_default_graph()

        # Track round number for graph context
        self.round_number = 0

    def new_round(self):
        """
        Initialize a new round in the execution history with a structured record.
        """
        # If the parent class has a new_round(), call it.
        if hasattr(super(), "new_round"):
            super().new_round()
            
        round_record = {
            "round_number": len(self.execution_history['rounds']) + 1,
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
            "analysis_cards": {},
            "agent_actions": [],      # List of agent code execution details
            "agent_messages": {},     # Dictionary keyed by member_id
            "population_strategy_profile": None,
            "mechanism_modifications": {
                "attempts": [],       # Proposed modifications this round
                "executed": []        # Those that have been successfully executed
            },
            "errors": {
                "agent_code_errors": [],
                "mechanism_errors": [],
                "analyze_code_errors": {}
            },
            "relationships": {},       # Relationships logged per member (if needed)
            "generated_code": {}
        }
        self.execution_history['rounds'].append(round_record)
    
    def _load_base_class_code(self) -> dict:
        """Load the source code of base classes as strings."""
        base_code = {}
        
        # Get the source code for Island
        try:
            base_code['base_island'] = inspect.getsource(Island)
        except Exception as e:
            base_code['base_island'] = f"Error loading Island code: {str(e)}"
            
        # Get the source code for Land
        try:
            from MetaIsland.base_land import Land
            base_code['base_land'] = inspect.getsource(Land)
        except Exception as e:
            base_code['base_land'] = f"Error loading Land code: {str(e)}"
            
        # Get the source code for Member
        try:
            from MetaIsland.base_member import Member
            base_code['base_member'] = inspect.getsource(Member)
        except Exception as e:
            base_code['base_member'] = f"Error loading Member code: {str(e)}"
            
        self.base_code = base_code
        ordered_keys = ["base_island", "base_land", "base_member"]
        formatted = []
        for key in ordered_keys:
            if key in base_code:
                formatted.append(f"[{key}]\n{base_code[key]}")
        for key, value in base_code.items():
            if key not in ordered_keys:
                formatted.append(f"[{key}]\n{value}")
        return "\n\n".join(formatted)

    def offer(self, member_1, member_2):
        super()._offer(member_1, member_2)
        
    def offer_land(self, member_1, member_2):
        super()._offer_land(member_1, member_2)
        
    def attack(self, member_1, member_2):
        super()._attack(member_1, member_2)

    def bear(self, member_1, member_2):
        super()._bear(member_1, member_2)
    
    def expand(self, member_1):
        super()._expand(member_1)

    def parse_relationship_matrix(self, relationship_dict):
        """
        Parse and return a human-readable summary of the relationship matrices.
        
        :param relationship_dict: A dictionary with keys like 'victim', 'benefit', 'benefit_land'
                                 each containing a NxN numpy array of relationships.
        :return: A list of strings describing the relationships.
        """
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
                        member_i_id = self.current_members[i].id
                        member_j_id = self.current_members[j].id
                        statement = (f"{rel_map[relation_type]} "
                                     f"(value={val:.2f})")
                        # Replace {i} with actual index+1 (or keep zero-based)
                        # Same for {j}
                        statement = statement.format(i=member_i_id, j=member_j_id)
                        summary.append(statement)
        
        return summary
    
    def get_current_member_features(self) -> pd.DataFrame:
        """Collect features for all current members"""
        feature_rows = []
        
        for member in self.current_members:
            # Get self attributes
            feature_row = {
                "self_productivity": member.overall_productivity,
                "self_vitality": member.vitality, 
                "self_cargo": member.cargo,
                "self_age": member.age,
                "self_neighbor": len(member.current_clear_list),
                "member_id": member.id
            }
            feature_rows.append(feature_row)
                
        return pd.DataFrame(feature_rows)

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

    def _auto_update_strategy_memory(
        self,
        member,
        round_num: int,
        signature: tuple,
        metrics: dict,
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
        ctx_text = self._format_context_tags(context_tags) if context_tags else "none"
        note = (
            f"auto r{round_num} sig={sig_text} "
            f"d_surv={_fmt(metrics.get('delta_survival'), 2)} "
            f"d_vit={_fmt(metrics.get('delta_vitality'), 1)} "
            f"d_cargo={_fmt(metrics.get('delta_cargo'), 1)} "
            f"d_rel={_fmt(metrics.get('delta_relation_balance'), 2)} "
            f"d_land={_fmt(metrics.get('delta_land'), 1)} "
            f"ctx={ctx_text}"
        )

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

    def _coerce_card_text(self, value) -> str:
        """Coerce a strategy card field into a compact string."""
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value).strip()
        return text

    def _coerce_card_list(self, value) -> List[str]:
        """Coerce a strategy card list field into a list of strings."""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            items = list(value)
        elif isinstance(value, str):
            if "," in value:
                items = value.split(",")
            elif "|" in value:
                items = value.split("|")
            else:
                text = value.strip()
                return [text] if text else []
        else:
            items = [value]
        return [str(item).strip() for item in items if str(item).strip()]

    def _normalize_strategy_card(self, raw: dict) -> Optional[dict]:
        """Normalize strategy card keys to a consistent schema."""
        if not isinstance(raw, dict):
            return None

        def pick(*keys):
            for key in keys:
                if key in raw:
                    return raw[key]
            return None

        card = {
            "hypothesis": self._coerce_card_text(
                pick("hypothesis", "hyp", "thesis")
            ),
            "baseline_signature": self._coerce_card_list(
                pick("baseline_signature", "baseline_tags", "baseline", "baseline_action")
            ),
            "variation_signature": self._coerce_card_list(
                pick("variation_signature", "variation_tags", "variation", "variant_action")
            ),
            "success_metrics": self._coerce_card_list(
                pick("success_metrics", "metrics", "success", "targets")
            ),
            "guardrails": self._coerce_card_list(
                pick("guardrails", "constraints", "stops", "stop_conditions")
            ),
            "coordination": self._coerce_card_list(
                pick("coordination", "coordination_asks", "messages", "coordination_plan")
            ),
            "memory_note": self._coerce_card_text(
                pick("memory_note", "note", "memory", "log_note")
            ),
            "diversity_note": self._coerce_card_text(
                pick("diversity_note", "diversity", "anti_monoculture")
            ),
        }

        confidence = pick("confidence", "conf", "certainty")
        if confidence is not None:
            try:
                card["confidence"] = float(confidence)
            except (TypeError, ValueError):
                card["confidence"] = self._coerce_card_text(confidence)

        if not any(value for value in card.values()):
            return None
        return card

    def _extract_strategy_card(self, text: str) -> Optional[dict]:
        """Extract a JSON strategy card from analysis text."""
        if not text:
            return None
        match = re.search(
            r"```json\s*(\{.*?\})\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        if not match:
            match = re.search(
                r"STRATEGY_CARD_JSON\s*:?\s*(\{.*?\})",
                text,
                flags=re.IGNORECASE | re.DOTALL
            )
        if not match:
            return None
        raw = match.group(1).strip()
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                return None
        if not isinstance(parsed, dict):
            return None
        return self._normalize_strategy_card(parsed)

    def _format_strategy_card_brief(self, card: dict, limit: int = 180) -> str:
        """Format a compact, single-line summary of a strategy card."""
        if not card:
            return ""
        parts = []
        hypothesis = card.get("hypothesis")
        if hypothesis:
            parts.append(f"hyp={hypothesis}")
        baseline = card.get("baseline_signature") or []
        if baseline:
            parts.append(f"base={','.join(baseline)}")
        variation = card.get("variation_signature") or []
        if variation:
            parts.append(f"var={','.join(variation)}")
        success = card.get("success_metrics") or []
        if success:
            parts.append(f"success={','.join(success)}")
        guardrails = card.get("guardrails") or []
        if guardrails:
            parts.append(f"guards={','.join(guardrails)}")
        diversity_note = card.get("diversity_note")
        if diversity_note:
            parts.append(f"diversity={diversity_note}")
        confidence = card.get("confidence")
        if confidence is not None:
            parts.append(f"conf={confidence}")
        summary = " | ".join(parts)
        return self._truncate_message(summary, limit=limit)

    def _append_strategy_report(
        self,
        member,
        note: str,
        key: str = "reports",
        max_items: int = 6,
    ) -> None:
        """Append a short analysis note to member.strategy_memory."""
        if member is None or not note:
            return
        if not hasattr(member, "strategy_memory") or member.strategy_memory is None:
            member.strategy_memory = {key: []}
        mem = member.strategy_memory
        if isinstance(mem, dict):
            bucket = mem.get(key)
            if not isinstance(bucket, list):
                bucket = []
            bucket.append(note)
            if len(bucket) > max_items:
                del bucket[:-max_items]
            mem[key] = bucket
        elif isinstance(mem, list):
            mem.append(note)
            if len(mem) > max_items:
                del mem[:-max_items]
        else:
            member.strategy_memory = {key: [str(mem), note]}

    def _record_analysis_card(self, member_id: int, analysis_text: str) -> None:
        """Parse and store a strategy card from analysis output."""
        card = self._extract_strategy_card(analysis_text)
        if not card:
            return
        if not self.execution_history.get("rounds"):
            return
        round_data = self.execution_history["rounds"][-1]
        round_data.setdefault("analysis_cards", {})[member_id] = card
        try:
            member = self.current_members[member_id]
        except Exception:
            member = None
        round_num = round_data.get("round_number", len(self.execution_history["rounds"]))
        memory_note = card.get("memory_note")
        if memory_note:
            note = f"analysis r{round_num} | {memory_note}"
        else:
            brief = self._format_strategy_card_brief(card, limit=140)
            note = f"analysis r{round_num} | {brief}" if brief else f"analysis r{round_num}"
        note = self._truncate_message(note, limit=180)
        self._append_strategy_report(member, note)

    def get_analysis_card_summary(self, member_id: int, window: int = 3) -> str:
        """Summarize recent analysis strategy cards for prompt guidance."""
        if not self.execution_history.get("rounds"):
            return "No analysis cards available."
        window = max(1, int(window))
        cards = []
        for round_data in self.execution_history["rounds"][-window:]:
            card = round_data.get("analysis_cards", {}).get(member_id)
            if card:
                round_num = round_data.get("round_number")
                cards.append((round_num, card))
        if not cards:
            return "No analysis cards available."
        lines = ["Recent analysis strategy cards:"]
        for round_num, card in cards:
            label = f"Round {round_num}" if round_num is not None else "Round ?"
            summary = self._format_strategy_card_brief(card, limit=160)
            lines.append(f"- {label}: {summary if summary else 'card parsed'}")
        return "\n".join(lines)

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

    def _parse_message_sender(self, message: str) -> Optional[int]:
        """Parse sender id from message format 'From member_X: ...'."""
        if not message:
            return None
        prefix = "From member_"
        if not str(message).startswith(prefix):
            return None
        remainder = str(message)[len(prefix):]
        sender_token = remainder.split(":", 1)[0].strip()
        try:
            return int(sender_token)
        except ValueError:
            return None

    def _summarize_communication(
        self,
        member_id: int,
        window_rounds: int = 3,
        sample_limit: int = 2,
    ) -> str:
        """Summarize recent message exchanges for coordination context."""
        if not self.execution_history.get('rounds'):
            return "No communication history available."

        total_rounds = len(self.execution_history['rounds'])
        window_rounds = max(1, int(window_rounds))
        start_round = max(1, total_rounds - window_rounds + 1)

        sent_total = 0
        received_total = 0
        recipients = set()
        senders = set()
        recent_received = []
        recent_sent = []

        def push_recent(bucket, item):
            if sample_limit <= 0:
                return
            bucket.append(item)
            if len(bucket) > sample_limit:
                bucket.pop(0)

        for round_data in self.execution_history['rounds'][start_round - 1:]:
            comm = round_data.get('agent_messages', {}).get(member_id)
            if not comm:
                continue
            for msg in comm.get('received', []) or []:
                received_total += 1
                sender_id = self._parse_message_sender(msg)
                if sender_id is not None:
                    senders.add(sender_id)
                push_recent(recent_received, self._truncate_message(msg))
            for recipient_id, msg in comm.get('sent', []) or []:
                sent_total += 1
                recipients.add(recipient_id)
                msg_text = self._truncate_message(msg)
                if recipient_id is not None:
                    msg_text = f"member_{recipient_id}: {msg_text}"
                push_recent(recent_sent, msg_text)

        if sent_total == 0 and received_total == 0:
            return (
                f"Communication summary (rounds {start_round}-{total_rounds}): "
                "no messages exchanged."
            )

        lines = [
            "Communication summary (recent rounds):",
            f"- Window rounds: {start_round}-{total_rounds}",
            f"- Sent: {sent_total} messages to {len(recipients)} unique recipients",
            f"- Received: {received_total} messages from {len(senders)} unique senders",
        ]

        if recipients:
            recipient_list = sorted(recipients)
            display = ", ".join(f"member_{rid}" for rid in recipient_list[:5])
            if len(recipient_list) > 5:
                display += ", ..."
            lines.append(f"- Recent recipients: {display}")
        if senders:
            sender_list = sorted(senders)
            display = ", ".join(f"member_{sid}" for sid in sender_list[:5])
            if len(sender_list) > 5:
                display += ", ..."
            lines.append(f"- Recent senders: {display}")
        if recent_received:
            lines.append(f"- Recent received: {recent_received}")
        if recent_sent:
            lines.append(f"- Recent sent: {recent_sent}")

        return "\n".join(lines)

    def _peek_messages(
        self,
        member_id: int,
        round_num: Optional[int] = None,
        create_snapshot: bool = True,
    ) -> list:
        """Return unread messages for the member without clearing them."""
        if round_num is None:
            round_num = len(self.execution_history.get("rounds", []))

        inbox = self.messages.get(member_id, [])
        snapshot_round = self._message_snapshot_round.get(member_id)

        if create_snapshot and snapshot_round != round_num:
            self._message_snapshot_round[member_id] = round_num
            self._message_snapshot_len[member_id] = len(inbox)
            snapshot_round = round_num

        snapshot_len = self._message_snapshot_len.get(member_id)
        if snapshot_round == round_num and snapshot_len is not None:
            return inbox[:snapshot_len]
        return inbox

    def _consume_message_snapshots(self, round_num: int) -> None:
        """Remove messages that have been surfaced to agents this round."""
        if not self._message_snapshot_round:
            return

        for member_id, snapshot_round in list(self._message_snapshot_round.items()):
            if snapshot_round != round_num:
                continue
            snapshot_len = int(self._message_snapshot_len.get(member_id, 0) or 0)
            inbox = self.messages.get(member_id, [])
            if snapshot_len > 0:
                snapshot_len = min(snapshot_len, len(inbox))
                remaining = inbox[snapshot_len:]
                if remaining:
                    self.messages[member_id] = remaining
                else:
                    self.messages.pop(member_id, None)
            self._message_snapshot_round.pop(member_id, None)
            self._message_snapshot_len.pop(member_id, None)

    def _format_signature(self, signature: tuple) -> str:
        """Format a signature tuple for readable summaries."""
        if not signature:
            return "none"
        return ", ".join(signature)

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
        history = self.code_memory.get(member_id, [])
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
        stats = snapshot.get(member_id)
        if stats is None and 0 <= member_id < len(self.current_members):
            stats = snapshot.get(self.current_members[member_id].id)
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
            communication_line,
        ]

        return "\n".join(lines)

    def get_strategy_profile_summary(self, member_id: int, window: int = 5) -> str:
        """Summarize action signature coverage to encourage strategy diversity."""
        if member_id not in self.code_memory or not self.code_memory[member_id]:
            return "No strategy profile available."

        memory = self.code_memory[member_id]
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
        if member_id not in self.code_memory or not self.code_memory[member_id]:
            return "No contextual strategy data yet."

        memory = self.code_memory[member_id]
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
        if member_id not in self.code_memory or not self.code_memory[member_id]:
            return "No strategy recommendations available."

        memory = self.code_memory[member_id]
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

        std_cutoff = 0.0
        if sig_perf_stats:
            std_values = sorted(stat[1] for stat in sig_perf_stats)
            std_cutoff = std_values[len(std_values) // 2]

        baseline_candidate = None
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

        baseline_sig = None
        if baseline_candidate:
            baseline_sig = baseline_candidate[-1]
        elif best_by_avg is not None:
            baseline_sig = best_by_avg[2]

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
        exploration_bonus = max(0.25, base_exploration_bonus * risk_budget) + exploration_pressure

        sig_stats = []
        for sig, perfs in perf_by_sig.items():
            count = len(perfs)
            avg = sum(perfs) / count if count else 0.0
            ucb = avg + exploration_bonus * math.sqrt(math.log(total + 1) / count)
            sig_stats.append((ucb, avg, count, sig))
        sig_stats.sort(key=lambda x: x[0], reverse=True)

        variation_candidate = None
        if sig_stats:
            scored = []
            for ucb, avg, count, sig in sig_stats:
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
                    (adjusted, ucb, avg, count, sig, overlap, dominance_penalty)
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
        if member_signature_shares:
            lines.append(
                f"- Recent population signature share: avg {avg_signature_share:.2f}, "
                f"dominant rounds {dominant_share_hits}/{len(member_signature_shares)}"
            )
        if exploration_pressure > 0:
            lines.append(
                f"- Exploration pressure: base {base_exploration_bonus:.2f} -> "
                f"{exploration_bonus:.2f} ({', '.join(pressure_reasons)})"
            )

        if baseline_candidate is not None:
            avg, std, count, _novelty, sig = baseline_candidate
            lines.append(
                f"- Suggested baseline (stable): {self._format_signature(sig)} "
                f"(avg {avg:.2f}, std {std:.2f}, n={count})"
            )
        elif best_by_avg is not None:
            avg, count, sig = best_by_avg
            lines.append(
                f"- Suggested baseline: {self._format_signature(sig)} "
                f"(avg {avg:.2f}, n={count})"
            )

        if variation_candidate is not None:
            adjusted, ucb, avg, count, sig, overlap, dominance_penalty = variation_candidate
            detail_bits = [
                f"score {adjusted:.2f}",
                f"ucb {ucb:.2f}",
                f"avg {avg:.2f}",
                f"n={count}",
            ]
            if baseline_sig:
                detail_bits.append(f"overlap {overlap:.2f}")
            if dominance_penalty > 0:
                detail_bits.append(f"pop-penalty {dominance_penalty:.2f}")
            lines.append(
                f"- Risk-adjusted variation: {self._format_signature(sig)} "
                f"({', '.join(detail_bits)})"
            )
        elif sig_stats:
            ucb, avg, count, sig = sig_stats[0]
            lines.append(
                f"- Exploration candidate (UCB): {self._format_signature(sig)} "
                f"(score {ucb:.2f}, avg {avg:.2f}, n={count})"
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

    def format_mechanisms_for_prompt(self, mechanisms, label: str = "Mechanism") -> str:
        """Format mechanism entries for prompt inclusion."""
        if not mechanisms:
            return "No active mechanisms."

        lines = []
        for idx, mech in enumerate(mechanisms, start=1):
            if isinstance(mech, dict):
                name = mech.get('name') or mech.get('title') or mech.get('type') or f"{label} {idx}"
                lines.append(f"[{name}]")
                if mech.get('code'):
                    lines.append(mech['code'])
                else:
                    lines.append(json.dumps(mech, indent=2, default=str))
            else:
                lines.append(f"[{label} {idx}]")
                lines.append(str(mech))
            lines.append("")

        return "\n".join(lines).strip()

    def format_modification_attempts_for_prompt(self, attempts) -> str:
        """Format modification attempt history for prompt inclusion."""
        if not attempts:
            return "No recent modification attempts."

        lines = []
        for idx, attempt in enumerate(attempts, start=1):
            round_num = attempt.get('round')
            status = "ratified" if attempt.get('ratified') else "not ratified"
            header = f"[Attempt {idx}"
            if round_num is not None:
                header += f" | Round {round_num}"
            header += f" | {status}]"
            lines.append(header)

            code = attempt.get('code')
            if code:
                lines.append(code)

            error = attempt.get('error')
            if error:
                lines.append(f"Error: {error}")

            lines.append("")

        return "\n".join(lines).strip()

    def _select_memory_samples(self, memory, max_samples: int):
        """Select a diversity-aware sample of memory entries."""
        if not memory:
            return []

        if len(memory) <= max_samples:
            return [("Recent", mem) for mem in memory]

        indices = list(range(len(memory)))
        by_perf = sorted(indices, key=lambda idx: self._get_memory_performance(memory[idx]))

        signatures = self._get_memory_signatures(memory)
        signature_counts = Counter(signatures)

        best_idx = by_perf[-1]
        worst_idx = by_perf[0]
        recent_idx = indices[-1]
        median_idx = by_perf[len(by_perf) // 2]
        abs_idx = max(indices, key=lambda idx: abs(self._get_memory_performance(memory[idx])))
        rare_idx = min(
            indices,
            key=lambda idx: signature_counts.get(signatures[idx], len(memory) + 1)
        )

        candidates = [
            ("Recent", recent_idx),
            ("Best", best_idx),
            ("Rare", rare_idx),
            ("Worst", worst_idx),
            ("Median", median_idx),
            ("Volatile", abs_idx),
        ]

        selected = []
        seen = set()
        seen_signatures = set()
        for label, idx in candidates:
            if idx in seen:
                continue
            sig = signatures[idx]
            if selected and sig in seen_signatures:
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

    def get_code_memory_summary(self, member_id):
        """Generate a summary of previous code performances for the agent."""
        if member_id not in self.code_memory:
            return "No previous code history."
            
        memory = self.code_memory[member_id]
        if not memory:
            return "No previous code history."
            
        summary = ["Previous code strategies and their outcomes (diversity-aware sample):"]
        summary.append(self._summarize_signature_performance(memory))

        current_tags = self._get_member_context_tags(member_id)
        current_key = self._context_key_from_tags(current_tags)
        if current_tags:
            summary.append(
                f"Current context tags: {self._format_context_tags(current_tags)}"
            )
            if current_key:
                summary.append(f"Current context key: {current_key}")

        match_idx, match_score = self._find_contextual_memory_match(memory, current_tags)
        if match_idx is not None and match_score > 0:
            match_mem = memory[match_idx]
            match_context_key = match_mem.get('context_key') or self._context_key_from_tags(
                match_mem.get('context_tags', {})
            )
            match_round = match_mem.get('context', {}).get('round')
            match_perf = self._get_memory_performance(match_mem)
            match_sig = match_mem.get('signature')
            if match_sig is None:
                match_sig = self._extract_action_signature(match_mem.get('code', ''))
            label_parts = []
            if match_context_key:
                label_parts.append(match_context_key)
            if match_round is not None:
                label_parts.append(f"round {match_round}")
            label_parts.append(f"perf {match_perf:.2f}")
            label_parts.append(f"signature {self._format_signature(match_sig)}")
            metrics = match_mem.get('metrics', {}) or {}
            metric_parts = []
            for key in (
                "delta_survival",
                "delta_vitality",
                "delta_cargo",
                "delta_relation_balance",
                "delta_land",
            ):
                if key in metrics:
                    metric_parts.append(f"{key.replace('delta_', '')}={metrics[key]:.2f}")
            metric_text = f"; deltas {', '.join(metric_parts)}" if metric_parts else ""
            summary.append(
                "Closest context match: "
                f"{' | '.join(label_parts)} (match {match_score:.2f}){metric_text}"
            )

        selected = self._select_memory_samples(memory, max_samples=3)

        for i, (label, mem) in enumerate(selected, start=1):
            perf = self._get_memory_performance(mem)
            context = mem.get('context', {})
            round_num = context.get('round')
            round_suffix = f", Round {round_num}" if round_num is not None else ""

            summary.append(f"\nStrategy {i} [{label}] (Performance: {perf:.2f}{round_suffix}):")
            summary.append(f"Context: {context}")

            context_key = mem.get('context_key')
            if context_key:
                summary.append(f"Context key: {context_key}")
            context_tags = mem.get('context_tags')
            if context_tags:
                tag_text = ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(context_tags.items())
                )
                summary.append(f"Context tags: {tag_text}")
                if current_tags:
                    match = self._context_similarity_score(current_tags, context_tags)
                    if match > 0:
                        summary.append(f"Context match score: {match:.2f}")

            signature = mem.get('signature')
            if signature is None:
                signature = self._extract_action_signature(mem.get('code', ''))
            if signature:
                summary.append(f"Action signature: {', '.join(signature)}")
            if mem.get('signature_novelty') is not None:
                summary.append(f"Signature novelty: {mem.get('signature_novelty'):.2f}")

            message_summary = context.get("message_summary")
            if message_summary:
                summary.append(
                    "Messages: "
                    f"received {message_summary.get('received_count', 0)}, "
                    f"sent {message_summary.get('sent_count', 0)}"
                )
                received_sample = message_summary.get("received_sample") or []
                sent_sample = message_summary.get("sent_sample") or []
                if received_sample:
                    summary.append(f"Received sample: {received_sample}")
                if sent_sample:
                    summary.append(f"Sent sample: {sent_sample}")

            metrics = mem.get('metrics', {})
            if metrics:
                summary.append(
                    "Outcome deltas: "
                    + ", ".join(f"{k}={v:.2f}" for k, v in metrics.items())
                )
                sig_share = metrics.get('round_signature_share')
                if sig_share is not None:
                    try:
                        sig_share_val = float(sig_share)
                    except (TypeError, ValueError):
                        sig_share_val = None
                    if sig_share_val is not None:
                        share_text = f"Population signature share: {sig_share_val:.2f}"
                        pop_diversity = metrics.get('round_population_unique_ratio')
                        if pop_diversity is not None:
                            try:
                                share_text += f" | round diversity {float(pop_diversity):.2f}"
                            except (TypeError, ValueError):
                                pass
                        if metrics.get('round_signature_is_dominant'):
                            share_text += " [dominant]"
                        elif metrics.get('round_signature_is_unique'):
                            share_text += " [unique]"
                        summary.append(share_text)

            strategy_notes = mem.get('strategy_notes') or []
            if strategy_notes:
                summary.append(
                    "Strategy notes: " + "; ".join(str(note) for note in strategy_notes)
                )

            summary.append("Code:")
            summary.append(mem.get('code', ''))
            if 'error' in mem:
                summary.append(f"Error encountered: {mem['error']}")
                
        return "\n".join(summary)
    
    def get_execution_class_attributes(self, member_id):
        """Returns a dictionary of the execution class attributes for inspection."""
        # Get attributes using different methods
        class_attrs = dir(self.__class__)
        instance_attrs = dir(self)
        
        class_dict = self.__class__.__dict__
        instance_dict = self.__dict__
        
        class_vars = vars(self.__class__)
        instance_vars = vars(self)
        
        # Get members using inspect
        import inspect
        all_members = inspect.getmembers(self.__class__)
        function_members = inspect.getmembers(self.__class__, predicate=inspect.isfunction)
        
        return {
            "class_attrs": class_attrs,
            "instance_attrs": instance_attrs,
            "class_dict": class_dict,
            "instance_dict": instance_dict,
            "class_vars": class_vars,
            "instance_vars": instance_vars,
            "all_members": all_members,
            "function_members": function_members
        }

    def prepare_agent_data(self, member_id, error_context_type: str = "mechanism"):
        """Prepares and returns all necessary data for agent prompts."""
        member = self.current_members[member_id]
        # Gather relationship info
        relations = self.parse_relationship_matrix(self.relationship_dict)
        features = self.get_current_member_features()

        # Summaries of past code
        code_memory = self.get_code_memory_summary(member_id)

        # Track relationships for logging
        current_round = len(self.execution_history["rounds"])
        self.execution_history['rounds'][current_round-1]['relationships'] = relations

        # Analysis Memory
        analysis_memory = "No previous analysis"
        # Get analysis from execution history
        analysis_list = []
        for round_data in self.execution_history['rounds'][-3:]:  # Get last 3 rounds
            if 'analysis' in round_data and member_id in round_data['analysis']:
                analysis_list.append(round_data['analysis'][member_id])
        if analysis_list:
            analysis_memory = f"Previous analysis reports: {analysis_list}"

        analysis_card_summary = self.get_analysis_card_summary(member_id)
        
        # Performance Memory
        past_performance = "No previous actions"
        if member_id in self.performance_history and self.performance_history[member_id]:
            perf_list = self.performance_history[member_id]
            avg_perf = sum(perf_list) / len(perf_list)
            recent = perf_list[-3:] if len(perf_list) >= 3 else perf_list
            trend = (recent[-1] - recent[0]) if len(recent) >= 2 else recent[-1]
            volatility = float(np.std(recent)) if len(recent) >= 2 else 0.0
            past_performance = (
                f"Average performance change: {avg_perf:.2f}; "
                f"recent trend: {trend:.2f}; "
                f"recent volatility: {volatility:.2f}; "
                f"last change: {recent[-1]:.2f}"
            )
        if member_id in self.round_performance_history and self.round_performance_history[member_id]:
            round_list = self.round_performance_history[member_id]
            round_avg = sum(round_list) / len(round_list)
            round_recent = round_list[-3:] if len(round_list) >= 3 else round_list
            round_trend = (round_recent[-1] - round_recent[0]) if len(round_recent) >= 2 else round_recent[-1]
            round_volatility = float(np.std(round_recent)) if len(round_recent) >= 2 else 0.0
            past_performance += (
                f" | Round survival delta avg: {round_avg:.2f}; "
                f"recent round trend: {round_trend:.2f}; "
                f"recent round volatility: {round_volatility:.2f}; "
                f"last round delta: {round_recent[-1]:.2f}"
            )

        # Get previous errors for this member, based on prompt type
        error_context = "No previous execution errors"
        if self.execution_history['rounds']:
            errors = self.execution_history['rounds'][-1].get('errors', {})
            error_list = []

            if error_context_type in ("agent_action", "agent_code", "agent"):
                error_list = [
                    e for e in errors.get('agent_code_errors', [])
                    if e.get('member_id') == member_id
                ]
            elif error_context_type in ("analysis", "analyze"):
                analysis_error = errors.get('analyze_code_errors', {}).get(member_id)
                if analysis_error:
                    error_list = [analysis_error]
            else:  # mechanism by default
                error_list = [
                    e for e in errors.get('mechanism_errors', [])
                    if e.get('member_id') == member_id
                ]

            if error_list:
                last_error = error_list[-1]
                error_context = (
                    f"Last execution error (Round {last_error.get('round', 'unknown')}):\n"
                    f"Error type: {last_error.get('error')}\n"
                    f"Code that caused error:\n{last_error.get('code', '')}"
                )

        # Peek messages so multiple prompt phases share the same inbox
        received_messages = self._peek_messages(member_id)
        message_context = "\n".join(received_messages) if received_messages else "No messages received"
        communication_summary = self._summarize_communication(member_id)

        # Get current game mechanisms and modification attempts
        current_round = len(self.execution_history['rounds'])
        start_round = max(0, current_round - 3)  # Get last 3 rounds or all if less
        
        # Get executed modifications from recent rounds
        current_mechanisms = []
        for round_data in self.execution_history['rounds'][start_round:]:
            current_mechanisms.extend(round_data['mechanism_modifications']['executed'])
            
        # Get modification attempts from recent rounds for this member
        modification_attempts = {}
        for round_data in self.execution_history['rounds'][-3:]:
            round_num = round_data['round_number']
            member_attempts = [
                attempt for attempt in round_data['mechanism_modifications']['attempts']
            ]
            modification_attempts[round_num] = member_attempts

        report = None
        if (self.execution_history['rounds'] and 
            'analysis' in self.execution_history['rounds'][-1] and
            member_id in self.execution_history['rounds'][-1]['analysis']):
            report = self.execution_history['rounds'][-1]['analysis'][member_id]

        strategy_profile = self.get_strategy_profile_summary(member_id)
        population_strategy_profile = self.get_population_strategy_summary()
        population_exploration_summary = self.get_population_exploration_summary()
        strategy_recommendations = self.get_strategy_recommendations(member_id)
        contextual_strategy_summary = self.get_contextual_strategy_summary(member_id)
        population_state_summary = self.get_population_state_summary()

        if self.execution_history['rounds']:
            self.execution_history['rounds'][current_round-1][
                'population_state_summary'
            ] = population_state_summary

        return {
            'member': member,
            'relations': relations,
            'features': features,
            'code_memory': code_memory,
            'analysis_memory': analysis_memory,
            'analysis_card_summary': analysis_card_summary,
            'past_performance': past_performance,
            'error_context': error_context,
            'message_context': message_context,
            'communication_summary': communication_summary,
            'current_mechanisms': current_mechanisms,
            'modification_attempts': modification_attempts,
            'report': report,
            'strategy_profile': strategy_profile,
            'population_strategy_profile': population_strategy_profile,
            'population_exploration_summary': population_exploration_summary,
            'strategy_recommendations': strategy_recommendations,
            'contextual_strategy_summary': contextual_strategy_summary,
            'population_state_summary': population_state_summary
        }

    ## Prompting Agents
    
    async def analyze(self, member_id):
        """Analyze the game state and propose strategic actions."""
        result = await _analyze(self, member_id)
        self.save_generated_code(result, member_id, 'analysis')
        return result
    
    async def agent_code_decision(self, member_id) -> None:
        """Modified to save generated code"""
        agent_code_decision_result = await _agent_code_decision(self, member_id)
        if agent_code_decision_result:
            self.save_generated_code(agent_code_decision_result, member_id, 'agent_action')
        return agent_code_decision_result
    
    async def agent_mechanism_proposal(self, member_id) -> None:
        """Modified to save generated code"""
        agent_mechanism_proposal_result = await _agent_mechanism_proposal(self, member_id)
        if agent_mechanism_proposal_result:
            self.save_generated_code(agent_mechanism_proposal_result, member_id, 'mechanism')
        return agent_mechanism_proposal_result

    def _collect_member_snapshot(self):
        """Capture per-member state for round-level evaluation."""
        snapshot = {}
        for member in self.current_members:
            snapshot[member.id] = {
                'vitality': float(member.vitality),
                'cargo': float(member.cargo),
                'land': float(member.land_num),
                'relation_balance': float(self.compute_relation_balance(member)),
                'survival_chance': float(self.compute_survival_chance(member)),
            }
        return snapshot

    def execute_code_actions(self) -> None:
        """Executes all code that the agents wrote (if any) using a restricted namespace."""
        if not hasattr(self, 'agent_code_by_member'):
            self._logger.warning("No agent code to execute.")
            return

        round_num = len(self.execution_history['rounds'])
        round_start_snapshot = self._collect_member_snapshot()
        context_cutoffs = self._compute_population_context_cutoffs(round_start_snapshot)

        for member_id, code_str in self.agent_code_by_member.items():
            if not code_str:
                continue

            print(f"\nExecuting code for Member {member_id}:")
            # print(code_str)

            # Track old stats before executing
            old_survival = self.compute_survival_chance(self.current_members[member_id])
            old_relation_balance = self.compute_relation_balance(self.current_members[member_id])
            old_stats = {
                'vitality': self.current_members[member_id].vitality,
                'cargo': self.current_members[member_id].cargo,
                'land': self.current_members[member_id].land_num,
                'relation_balance': old_relation_balance,
                'survival_chance': old_survival
            }
            context_tags = self._classify_context_tags(old_stats, context_cutoffs)
            context_key = self._context_key_from_tags(context_tags)

            error_occurred = None
            # Track messages for this member
            messages_sent = []
            # Get received messages that were surfaced to the agent (if any)
            received_messages = self._peek_messages(member_id, create_snapshot=False)
            original_send_message = self.send_message

            # Modified exec environment with message tracking
            def tracked_send_message(sender, recipient, msg):
                nonlocal messages_sent
                messages_sent.append((recipient, msg))
                original_send_message(sender, recipient, msg)

            try:
                self.send_message = tracked_send_message
                local_env = {
                    'execution_engine': self,
                    'send_message': tracked_send_message
                }

                # Execute the code in a way that makes the function accessible
                cleaned_code = self.clean_code_string(code_str)
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
                    'type': 'agent_code_execution',
                    'member_id': member_id,
                    'code': code_str,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.execution_history['rounds'][-1]['errors']['agent_code_errors'].append(error_info)
                print(f"Error executing code for member {member_id}:")
                print(traceback.format_exc())
                self._logger.error(f"Error executing code for member {member_id}: {e}")
            finally:
                self.send_message = original_send_message

            # Track changes
            new_survival = self.compute_survival_chance(self.current_members[member_id])
            new_relation_balance = self.compute_relation_balance(self.current_members[member_id])
            new_stats = {
                'vitality': self.current_members[member_id].vitality,
                'cargo': self.current_members[member_id].cargo,
                'land': self.current_members[member_id].land_num,
                'relation_balance': new_relation_balance,
                'survival_chance': new_survival
            }
            
            performance_change = new_survival - old_survival

            # Store in code memory
            if member_id not in self.code_memory:
                self.code_memory[member_id] = []
                
            signature = self._extract_action_signature(code_str)
            signature_novelty = self._compute_signature_novelty(member_id, signature)
            metrics = {
                'delta_vitality': new_stats['vitality'] - old_stats['vitality'],
                'delta_cargo': new_stats['cargo'] - old_stats['cargo'],
                'delta_land': new_stats['land'] - old_stats['land'],
                'delta_relation_balance': new_stats['relation_balance'] - old_stats['relation_balance'],
                'delta_survival': performance_change,
            }
            self._auto_update_strategy_memory(
                self.current_members[member_id],
                round_num,
                signature,
                metrics,
                context_tags,
            )
            message_summary = None
            if received_messages or messages_sent:
                message_summary = {
                    'received_count': len(received_messages),
                    'sent_count': len(messages_sent),
                    'received_sample': [
                        self._truncate_message(msg) for msg in received_messages[-2:]
                    ],
                    'sent_sample': [
                        (recipient, self._truncate_message(msg))
                        for recipient, msg in messages_sent[-2:]
                    ],
                }
            strategy_notes = self._collect_strategy_notes(
                self.current_members[member_id]
            )
            memory_entry = {
                'code': code_str,
                'performance': performance_change,
                'signature': signature,
                'signature_novelty': signature_novelty,
                'context_tags': context_tags,
                'context_key': context_key,
                'metrics': metrics,
                'context': {
                    'old_stats': old_stats,
                    'new_stats': new_stats,
                    'message_summary': message_summary,
                    'round': round_num
                }
            }
            if strategy_notes:
                memory_entry['strategy_notes'] = strategy_notes
            if error_occurred:
                memory_entry['error'] = error_occurred
                
            self.code_memory[member_id].append(memory_entry)

            # Store performance in history
            if member_id not in self.performance_history:
                self.performance_history[member_id] = []
            self.performance_history[member_id].append(performance_change)

            # Log changes for this round
            self.execution_history['rounds'][-1]['agent_actions'].append({
                'member_id': member_id,
                'code_executed': code_str,
                'old_stats': old_stats,
                'new_stats': new_stats,
                'performance_change': performance_change,
                'messages_sent': messages_sent
            })

            # Log messages in round data
            self.execution_history['rounds'][-1]['agent_messages'][member_id] = {
                'received': received_messages,
                'sent': messages_sent
            }

        round_end_snapshot = self._collect_member_snapshot()
        round_deltas = {}
        for member_id, start_stats in round_start_snapshot.items():
            end_stats = round_end_snapshot.get(member_id)
            if not end_stats:
                continue
            round_deltas[member_id] = {
                'vitality': end_stats['vitality'] - start_stats['vitality'],
                'cargo': end_stats['cargo'] - start_stats['cargo'],
                'land': end_stats['land'] - start_stats['land'],
                'relation_balance': end_stats['relation_balance'] - start_stats['relation_balance'],
                'survival_chance': end_stats['survival_chance'] - start_stats['survival_chance'],
            }

        survival_deltas = [delta['survival_chance'] for delta in round_deltas.values()]
        relation_deltas = [delta['relation_balance'] for delta in round_deltas.values()]
        vitality_deltas = [delta['vitality'] for delta in round_deltas.values()]
        cargo_deltas = [delta['cargo'] for delta in round_deltas.values()]
        land_deltas = [delta['land'] for delta in round_deltas.values()]

        pop_avg_survival = float(np.mean(survival_deltas)) if survival_deltas else 0.0
        pop_std_survival = float(np.std(survival_deltas)) if len(survival_deltas) > 1 else 0.0
        pop_avg_relation = float(np.mean(relation_deltas)) if relation_deltas else 0.0
        pop_avg_vitality = float(np.mean(vitality_deltas)) if vitality_deltas else 0.0
        pop_avg_cargo = float(np.mean(cargo_deltas)) if cargo_deltas else 0.0
        pop_avg_land = float(np.mean(land_deltas)) if land_deltas else 0.0

        signature_stats = self._collect_round_signature_stats(round_num)

        self.execution_history['rounds'][-1]['round_metrics'] = {
            'population_avg_survival_delta': pop_avg_survival,
            'population_std_survival_delta': pop_std_survival,
            'population_avg_relation_delta': pop_avg_relation,
            'population_avg_vitality_delta': pop_avg_vitality,
            'population_avg_cargo_delta': pop_avg_cargo,
            'population_avg_land_delta': pop_avg_land,
            'population_signature_total': signature_stats.get('total', 0),
            'population_signature_unique_ratio': signature_stats.get('diversity_ratio', 0.0),
            'population_signature_entropy': signature_stats.get('entropy', 0.0),
            'population_signature_dominant_share': signature_stats.get('dominant_share', 0.0),
            'member_count': len(round_deltas),
        }

        action_totals = {
            'attack': self._get_record_total('attack'),
            'benefit': self._get_record_total('benefit'),
            'benefit_land': self._get_record_total('benefit_land'),
        }
        action_edges = {
            key: len(self.record_action_dict.get(key, {}))
            for key in ('attack', 'benefit', 'benefit_land')
        }
        self.execution_history['rounds'][-1]['action_totals'] = action_totals
        self.execution_history['rounds'][-1]['action_edges'] = action_edges

        for member_id, delta in round_deltas.items():
            self.round_performance_history.setdefault(member_id, []).append(delta['survival_chance'])
            mem_list = self.code_memory.get(member_id, [])
            if not mem_list:
                continue
            for mem in reversed(mem_list):
                if mem.get('context', {}).get('round') == round_num:
                    metrics = mem.setdefault('metrics', {})
                    metrics.update({
                        'round_delta_survival': delta['survival_chance'],
                        'round_relative_survival': delta['survival_chance'] - pop_avg_survival,
                        'round_delta_relation_balance': delta['relation_balance'],
                        'round_delta_vitality': delta['vitality'],
                        'round_delta_cargo': delta['cargo'],
                        'round_delta_land': delta['land'],
                        'round_population_avg_survival': pop_avg_survival,
                        'round_population_std_survival': pop_std_survival,
                    })
                    sig_total = signature_stats.get('total', 0)
                    sig_map = signature_stats.get('signatures', {})
                    sig_counts = signature_stats.get('counts', Counter())
                    sig = sig_map.get(member_id)
                    if sig is None:
                        sig = mem.get('signature')
                        if sig is None:
                            sig = self._extract_action_signature(mem.get('code', ''))
                        sig = tuple(sig) if sig else tuple()
                    if sig_total:
                        sig_count = sig_counts.get(sig, 0)
                        sig_share = sig_count / sig_total if sig_total else 0.0
                        metrics.update({
                            'round_signature_count': sig_count,
                            'round_signature_share': sig_share,
                            'round_signature_is_unique': sig_count == 1,
                            'round_signature_is_dominant': sig_share >= 0.5 if sig_total > 1 else True,
                            'round_population_unique_ratio': signature_stats.get('diversity_ratio', 0.0),
                            'round_population_signature_entropy': signature_stats.get('entropy', 0.0),
                            'round_population_signature_total': sig_total,
                            'round_population_signature_dominant_share': signature_stats.get('dominant_share', 0.0),
                        })
                    break

        self.execution_history['rounds'][-1]['population_strategy_profile'] = (
            self.get_population_strategy_summary()
        )

        # Save execution history after each round
        self.save_execution_history()

        # Remove messages that were already surfaced in prompts this round
        self._consume_message_snapshots(round_num)
        
        # Clear code after execution
        self.agent_code_by_member = {}

        # Add this line to preserve messages until next decision phase
        self.messages = {k:v for k,v in self.messages.items() if v}  # Only keep non-empty lists

    def save_execution_history(self):
        """Save the execution history to a JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.execution_history_path, f'execution_history_{timestamp}.json')
        
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
            if member_id not in self.current_members:
                continue  # Skip dead members
                
            member = self.current_members[member_id]
            current_survival = self.compute_survival_chance(member)
            avg_perf = sum(performance_list) / len(performance_list) if performance_list else 0
            
            # Get relationship summary
            relations = self.parse_relationship_matrix(self.relationship_dict)
            relation_summary = [r for r in relations if f"member_{member_id}" in r]
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
        # Get member's index in current_members list
        member_index = next((i for i, m in enumerate(self.current_members) if m.id == member.id), -1)
        if member_index == -1:
            return 0.0  # Member not found
        
        #  survival from own attributes
        base_survival = member.vitality + member.cargo
        
        # Get relationship bonuses/penalties
        relationship_modifier = 0
        for relation_type, matrix in self.relationship_dict.items():
            if member_index >= matrix.shape[0]:  # This checks ID against matrix size
                continue
            if relation_type == 'benefit':
                # Use nansum to handle potential NaN values
                relationship_modifier += np.nansum(matrix[member_index, :]) * 0.2
            elif relation_type == 'victim':
                # Use nansum to handle potential NaN values
                relationship_modifier -= np.nansum(matrix[member_index, :]) * 0.3
        
        # Neighborhood safety (more neighbors = more risk/opportunity)
        # Add type conversion to ensure numerical operation
        neighbor_count = float(len(member.current_clear_list))
        neighborhood_modifier = -0.1 * neighbor_count
        
        # Ensure final value is finite
        survival_score = base_survival + relationship_modifier + neighborhood_modifier
        return np.nan_to_num(survival_score, nan=50.0)  # Default to 50 if calculation fails

    def compute_relation_balance(self, member):
        """
        Compute a simple relationship balance score:
        benefits and land received minus victimization.
        """
        member_index = next((i for i, m in enumerate(self.current_members) if m.id == member.id), -1)
        if member_index == -1:
            return 0.0

        benefit = 0.0
        victim = 0.0
        benefit_land = 0.0

        if 'benefit' in self.relationship_dict:
            benefit = float(np.nansum(self.relationship_dict['benefit'][member_index, :]))
        if 'victim' in self.relationship_dict:
            victim = float(np.nansum(self.relationship_dict['victim'][member_index, :]))
        if 'benefit_land' in self.relationship_dict:
            benefit_land = float(np.nansum(self.relationship_dict['benefit_land'][member_index, :]))

        return benefit + 0.5 * benefit_land - victim

    def send_message(self, sender_id: int, recipient_id: int, message: str):
        """Allow agents to send messages to each other"""
        print(f"[MSG] Member {sender_id} -> Member {recipient_id}: {message!r}")
        # Add validation check
        if any(m.id == recipient_id for m in self.current_members):
            if recipient_id not in self.messages:
                self.messages[recipient_id] = []
            self.messages[recipient_id].append(f"From member_{sender_id}: {message}")
        else:
            print(f"Invalid message recipient {recipient_id} from member {sender_id}")

    def print_agent_messages(self):
        """Print message communication between agents"""
        print("\n=== Agent Message History ===")
        for round_idx, round_data in enumerate(self.execution_history['rounds']):
            print(f"\nRound {round_idx + 1} ({round_data['timestamp']}):")
            if not round_data['agent_messages']:
                print("  No messages exchanged")
                continue
            
            for member_id, comm in round_data['agent_messages'].items():
                print(f"Member {member_id}:")
                if comm['received']:
                    print("  Received:")
                    for msg in comm['received']:
                        print(f"    - {msg}")
                if comm['sent']:
                    print("  Sent:")
                    for recipient, msg in comm['sent']:
                        print(f"    -> Member {recipient}: {msg}")

    # def process_voting_mechanism(self):
    #     """Handle voting process for modification proposals"""
    #     current_round = len(self.execution_history['rounds'])
        
    #     # Check each member's votes in their voting box
    #     for member_id, vote_data in self.voting_box.items():
    #         if not vote_data['yes_votes']:
    #             continue
                
    #         # Add votes to the corresponding modifications
    #         for mod_index in vote_data['yes_votes']:
    #             if mod_index >= len(self.execution_history['rounds'][-1]['modification_attempts']):
    #                 continue
                    
    #             mod = self.execution_history['rounds'][-1]['modification_attempts'][mod_index]
                
    #             if mod['ratified'] or 'ratification_condition' not in mod:
    #                 continue

    #             # Check if this modification requires voting
    #             vote_conditions = [c for c in mod['ratification_condition'].get('conditions', []) 
    #                               if c['type'] == 'vote']
                
    #             # Initialize votes dict if needed
    #             if 'votes' not in mod:
    #                 mod['votes'] = {'yes': 0}
                
    #             # Add this member's vote
    #             mod['votes']['yes'] += 1
                
    #             # Check if threshold met
    #             for condition in vote_conditions:
    #                 total_voters = len(self.current_members)
    #                 yes_votes = mod['votes']['yes']
                    
    #                 if yes_votes / total_voters >= condition['threshold']:
    #                     mod['ratified'] = True
    #                     mod['ratified_round'] = current_round
    #                     print(f"Modification from member {mod['member_id']} ratified by vote")
                        
    #             # Add who voted for what
    #             if 'voter_records' not in mod:
    #                 mod['voter_records'] = []
    #             mod['voter_records'].append({
    #                 'member_id': member_id,
    #                 'vote': 'yes',
    #                 'timestamp': datetime.now().isoformat()
    #             })
        
    #     self.voting_box = {}
                    
    def execute_mechanism_modifications(self):
        """Execute ratified modifications"""
        current_round = len(self.execution_history['rounds'])
           
        # Process voting first
        # self.process_voting_mechanism()

        # Check remaining ratification conditions
        # for mod in self.execution_history['rounds'][-1]['mechanism_modifications']['attempts']:
        #     if not mod.get('ratified'):
        #         conditions = mod.get('ratification_condition', {}).get('conditions', [])
        #         all_conditions_met = True
                
        #         for condition in conditions:
        #             if condition['type'] != 'vote':  # Skip vote conditions handled earlier
        #                 # Add condition checking logic here
        #                 all_conditions_met = False
        #                 break
                        
        #         if all_conditions_met:
        #             mod['ratified'] = True
        #             mod['ratified_round'] = current_round
        #             self.execution_history['rounds'][-1]['mechanism_modifications']['executed'].append(mod)

        # Execute ratified modifications
        exec_env = {'execution_engine': self}
        
        # for mod in self.execution_history['rounds'][-1]['mechanism_modifications']['attempts']:
        #     # if not mod.get('ratified'):
        #     #     continue
        
        mods = self.execution_history['rounds'][-1]['mechanism_modifications']['attempts']
        
        base_code = self.base_class_code
        
        # Get aggregated mechanism modification
        base_code_prompt = f"""
            [Base Code]
            Here is the base code for the Island and Member classes that you should reference when making modifications. Study the mechanisms carefully to ensure your code interacts correctly with the available attributes and methods. Pay special attention to:
            - Valid attribute access patterns
            - Method parameters and return values 
            - Constraints and preconditions for actions
            - Data structure formats and valid operations
            {base_code}
            """
        try:
            prompt = f"""
            [Base Mechanism Code]
            {base_code_prompt}
            
            You are an AI tasked with aggregating multiple mechanism modification proposals into a single coherent modification.
            
            Here are the individual proposals to aggregate:
            {[mod['code'] for mod in mods]}
            
            Please analyze these proposals and generate a single aggregated mechanism modification that:
            1. Preserves the key functionality from each proposal
            2. Resolves any conflicts between proposals
            3. Maintains consistency with the game mechanics
            4. Results in a single propose_modification() function
            
            def propose_modification(self):
            \"""
            Include clear reasoning for each modification to help other agents
            understand tee intended benefits and evaluate the proposal.
            \"""
        
            Return only the aggregated code.
            """
            
            completion = client.chat.completions.create(
                model=f'{provider}:{model_id}',
                messages=[{"role": "user", "content": prompt}]
            )
            
            aggregated_code = completion.choices[0].message.content.strip()
            code_result = self.clean_code_string(aggregated_code)
            
            self.save_generated_code(code_result, '-1', 'aggregated_mechanism')
            # print(f"Aggregated Mechanism Code: {code_result}")
            
            mod = {
                'member_id': 'aggregated',
                'code': code_result,
                'round': len(self.execution_history['rounds'])
            }
            
        except Exception as e:
            print(f"Error getting aggregated modification from O1: {e}")
            # Fall back to first modification if aggregation fails
            mod = mods[0] if mods else None
                
        try:
            print(f"\nExecuting modification code for Member {mod['member_id']}:")
            # print(mod['code'])

            # Execute modification code
            exec(mod['code'], exec_env)
            exec_env['propose_modification'](self)
            print(f"Aggregated Mechanism Modification code executed successfully.")
            
            # Get execution class attributes
            class_attrs = self.get_execution_class_attributes(mod['member_id'])
            # Define a function to save mechanism execution data to JSON
            def save_mechanism_execution_to_json(mechanism_data, json_path):
                with open(json_path, 'w') as f:
                    json.dump(mechanism_data, f, indent=4)
            
            # Prepare the mechanism execution data
            mechanism_data = {
                'member_id': mod['member_id'],
                'round': current_round,
                'code': mod['code'],
                'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'class_attributes': class_attrs
            }
            
            # Create a filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_filename = f'mechanism_execution_{mod["member_id"]}_round_{current_round}_{timestamp}.json'
            json_path = os.path.join(self.mechanism_code_path, json_filename)
            
            # Save the data to JSON file
            save_mechanism_execution_to_json(mechanism_data, json_path)
            
            # Track successful modification
            mod['executed_round'] = current_round
            self.execution_history['rounds'][-1]['mechanism_modifications']['executed'].append(mod)
            
        except Exception as e:
            error_info = {
                'round': current_round,
                'type': 'execute_mechanism_modifications', 
                'error': str(e),
                'traceback': traceback.format_exc(),
                'code': mod['code'],
                'member_id': mod['member_id']
            }
            self.execution_history['rounds'][-1]['errors']['mechanism_errors'].append(error_info)
            print(f"Error executing code for member {mod['member_id']}:")
            print(traceback.format_exc())
            self._logger.error(f"Error executing code for member {mod['member_id']}: {e}")

    def save_generated_code(self, code: str, member_id: int, code_type: str) -> None:
        """
        Save generated code to appropriate directory
        
        Args:
            code (str): The code to save
            member_id (int): ID of the member who generated the code
            code_type (str): Type of code ('agent_action' or 'mechanism')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        round_num = len(self.execution_history['rounds'])
        
        if code_type == 'agent_action':
            save_dir = self.agent_code_path
            filename = f'agent_{member_id}_round_{round_num}_{timestamp}.py'
        elif code_type == 'analysis':
            save_dir = self.analysis_code_path
            filename = f'analysis_{member_id}_round_{round_num}_{timestamp}.txt'
        elif code_type == 'aggregated_mechanism':
            save_dir = self.mechanism_code_path
            filename = f'aggregated_mechanism_{member_id}_round_{round_num}_{timestamp}.py'
        else:  # mechanism
            save_dir = self.mechanism_code_path
            filename = f'mechanism_{member_id}_round_{round_num}_{timestamp}.py'
            
        file_path = os.path.join(save_dir, filename)
        
        try:
            with open(file_path, 'w') as f:
                f.write(f"# Generated code for Member {member_id} in Round {round_num}\n")
                f.write(f"# Generated at: {timestamp}\n\n")
                if code_type != 'analysis':
                    f.write(self.clean_code_string(code))
                else:
                    f.write(code)
            print(f"Saved generated code to {file_path}")
        except Exception as e:
            print(f"Error saving generated code: {e}")
            print(traceback.format_exc())

    def _setup_default_graph(self):
        """Setup the default execution graph with all nodes"""
        # Create nodes
        nodes = {
            'new_round': NewRoundNode(),
            'analyze': AnalyzeNode(),
            'propose_mech': ProposeMechanismNode(),
            'judge_mech': JudgeNode(),
            'execute_mech': ExecuteMechanismsNode(),
            'agent_decide': AgentDecisionNode(),
            'execute_actions': ExecuteActionsNode(),
            'contracts': ContractNode(),
            'produce': ProduceNode(),
            'consume': ConsumeNode(),
            'environment': EnvironmentNode(),
            'log_status': LogStatusNode()
        }

        # Add all nodes to graph
        for node in nodes.values():
            self.graph.add_node(node)

        # Connect nodes to define execution flow
        connections = [
            ('new_round', 'analyze'),
            ('analyze', 'propose_mechanisms'),
            ('propose_mechanisms', 'judge'),
            ('judge', 'execute_mechanisms'),
            ('execute_mechanisms', 'agent_decisions'),
            ('agent_decisions', 'execute_actions'),
            ('execute_actions', 'contracts'),
            ('contracts', 'produce'),
            ('produce', 'consume'),
            ('consume', 'environment'),
            ('environment', 'log_status')
        ]

        for from_name, to_name in connections:
            self.graph.connect(from_name, to_name)

        print(f"\n[Graph] Initialized with {len(nodes)} nodes")
        print(self.graph.visualize())

    async def run_round_with_graph(self):
        """Run one complete round using the graph execution engine"""
        self.round_number += 1
        self.graph.context = {
            'execution': self,
            'round': self.round_number
        }

        print(f"\n{'='*60}")
        print(f"=== Round {self.round_number} (Graph Execution) ===")
        print(f"{'='*60}")

        await self.graph.execute_round()

        print(f"\n[Graph] Round {self.round_number} complete")

    def enable_graph_node(self, node_name: str):
        """Enable a graph node"""
        self.graph.enable_node(node_name)
        print(f"[Graph] Enabled node: {node_name}")

    def disable_graph_node(self, node_name: str):
        """Disable a graph node"""
        self.graph.disable_node(node_name)
        print(f"[Graph] Disabled node: {node_name}")

async def main(use_graph=True):
    """
    Main execution function

    Args:
        use_graph: If True, use new graph-based execution. If False, use old execution.
    """
    from time import time
    from utils import save
    import os

    rng = np.random.default_rng()
    os.makedirs("../MetaIsland/data", exist_ok=True)
    path = save.datetime_dir("../MetaIsland/data")
    exec = IslandExecution(5, (10, 10), path, 2023)
    IslandExecution._RECORD_PERIOD = 1

    round_num = 5  # Start with fewer rounds for testing

    exec.island_ideology = """
    [Island Ideology]

    Island is a place of abundant resources and opportunity. Agents can:
    - Extract resources from land based on land quality
    - Build businesses to transform resources into products
    - Create contracts with other agents for trade and services
    - Propose physical constraints and economic mechanisms
    - Form supply chains through interconnected contracts

    The economy is entirely agent-driven. Agents write code to:
    - Define what resources exist and how they're extracted
    - Create markets and pricing mechanisms
    - Build production chains and businesses
    - Establish labor markets and specialization

    Success depends on creating realistic, mutually beneficial economic systems.
    """

    if use_graph:
        print("\n" + "="*60)
        print("USING NEW GRAPH-BASED EXECUTION")
        print("="*60)

        # Optionally disable some nodes for testing
        # exec.disable_graph_node('judge_mech')  # Uncomment to skip judging

        for round_i in range(round_num):
            await exec.run_round_with_graph()

            # Print final statistics
            print(f"\n=== Round {round_i + 1} Statistics ===")
            print(f"Surviving members: {len(exec.current_members)}")
            print(f"Contracts: {exec.contracts.get_statistics()}")
            print(f"Physics: {exec.physics.get_statistics()}")
            print(f"Judge: {exec.judge.get_statistics()}")
    else:
        print("\n" + "="*60)
        print("USING OLD EXECUTION (LEGACY MODE)")
        print("="*60)

        for round_i in range(round_num):
            # Create log file for this round
            header = f"\n{'='*50}\n=== Round {round_i + 1} ===\n{'='*50}"
            print(header)

            exec.new_round()
            exec.get_neighbors()
            exec.produce()

            print("\nGenerating mechanism modifications...")

            # Create all tasks upfront
            member_tasks = []
            for i in range(len(exec.current_members)):
                print(f"Member {i} starts analyzing...")
                analysis_task = exec.analyze(i)
                print(f"Member {i} starts proposing mechanism...")
                mechanism_task = exec.agent_mechanism_proposal(i)
                member_tasks.extend([analysis_task, mechanism_task])

            # Run all tasks concurrently
            await asyncio.gather(*member_tasks)

            print("\nExecuting mechanism modifications...")
            exec.execute_mechanism_modifications()

            print("\nGenerating agent decisions...")

            # Convert the second loop to async
            decision_tasks = []
            for i in range(len(exec.current_members)):
                print(f"Member {i} is deciding...")
                decision_tasks.append(exec.agent_code_decision(i))
            await asyncio.gather(*decision_tasks)

            print("\nExecuting agent actions...")
            exec.execute_code_actions()
            exec.consume()

            print("\nPerformance Report:")

            # Capture performance output
            exec.print_agent_performance()
            exec.print_agent_messages()

            # Print status
            exec.log_status(action=True, log_instead_of_print=True)

            print(f"\nSurviving members at end of round: {len(exec.current_members)}")

            for member in exec.current_members:
                survival_chance = exec.compute_survival_chance(member)
                status = f"Member {member.id}: Vitality={member.vitality:.2f}, Cargo={member.cargo:.2f}, Survival={survival_chance:.2f}"
                print(status)

if __name__ == "__main__":
    asyncio.run(main())
