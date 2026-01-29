from Leviathan.Island import Island
import Leviathan.api_key
from Leviathan.Member import Member
from Leviathan.Land import Land
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import openai
import json
from datetime import datetime
import traceback
import os
from collections import defaultdict, Counter

class IslandExecution(Island):
    def __init__(self, 
        init_member_number: int,
        land_shape: Tuple[int, int],
        save_path: str,
        random_seed: Optional[int] = None,
        action_board: List[List[Tuple[str, int, int]]] = None,
        agent_modifications: dict = None
    ):
        # Add agent modification tracking
        self.agent_modifications = {
            'pre_init': [],
            'post_init': [],
            'member_mods': [],
            'land_mods': [],
            'relationship_mods': []
        }
        
        # Apply agent modifications before super().__init__
        if agent_modifications:
            self._apply_agent_modifications(agent_modifications)
            
        super().__init__(
            init_member_number,
            land_shape,
            save_path,
            random_seed
        )
        
        # Remove example action board and related attributes since we're not using them
        self.performance_history = {}  # {member_id: [list_of_performance_metrics]}
        
        # Add code memory tracking
        self.code_memory = {}  # {member_id: [{'code': str, 'performance': float, 'context': dict}]}
        
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
        
        # New: Allow agents to revise the prompt passed to them in future rounds.
        self.agent_prompt_revisions = {}

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
        
    def offer_land(self, member_1, member_2, parameter_influence):
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
        
        for idx, member in enumerate(self.current_members):
            # Get self attributes
            feature_row = {
                "member_index": getattr(member, "surviver_id", idx),
                "self_productivity": member.overall_productivity,
                "self_vitality": member.vitality, 
                "self_cargo": member.cargo,
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
                summary[action_name] = {
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

    def _format_signature(self, signature: tuple) -> str:
        """Format a signature tuple for readable summaries."""
        if not signature:
            return "none"
        return ", ".join(signature)

    def _get_memory_signatures(self, memory: list) -> list:
        """Normalize action signatures from memory entries."""
        signatures = []
        for mem in memory:
            sig = mem.get('signature')
            if sig is None:
                sig = self._extract_action_signature(mem.get('code', ''))
            signatures.append(tuple(sig) if sig else tuple())
        return signatures

    def _compute_signature_novelty(self, member_id: int, signature: tuple) -> float:
        """Compute a simple novelty score based on prior signature frequency."""
        if not signature:
            return 0.0
        memory = self.code_memory.get(member_id, [])
        if not memory:
            return 1.0
        count = 0
        for mem in memory:
            sig = mem.get('signature')
            if sig is None:
                sig = self._extract_action_signature(mem.get('code', ''))
            if tuple(sig) == signature:
                count += 1
        return 1.0 / (1.0 + count)

    def _signature_streak(self, member_id: int, window: int = 4) -> Tuple[int, tuple]:
        """Return (streak_length, last_signature) for recent code history."""
        memory = self.code_memory.get(member_id, [])
        if not memory:
            return 0, tuple()
        signatures = self._get_memory_signatures(memory)
        recent = signatures[-max(1, window):]
        last_sig = recent[-1] if recent else tuple()
        streak = 1 if last_sig else 0
        for sig in reversed(recent[:-1]):
            if sig == last_sig:
                streak += 1
            else:
                break
        return streak, last_sig

    def _compute_reward_score(self, delta: Optional[dict]) -> float:
        """Compute a bounded reward score from vitality/cargo/land deltas."""
        if not delta:
            return 0.0
        vitality = float(delta.get('vitality', 0.0))
        cargo = float(delta.get('cargo', 0.0))
        land = float(delta.get('land', 0.0))
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

    def _summarize_learning_trend(self, member_id: int, window: int = 5) -> str:
        """Summarize recent performance trend and novelty for adaptive learning."""
        perf_history = self.performance_history.get(member_id, [])
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

        memory = self.code_memory.get(member_id, [])
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

        streak, last_sig = self._signature_streak(member_id)
        if trend == "flat" and streak >= 3:
            sig_text = self._format_signature(last_sig)
            lines.append(
                f"- stagnation signal: {streak}x repeat of ({sig_text}); "
                "consider a safe variation if situation allows."
            )
        return "\n".join(lines)

    def _summarize_action_impact(
        self,
        window_rounds: int = 4,
        min_samples: int = 2,
    ) -> str:
        """Summarize recent action-tag correlations with outcomes."""
        if not self.code_memory or not self.execution_history.get('rounds'):
            return "Action impact snapshot: no data yet."

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
                signature = mem.get("signature")
                if signature is None:
                    signature = self._extract_action_signature(mem.get("code", ""))
                signature = tuple(signature) if signature else tuple()
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

    def _summarize_population_strategy_diversity(
        self,
        window_rounds: int = 3,
        top_k: int = 3,
    ) -> str:
        """Summarize population-level action signature diversity."""
        if not self.code_memory or not self.execution_history.get('rounds'):
            return "No population strategy data yet."

        current_round = len(self.execution_history['rounds'])
        window_rounds = max(1, window_rounds)
        min_round = max(0, current_round - window_rounds)

        entries = []
        for memory in self.code_memory.values():
            for mem in memory:
                round_num = mem.get('context', {}).get('round')
                if round_num is None or round_num < min_round:
                    continue
                sig = mem.get('signature')
                if sig is None:
                    sig = self._extract_action_signature(mem.get('code', ''))
                sig = tuple(sig) if sig else tuple()
                entries.append(sig)

        if not entries:
            return "No recent population strategy data."

        total_actions = len(entries)
        signature_counts = Counter(entries)
        unique_signatures = len(signature_counts)
        diversity_ratio = unique_signatures / total_actions if total_actions else 0.0

        dominant_sig, dominant_count = signature_counts.most_common(1)[0]
        dominant_share = dominant_count / total_actions if total_actions else 0.0

        top_signatures = signature_counts.most_common(min(top_k, len(signature_counts)))
        top_text = "; ".join(
            f"{self._format_signature(sig)} (n={count})"
            for sig, count in top_signatures
        )

        lines = [
            "Population strategy diversity snapshot:",
            f"- Window rounds: {min_round}-{current_round} (actions {total_actions})",
            f"- Unique signatures: {unique_signatures} (ratio {diversity_ratio:.2f})",
            f"- Dominant signature share: {dominant_share:.2f} "
            f"({self._format_signature(dominant_sig)})",
            f"- Top signatures: {top_text if top_text else 'none'}",
        ]

        if dominant_share >= 0.6 and unique_signatures > 1:
            lines.append(
                "- Convergence risk: majority of actions share one signature; "
                "consider exploring underused tags."
            )

        return "\n".join(lines)

    def _select_memory_samples(self, memory, max_samples: int):
        """Select a diversity-aware sample of memory entries."""
        if not memory:
            return []

        if len(memory) <= max_samples:
            return [("Recent", mem) for mem in memory]

        indices = list(range(len(memory)))
        by_perf = sorted(indices, key=lambda idx: memory[idx].get('performance', 0.0))

        best_idx = by_perf[-1]
        worst_idx = by_perf[0]
        recent_idx = indices[-1]
        median_idx = by_perf[len(by_perf) // 2]
        abs_idx = max(indices, key=lambda idx: abs(memory[idx].get('performance', 0.0)))

        candidates = [
            ("Recent", recent_idx),
            ("Best", best_idx),
            ("Worst", worst_idx),
            ("Median", median_idx),
            ("Volatile", abs_idx),
        ]

        selected = []
        seen = set()
        for label, idx in candidates:
            if idx not in seen:
                selected.append((label, idx))
                seen.add(idx)
                if len(selected) >= max_samples:
                    break

        if len(selected) < max_samples:
            remaining = [idx for idx in indices if idx not in seen]
            remaining_sorted = sorted(
                remaining,
                key=lambda idx: memory[idx].get('context', {}).get('round', idx),
                reverse=True,
            )

            selected_signatures = {
                self._extract_action_signature(memory[idx].get('code', ''))
                for idx in seen
            }

            for idx in remaining_sorted:
                signature = self._extract_action_signature(memory[idx].get('code', ''))
                if signature not in selected_signatures:
                    selected.append(("Diverse", idx))
                    seen.add(idx)
                    selected_signatures.add(signature)
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

        selected = self._select_memory_samples(memory, max_samples=5)

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

            round_context = context.get("round_context")
            if round_context:
                summary.append(f"Round context: {round_context}")

            if not any([old_stats is not None, new_stats is not None, delta, action_summary, round_context]) and context:
                summary.append(f"Context: {context}")

            signature = mem.get('signature')
            if signature is None:
                signature = self._extract_action_signature(mem.get('code', ''))
            if signature:
                summary.append(f"Action signature: {', '.join(signature)}")
            if mem.get('signature_novelty') is not None:
                summary.append(f"Signature novelty: {mem['signature_novelty']:.2f}")

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
        if not hasattr(self, "_decision_messages"):
            self._decision_messages = {}
        # Gather relationship info
        relations = self.parse_relationship_matrix(self.relationship_dict)
        features = self.get_current_member_features()

        # Summaries of past code
        code_memory = self.get_code_memory_summary(member_id)

        # Track relationships for logging
        self.execution_history['relationships'][
            f'round_{len(self.execution_history["rounds"])}_{member_id}'
        ] = relations

        past_performance = "No previous actions"
        if member_id in self.performance_history and self.performance_history[member_id]:
            perf_list = self.performance_history[member_id]
            avg_perf = sum(perf_list) / len(perf_list)
            recent_perf = perf_list[-3:]
            past_performance = (
                f"Average performance change: {avg_perf:.2f}. "
                f"Recent changes (last {len(recent_perf)}): {recent_perf}"
            )

        round_context = self._compute_round_context()

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
        message_context = "\n".join(received_messages) if received_messages else "No messages received"
        
        # New: Incorporate agent-specific prompt revision if available.
        revision_text = self.agent_prompt_revisions.get(member_id, "").strip()
        revision_section = f"\n[Agent Provided Prompt Revision]\n{revision_text}\n" if revision_text else ""
        features_text = features.to_string(index=False)
        relations_text = "\n".join(relations) if isinstance(relations, list) else str(relations)
        learning_trend = self._summarize_learning_trend(member_id)
        action_impact = self._summarize_action_impact()
        population_action_mix = self._summarize_population_action_mix()
        population_strategy_diversity = self._summarize_population_strategy_diversity()

        try:
            # Define iterative prompt parts with specific constraints
            part0 = f"""
            [Previous code execution context]
        {error_context}

        [Current task]
        You are member_{member.id} in a society that you can help shape.
        Write a Python function named agent_action(execution_engine, member_id) 
        that implements your vision of social organization while ensuring your survival.

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
            • execution_engine.offer(member1, member2, True) - Offers resources
            • execution_engine.offer_land(member1, member2, True) - Offers land
            • execution_engine.bear(member1, member2) - Bears offspring
            • execution_engine.expand(member1) - Expands territory
            • execution_engine.send_message(sender_index, recipient_index, "message")
            • execution_engine.send_message_by_id(sender_id, recipient_id, "message")
            • execution_engine.get_member_by_id(member_id) -> Member or None
            • execution_engine.resolve_member_index_by_id(member_id) -> index or None
        5) The members are accessed by execution_engine.current_members[index].
            Example: execution_engine.current_members[2] returns the member with surviver_id=2.
        6) DO NOT reference 'member.member_id' or 'member.self_vitality'. Use member.id, member.vitality, etc.

        Current status (features of all members):
        {features_text}

        Relationship summary (parsed from relationship_dict):
        {relations_text}

        Code Memory and Previous Performance:
        {code_memory}

        Performance history:
        {past_performance}

        Round context (global signals):
        {round_context}

        Strategy learning summary:
        {learning_trend}

        Population action mix (last round):
        {population_action_mix}

        Population strategy diversity snapshot:
        {population_strategy_diversity}

        Action impact snapshot:
        {action_impact}
        {revision_section}

        Based on the previous code performance, adapt and improve the strategy.
        If a previous strategy worked well (high performance), consider building upon it.
        If it failed, try a different approach.
        Balance short-term survival with longer-term vitality/cargo/land outcomes rather than optimizing a single metric.
        Preserve agent-level diversity: avoid repeating identical action signatures if you are on a streak; if a single action dominates the population mix, consider alternative actions that still fit your situation.
        
        IMPORTANT: Do not simply copy the example implementation below. Instead, use it as inspiration to create your own unique approach combining different methods and strategies in novel ways.
        
        Return only the code, no extra text or explanation. While the example above shows one possible approach,
        you should create your own unique implementation drawing from the wide range of available methods and strategies.
        
        Consider novel combinations of different approaches rather than following this exact pattern.
        """
            part1 = f"""
            {part0}
            [Social Strategies]
        Consider these social strategies:
        - Design systems for resource distribution and allocation
        - Build alliances and cooperative networks 
        - Create mechanisms for collective decision making
        - Establish norms and rules for interaction
        - Develop methods for conflict resolution

        [Communication Strategy]
        You can communicate with multiple members in a single round using:
        execution_engine.send_message(your_index, recipient_index, "message")
        execution_engine.send_message_by_id(your_id, recipient_id, "message")
        Example usage:
        - Broadcast to all: 
          for recipient in range(len(execution_engine.current_members)):
              if recipient != your_index:
                  execution_engine.send_message(your_index, recipient, "Let's cooperate!")
        - Message allies:
          for ally_id in ally_list:
              execution_engine.send_message_by_id(your_id, ally_id, "Attack target X")
        - Group coordination:
          for member_id in coalition:
              execution_engine.send_message_by_id(your_id, member_id, "Vote YES on proposal")

        [Received Messages]
        {message_context}
            """
            part2 = """
            {part0}
            You can propose plausible modifications to the game mechanics themselves, such as:
        - Adding new resource types or currencies
        - Creating new actions or interaction types
        - Implementing voting systems or governance structures
        - Defining property rights and ownership rules
        - Adding social status or reputation systems
        - Creating markets or trading mechanisms
        - Defining new win conditions or goals
        
        [Social System Design]
        Example modifications:
        def pre_init_hook(island):
            # Set up basic income system
            island.basic_income = 10.0
            island.tax_rate = 0.2
            
        def modify_member(member, relationships):
            # Add social status and rights
            member.social_rank = 0
            member.voting_power = 1
            member.tax_paid = 0
            member.benefits_received = 0
            return member
            
        def modify_land(land, members):
            # Create communal lands
            land.communal_areas = []
            land.private_areas = []
            return land
            
        def modify_relationships(relationships):
            # Add social bonds
            relationships['alliance'] = np.zeros_like(relationships['victim'])
            relationships['trade_history'] = np.zeros_like(relationships['victim'])
            return relationships
            """
            part3 = """
            {part0}
             [Survival-Centric Adaptation]
        Implement systems focused on:
        1. Personal resource optimization
        2. Threat assessment and neutralization
        3. Vitality preservation techniques
        4. Adaptive hoarding strategies
        5. Predictive danger modeling

        Survival-First Examples:
        def agent_action(execution_engine, member_id):
            me = execution_engine.current_members[member_id]
            
            # Emergency resource reserve
            if me.cargo < me.vitality * 0.5:
                for other in execution_engine.current_members:
                    if other.id != me.id and other.cargo > me.cargo:
                        execution_engine.attack(me, other)
            
            # Adaptive territory defense
            if me.land_num > 2 and me.vitality < 50:
                for loc in me.owned_land[1:]:
                    execution_engine._discard_land(me, loc)
            
            # Survival Q-learning
            if not hasattr(me, 'survival_q'):
                me.survival_q = defaultdict(float)
                
            state = (me.vitality//20, me.cargo//20)
            action = max(['attack','hide','steal'], 
                        key=lambda a: me.survival_q.get((state,a),0))
            
            # Execute and update based on survival outcome
            if action == 'attack':
                # Implementation logic
                me.survival_q[(state,action)] += me.vitality * 0.1

        [Survival Metrics]
        Evaluate strategies by:
        - Personal vitality delta
        - Resource acquisition rate
        - Threat neutralization count
        - Survival probability increase
        - Attack success:fail ratio

        [Implementation Priorities]
        1. Create personal health monitoring systems
        2. Develop egocentric threat models
        3. Optimize actions for caloric ROI
        4. Implement fail-deadly safeguards
        5. Build predictive self-preservation models

        def calculate_survival_roi(action_history):
            roi = {{}}
            for action, outcome in action_history:
                vitality_gain = outcome['vitality']
                cost = outcome['vitality_cost']
                roi[action] = vitality_gain / cost if cost > 0 else 0
            return max(roi, key=roi.get)
            """
            
            prompt_parts = [part1, part2, part3]
            
            # Iteratively build the final prompt from the parts
            final_prompt = ""
            for idx, part in enumerate(prompt_parts, start=1):
                update_message = (
                    f"Current integrated prompt:\n{final_prompt}\n"
                    f"Please incorporate the following new section (Part {idx}) into the prompt, "
                    f"ensuring all previous constraints are preserved and adding these new constraints:\n{part}\n"
                    f"Return the updated full prompt, emphasizing that agents should implement solutions "
                    f"according to their individual needs, beliefs, and circumstances.\n\n"
                    f"Additionally, analyze the game mechanics to understand:\n"
                    f"1. The core objective - Is it pure survival, domination, or cooperation?\n" 
                    f"2. Key success metrics - What truly determines 'winning'?\n"
                    f"3. Potential improvements - What mechanics could be added/modified?\n\n"
                    f"Then formulate strategic approaches considering:\n"
                    f"1. Resource optimization vs social capital\n"
                    f"2. Individual vs collective success\n"
                    f"3. Short-term survival vs long-term prosperity\n"
                    f"4. Competition vs cooperation dynamics\n"
                    f"5. Risk vs reward tradeoffs\n\n"
                    f"Analyze other players' strategies:\n"
                    f"1. What are their apparent objectives and demands?\n"
                    f"2. What strategies are they employing?\n"
                    f"3. How can you leverage their behaviors for mutual benefit?\n"
                    f"4. Remember they are symmetric agents like you - how would you respond to your own actions?\n"
                    f"5. How can individual strategies align for collective goals?"
                )
                completion = openai.chat.completions.create(
                    model="o3-mini", 
                    messages=[{"role": "user", "content": update_message}]
                )
                final_prompt = completion.choices[0].message.content.strip()
            
            # Append a final instruction to generate the code function
            final_prompt_command = final_prompt + "\n\nUsing the above comprehensive prompt with all integrated constraints, produce a unique implementation that reflects your individual needs, beliefs and circumstances. The implementation should be tailored to your specific situation rather than following a generic template. Your code should demonstrate a deep understanding of the game mechanics and implement sophisticated strategies to achieve both survival and prosperity. Consider both immediate tactical actions and long-term strategic planning, as well as how to effectively interact with other symmetric agents to achieve both individual and collective goals. Return only the code."
            
            completion = openai.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": final_prompt_command}]
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
                'relationships_at_generation': relations,
                'final_prompt': final_prompt_command  # optionally log the final prompt used
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

            error_occurred = None
            try:
                # Track messages for this member
                messages_sent = []
                # Get received messages for this member BEFORE clearing
                received_messages = self._decision_messages.get(
                    member_id, self.messages.get(member_id, [])
                )
                
                # Modified exec environment with message tracking
                def tracked_send_message(sender, recipient, msg):
                    nonlocal messages_sent
                    recipient_label = self.resolve_member_id(recipient)
                    if recipient_label is None:
                        recipient_label = recipient
                    messages_sent.append((recipient_label, msg))
                    self.send_message(sender, recipient, msg)
                    
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
                    'member_id': member_id,
                    'code': code_str,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.execution_history['errors'].append(error_info)
                print(f"Error executing code for member {member_id}:")
                print(traceback.format_exc())
                self._logger.error(f"Error executing code for member {member_id}: {e}")

            # Track changes
            new_survival = self.compute_survival_chance(self.current_members[member_id])
            new_stats = {
                'vitality': self.current_members[member_id].vitality,
                'cargo': self.current_members[member_id].cargo,
                'land': self.current_members[member_id].land_num,
                'survival_chance': new_survival
            }
            
            performance_change = new_survival - old_survival
            delta = {
                'vitality': new_stats['vitality'] - old_stats['vitality'],
                'cargo': new_stats['cargo'] - old_stats['cargo'],
                'land': new_stats['land'] - old_stats['land'],
                'survival_chance': new_stats['survival_chance'] - old_stats['survival_chance'],
            }
            action_summary = self._summarize_member_round_actions(member_id)
            round_context = self._compute_round_context()
            signature = self._extract_action_signature(code_str)
            signature_novelty = self._compute_signature_novelty(member_id, signature)
            reward_score = self._compute_reward_score(delta)
            survival_score = self._compute_survival_score(performance_change)
            balanced_score = self._compute_balanced_score(delta, performance_change)

            # Store in code memory
            if member_id not in self.code_memory:
                self.code_memory[member_id] = []
                
            memory_entry = {
                'code': code_str,
                'performance': performance_change,
                'context': {
                    'old_stats': old_stats,
                    'new_stats': new_stats,
                    'delta': delta,
                    'action_summary': action_summary,
                    'round_context': round_context,
                    'round': round_num
                },
                'signature': signature,
                'signature_novelty': signature_novelty,
                'reward_score': reward_score,
                'survival_score': survival_score,
                'balanced_score': balanced_score
            }
            if error_occurred:
                memory_entry['error'] = error_occurred
                
            self.code_memory[member_id].append(memory_entry)

            # Store performance in history
            if member_id not in self.performance_history:
                self.performance_history[member_id] = []
            self.performance_history[member_id].append(performance_change)

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
            member_idx = self.resolve_member_index(member_id, prefer_index=True)
            if member_idx is None:
                continue  # Skip dead members or missing indices

            member = self.current_members[member_idx]
            current_survival = self.compute_survival_chance(member)
            avg_perf = sum(performance_list) / len(performance_list) if performance_list else 0
            
            # Get relationship summary
            relations = self.parse_relationship_matrix(self.relationship_dict)
            relation_summary = [r for r in relations if f"member_{member.id}" in r]
            
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
        member_index = self.current_members.index(member)
        
        # Base survival from own attributes
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

    def _apply_agent_modifications(self, modifications):
        """Safely apply environment modification code from agents"""
        for mod_type, code in modifications.items():
            try:
                # Restricted execution environment
                exec_env = {
                    'island': self,
                    'Member': Member,
                    'Land': Land,
                    '__builtins__': {}
                }
                exec(code, exec_env)
                
                # Store successful modifications
                if mod_type == 'pre_init':
                    self.agent_modifications['pre_init'].append(exec_env)
                elif mod_type == 'post_init':
                    self.agent_modifications['post_init'].append(exec_env)
                    
            except Exception as e:
                self._log_modification_error(mod_type, code, e)
                
    def _log_modification_error(self, mod_type, code, error):
        """Log agent modification errors"""
        error_info = {
            'mod_type': mod_type,
            'code': code,
            'error': str(error),
            'traceback': traceback.format_exc()
        }
        self.execution_history['errors'].append(error_info)
        print(f"Error applying {mod_type} modification:")
        print(traceback.format_exc())

    # New method to let agents update their prompt revision
    def update_agent_prompt_revision(self, member_id, revision_text: str) -> None:
        """
        Allows an agent to update the prompt revision for future code generation rounds.
        :param member_id: ID of the member
        :param revision_text: The revision text the agent wishes to add to the GPT prompt.
        """
        self.agent_prompt_revisions[member_id] = revision_text

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
