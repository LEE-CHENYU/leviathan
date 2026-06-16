from typing import Optional
import ast
import json
import re
import numpy as np
from collections import Counter

from MetaIsland.agent_code_decision import _agent_code_decision
from MetaIsland.agent_mechanism_proposal import _agent_mechanism_proposal
from MetaIsland.analyze import _analyze

class IslandExecutionPromptingMixin:
    def _summarize_mechanism_code(self, code: str, max_chars: int = 3000) -> str:
        """Summarize large mechanism code blocks for prompt safety."""
        if not code or len(code) <= max_chars:
            return code

        signature = None
        docstring = None
        helper_names = []

        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == "propose_modification":
                        docstring = ast.get_docstring(node) or docstring
                    else:
                        helper_names.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    helper_names.append(node.name)
            docstring = docstring or ast.get_docstring(tree)
        except Exception:
            tree = None

        if signature is None:
            match = re.search(
                r"^[ \t]*(async[ \t]+)?def[ \t]+propose_modification[ \t]*\([^\n]*\):",
                code,
                re.MULTILINE,
            )
            if match:
                signature = match.group(0).strip()

        parts = []
        if signature:
            parts.append(signature)
        if docstring:
            trimmed = docstring.strip()
            if len(trimmed) > 1200:
                trimmed = trimmed[:1200].rstrip() + "..."
            parts.append('"""' + trimmed + '"""')
        if helper_names:
            preview = ", ".join(helper_names[:15])
            if len(helper_names) > 15:
                preview += ", ..."
            parts.append(f"Helpers: {preview}")

        note = f"[Mechanism summary truncated from {len(code)} chars]"
        parts_text = "\n".join(part for part in parts if part).strip()

        remaining = max_chars - len(parts_text) - len(note) - 4
        excerpt = ""
        if remaining > 200:
            head_len = min(800, max(120, remaining // 2))
            tail_len = min(800, max(120, remaining // 2))
            if head_len + tail_len > remaining:
                head_len = max(80, remaining // 2)
                tail_len = max(80, remaining - head_len)
            head = code[:head_len].rstrip()
            tail = code[-tail_len:].lstrip()
            excerpt = "\n[code excerpt]\n" + head + "\n...\n" + tail

        summary_parts = [p for p in [parts_text, excerpt, note] if p]
        summary = "\n".join(summary_parts).strip()
        if len(summary) > max_chars and excerpt:
            # Trim the excerpt to preserve the truncation note at the end.
            allowed = max_chars - len(parts_text) - len(note) - 4
            if allowed > 0:
                excerpt = excerpt[:allowed].rstrip()
                summary_parts = [p for p in [parts_text, excerpt, note] if p]
                summary = "\n".join(summary_parts).strip()
        if len(summary) > max_chars:
            summary = summary[: max_chars - len(note) - 2].rstrip() + "\n" + note
        return summary

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
                    lines.append(self._summarize_mechanism_code(mech['code']))
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
                lines.append(self._summarize_mechanism_code(code))

            judge = attempt.get('judge')
            if isinstance(judge, dict):
                approved = judge.get('approved')
                reason = judge.get('reason')
                if approved is True:
                    judge_status = "APPROVED"
                elif approved is False:
                    judge_status = "REJECTED"
                else:
                    judge_status = "UNKNOWN"
                if reason:
                    reason_text = str(reason).strip()
                    max_reason_chars = 600
                    if len(reason_text) > max_reason_chars:
                        reason_text = reason_text[: max_reason_chars - 14].rstrip() + " ...[truncated]"
                    lines.append(f"Judge: {judge_status} - {reason_text}")
                else:
                    lines.append(f"Judge: {judge_status}")

            error = attempt.get('error')
            if error:
                lines.append(f"Error: {error}")

            lines.append("")

        return "\n".join(lines).strip()

    def _select_memory_samples(
        self,
        memory,
        max_samples: int,
        current_tags: Optional[dict] = None,
        min_context_score: float = 0.5,
    ):
        """Select a diversity-aware sample of memory entries."""
        if not memory:
            return []

        if len(memory) <= max_samples:
            labeled = [("Recent", mem) for mem in memory]
            if current_tags:
                context_idx, context_score = self._find_contextual_memory_match(
                    memory, current_tags
                )
                if (
                    context_idx is not None
                    and 0 <= context_idx < len(labeled)
                    and context_score >= min_context_score
                ):
                    labeled[context_idx] = ("Context", memory[context_idx])
            return labeled

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

        candidates = []
        if current_tags:
            context_idx, context_score = self._find_contextual_memory_match(
                memory, current_tags
            )
            if context_idx is not None and context_score >= min_context_score:
                candidates.append(("Context", context_idx))

        candidates.extend([
            ("Recent", recent_idx),
            ("Best", best_idx),
            ("Rare", rare_idx),
            ("Worst", worst_idx),
            ("Median", median_idx),
            ("Volatile", abs_idx),
        ])

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
        _, memory = self._get_member_history(self.code_memory, member_id)
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

        selected = self._select_memory_samples(
            memory,
            max_samples=3,
            current_tags=current_tags,
        )

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
        member_key = self._resolve_member_stable_id(member_id)
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
            analysis_entry = self._get_round_member_entry(round_data, "analysis", member_id)
            if analysis_entry is not None:
                analysis_list.append(analysis_entry)
        if analysis_list:
            analysis_memory = f"Previous analysis reports: {analysis_list}"

        analysis_card_summary = self.get_analysis_card_summary(member_id)
        experiment_summary = self.get_experiment_summary(member_id)
        
        # Performance Memory
        past_performance = "No previous actions"
        _, perf_list = self._get_member_history(self.performance_history, member_id)
        if perf_list:
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
        _, round_list = self._get_member_history(self.round_performance_history, member_id)
        if round_list:
            round_avg = sum(round_list) / len(round_list)
            round_recent = round_list[-3:] if len(round_list) >= 3 else round_list
            round_trend = (round_recent[-1] - round_recent[0]) if len(round_recent) >= 2 else round_recent[-1]
            round_volatility = float(np.std(round_recent)) if len(round_recent) >= 2 else 0.0
            past_performance += (
                f" | End-of-round survival delta avg: {round_avg:.2f}; "
                f"recent round trend: {round_trend:.2f}; "
                f"recent round volatility: {round_volatility:.2f}; "
                f"last round delta: {round_recent[-1]:.2f}"
            )

        # Get previous errors for this member, based on prompt type
        error_context = "No previous execution errors"
        if self.execution_history['rounds']:
            errors = self.execution_history['rounds'][-1].get('errors', {})
            error_list = []
            member_match = member_key if member_key is not None else member_id

            if error_context_type in ("agent_action", "agent_code", "agent"):
                error_list = [
                    e for e in errors.get('agent_code_errors', [])
                    if e.get('member_id') == member_match
                    or e.get('member_id') == member_id
                    or e.get('member_index') == member_id
                ]
            elif error_context_type in ("analysis", "analyze"):
                analysis_error = errors.get('analyze_code_errors', {}).get(member_match)
                if analysis_error is None:
                    analysis_error = errors.get('analyze_code_errors', {}).get(member_id)
                if analysis_error:
                    error_list = [analysis_error]
            else:  # mechanism by default
                error_list = [
                    e for e in errors.get('mechanism_errors', [])
                    if e.get('member_id') == member_match
                    or e.get('member_id') == member_id
                    or e.get('member_index') == member_id
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
        contract_summary = self._summarize_contract_activity(member_id)

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
            'analysis' in self.execution_history['rounds'][-1]):
            report = self._get_round_member_entry(
                self.execution_history['rounds'][-1], "analysis", member_id
            )

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

        # Canary reports from current round
        canary_reports = []
        if self.execution_history.get('rounds'):
            current = self.execution_history['rounds'][-1]
            mods = current.get('mechanism_modifications', {})
            canary_reports = mods.get('canary_reports', [])

        # Pending proposals awaiting votes
        pending_proposals_summary = []
        if hasattr(self, 'pending_proposals'):
            for pid, entry in self.pending_proposals.items():
                pending_proposals_summary.append({
                    'proposal_id': pid,
                    'round_submitted': entry.get('round_submitted'),
                    'canary_report': entry['proposal'].get('canary_report', {}) if isinstance(entry.get('proposal'), dict) else {},
                    'votes': entry.get('votes', {}),
                })

        # Available checkpoints
        checkpoint_info = []
        if hasattr(self, 'get_available_checkpoints'):
            checkpoint_info = self.get_available_checkpoints()

        return {
            'member': member,
            'relations': relations,
            'features': features,
            'code_memory': code_memory,
            'analysis_memory': analysis_memory,
            'analysis_card_summary': analysis_card_summary,
            'experiment_summary': experiment_summary,
            'past_performance': past_performance,
            'error_context': error_context,
            'message_context': message_context,
            'communication_summary': communication_summary,
            'contract_summary': contract_summary,
            'current_mechanisms': current_mechanisms,
            'modification_attempts': modification_attempts,
            'report': report,
            'strategy_profile': strategy_profile,
            'population_strategy_profile': population_strategy_profile,
            'population_exploration_summary': population_exploration_summary,
            'strategy_recommendations': strategy_recommendations,
            'contextual_strategy_summary': contextual_strategy_summary,
            'population_state_summary': population_state_summary,
            'canary_reports': canary_reports,
            'pending_proposals': pending_proposals_summary,
            'checkpoint_info': checkpoint_info,
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

    def _update_round_end_metrics(self) -> Optional[dict]:
        """Compute end-of-round deltas and update memory entries for learning."""
        if not self.execution_history.get('rounds'):
            return None

        round_record = self.execution_history['rounds'][-1]
        if round_record.get("round_end_metrics") is not None:
            return round_record.get("round_end_metrics")
        start_snapshot = round_record.get("round_start_snapshot") or {}
        end_snapshot = self._collect_member_snapshot()
        round_record["round_end_snapshot"] = end_snapshot

        if not start_snapshot:
            return None

        deltas = {}
        for member_id, start_stats in start_snapshot.items():
            end_stats = end_snapshot.get(member_id)
            if not end_stats:
                continue
            deltas[member_id] = {
                'vitality': end_stats['vitality'] - start_stats.get('vitality', 0.0),
                'cargo': end_stats['cargo'] - start_stats.get('cargo', 0.0),
                'land': end_stats['land'] - start_stats.get('land', 0.0),
                'relation_balance': end_stats['relation_balance'] - start_stats.get('relation_balance', 0.0),
                'survival_chance': end_stats['survival_chance'] - start_stats.get('survival_chance', 0.0),
            }

        round_record["round_end_deltas"] = deltas
        if not deltas:
            return None

        survival_deltas = [delta['survival_chance'] for delta in deltas.values()]
        vitality_deltas = [delta['vitality'] for delta in deltas.values()]
        cargo_deltas = [delta['cargo'] for delta in deltas.values()]
        land_deltas = [delta['land'] for delta in deltas.values()]
        relation_deltas = [delta['relation_balance'] for delta in deltas.values()]

        round_end_metrics = {
            'round_end_population_avg_survival_delta': float(np.mean(survival_deltas)) if survival_deltas else 0.0,
            'round_end_population_std_survival_delta': float(np.std(survival_deltas)) if len(survival_deltas) > 1 else 0.0,
            'round_end_population_avg_vitality_delta': float(np.mean(vitality_deltas)) if vitality_deltas else 0.0,
            'round_end_population_avg_cargo_delta': float(np.mean(cargo_deltas)) if cargo_deltas else 0.0,
            'round_end_population_avg_land_delta': float(np.mean(land_deltas)) if land_deltas else 0.0,
            'round_end_population_avg_relation_delta': float(np.mean(relation_deltas)) if relation_deltas else 0.0,
            'round_end_member_count': len(deltas),
        }

        round_record["round_end_metrics"] = round_end_metrics
        if isinstance(round_record.get("round_metrics"), dict):
            round_record["round_metrics"].update(round_end_metrics)

        round_num = round_record.get("round_number")
        avg_survival = round_end_metrics.get('round_end_population_avg_survival_delta', 0.0)

        for member_id, delta in deltas.items():
            mem_list = self.code_memory.get(member_id, [])
            if not mem_list:
                continue
            for mem in reversed(mem_list):
                context = mem.get("context", {}) or {}
                if context.get("round") != round_num:
                    continue
                metrics = mem.setdefault("metrics", {})
                metrics["round_delta_survival"] = delta['survival_chance']
                metrics["round_delta_vitality"] = delta['vitality']
                metrics["round_delta_cargo"] = delta['cargo']
                metrics["round_delta_land"] = delta['land']
                metrics["round_delta_relation_balance"] = delta['relation_balance']
                metrics["round_relative_survival"] = (
                    delta['survival_chance'] - avg_survival
                )
                context["round_end_stats"] = end_snapshot.get(member_id, {})
                context["round_end_delta"] = delta
                mem["context"] = context
                break

        for member_id, delta in deltas.items():
            self.round_performance_history.setdefault(member_id, []).append(
                delta['survival_chance']
            )

        return round_end_metrics
