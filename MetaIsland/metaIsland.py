from typing import List, Tuple, Optional
import numpy as np
import json
from datetime import datetime
import traceback
import os
from collections import Counter
import inspect
import asyncio

from MetaIsland.base_island import Island
from MetaIsland.meta_island_strategy import IslandExecutionStrategyMixin, StrategyMemory
from MetaIsland.meta_island_signature import IslandExecutionSignatureMixin
from MetaIsland.meta_island_population import IslandExecutionPopulationMixin
from MetaIsland.meta_island_prompting import IslandExecutionPromptingMixin

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

from MetaIsland.llm_client import get_llm_client

client = get_llm_client()

from MetaIsland.model_router import model_router
from MetaIsland.llm_utils import ensure_non_empty_response

provider, model_id = model_router("default")


class IslandExecution(
    IslandExecutionStrategyMixin,
    IslandExecutionSignatureMixin,
    IslandExecutionPopulationMixin,
    IslandExecutionPromptingMixin,
    Island,
):
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
        
        self.performance_history = {}  # {stable_member_id: [list_of_performance_metrics]}
        self.round_performance_history = {}  # {stable_member_id: [per-round survival deltas]}
        
        # Add code memory tracking
        self.code_memory = {}  # {stable_member_id: [{'code': str, 'performance': float, 'context': dict}]}
        
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

        # Diversity controller for adaptive exploration pressure
        self._diversity_controller = {
            "alpha": 0.25,
            "target_diversity": 0.45,
            "target_entropy": 0.6,
            "adjustment": 0.0,
            "min_adjust": -0.15,
            "max_adjust": 0.25,
        }

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
            "agent_messages": {},     # Dictionary keyed by stable member id
            "population_strategy_profile": None,
            "mechanism_modifications": {
                "attempts": [],       # Proposed modifications this round
                "approved_ids": [],   # Approved member ids (judge-gated)
                "approved_count": 0,  # Approved proposal count
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
        # Capture round-start snapshot for end-of-round evaluation
        try:
            round_record["round_start_snapshot"] = self._collect_member_snapshot()
        except Exception:
            round_record["round_start_snapshot"] = {}
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

            member_index = self._resolve_member_index(member_id)
            if member_index is None:
                continue
            member_key = self._resolve_member_stable_id(member_index)
            if member_key is None:
                continue

            print(f"\nExecuting code for Member {member_key} (idx {member_index}):")
            # print(code_str)

            self._ensure_strategy_memory_appendable(self.current_members[member_index])

            # Track old stats before executing
            old_survival = self.compute_survival_chance(self.current_members[member_index])
            old_relation_balance = self.compute_relation_balance(self.current_members[member_index])
            old_stats = {
                'vitality': self.current_members[member_index].vitality,
                'cargo': self.current_members[member_index].cargo,
                'land': self.current_members[member_index].land_num,
                'relation_balance': old_relation_balance,
                'survival_chance': old_survival
            }
            context_tags = self._classify_context_tags(old_stats, context_cutoffs)
            context_key = self._context_key_from_tags(context_tags)

            error_occurred = None
            # Track messages for this member
            messages_sent = []
            # Get received messages that were surfaced to the agent (if any)
            received_messages = self._peek_messages(member_index, create_snapshot=False)
            original_send_message = self.send_message

            # Modified exec environment with message tracking
            def tracked_send_message(sender, recipient, msg):
                nonlocal messages_sent
                resolved_recipient = self._resolve_member_stable_id(recipient) or recipient
                messages_sent.append((resolved_recipient, msg))
                original_send_message(sender, recipient, msg)

            try:
                self.send_message = tracked_send_message
                local_env = {
                    'execution_engine': self,
                    'send_message': tracked_send_message,
                    'np': np,
                }

                # Execute the code in a way that makes the function accessible
                code_stats = {
                    "raw_len": len(code_str) if code_str else 0,
                    "raw_lines": code_str.count("\n") + 1 if code_str else 0,
                }
                cleaned_code = self.clean_code_string(code_str)
                code_stats["cleaned_len"] = len(cleaned_code) if cleaned_code else 0
                code_stats["cleaned_lines"] = cleaned_code.count("\n") + 1 if cleaned_code else 0
                exec(cleaned_code, local_env)

                if 'agent_action' in local_env and callable(local_env['agent_action']):
                    print(f"Executing agent_action() for Member {member_key}")
                    # Pass self as execution_engine and member_id
                    local_env['agent_action'](self, member_index)
                else:
                    error_occurred = "No valid agent_action() found"
                    print(f"No valid agent_action() found for Member {member_key}")
                    self._logger.warning(f"No valid agent_action() found for member {member_key}.")

            except Exception as e:
                error_occurred = str(e)
                error_info = {
                    'round': round_num,
                    'type': 'agent_code_execution',
                    'member_id': member_key,
                    'member_index': member_index,
                    'code': code_str,
                    'error': str(e),
                    'error_category': self._classify_agent_execution_error(e),
                    'error_details': self._describe_execution_error(e),
                    'code_stats': code_stats,
                    'traceback': traceback.format_exc()
                }
                self.execution_history['rounds'][-1]['errors']['agent_code_errors'].append(error_info)
                print(f"Error executing code for member {member_key}:")
                print(traceback.format_exc())
                self._logger.error(f"Error executing code for member {member_key}: {e}")
            finally:
                self.send_message = original_send_message

            # Track changes
            new_survival = self.compute_survival_chance(self.current_members[member_index])
            new_relation_balance = self.compute_relation_balance(self.current_members[member_index])
            new_stats = {
                'vitality': self.current_members[member_index].vitality,
                'cargo': self.current_members[member_index].cargo,
                'land': self.current_members[member_index].land_num,
                'relation_balance': new_relation_balance,
                'survival_chance': new_survival
            }
            
            performance_change = new_survival - old_survival

            # Store in code memory
            if member_key not in self.code_memory:
                self.code_memory[member_key] = []
                
            signature = self._extract_action_signature(code_str)
            signature_novelty = self._compute_signature_novelty(member_index, signature)
            metrics = {
                'delta_vitality': new_stats['vitality'] - old_stats['vitality'],
                'delta_cargo': new_stats['cargo'] - old_stats['cargo'],
                'delta_land': new_stats['land'] - old_stats['land'],
                'delta_relation_balance': new_stats['relation_balance'] - old_stats['relation_balance'],
                'delta_survival': performance_change,
            }
            self._auto_update_strategy_memory(
                self.current_members[member_index],
                round_num,
                signature,
                metrics,
                context_tags,
            )
            experiment = self._record_experiment_outcome(
                member_index,
                round_num,
                signature,
                metrics,
                context_key=context_key,
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
                self.current_members[member_index]
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
            if experiment:
                memory_entry['experiment'] = experiment
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
            self.execution_history['rounds'][-1]['agent_actions'].append({
                'member_id': member_key,
                'member_index': member_index,
                'code_executed': code_str,
                'old_stats': old_stats,
                'new_stats': new_stats,
                'performance_change': performance_change,
                'messages_sent': messages_sent
            })

            # Log messages in round data
            self.execution_history['rounds'][-1]['agent_messages'][member_key] = {
                'member_index': member_index,
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
        plan_stats = self._collect_plan_alignment_stats(round_num)
        plan_feasibility = self._collect_plan_feasibility_stats(round_num)
        error_stats = self._collect_agent_error_stats(round_num)
        plan_samples = plan_stats.get("plan_samples", 0)
        plan_matched = plan_stats.get("baseline", 0) + plan_stats.get("variation", 0)
        plan_alignment_rate = (
            plan_matched / plan_samples if plan_samples else None
        )
        baseline_rate = (
            plan_stats.get("baseline", 0) / plan_samples if plan_samples else None
        )
        variation_rate = (
            plan_stats.get("variation", 0) / plan_samples if plan_samples else None
        )
        unmatched_rate = (
            plan_stats.get("unmatched", 0) / plan_samples if plan_samples else None
        )
        coverage_rate = (
            plan_samples / plan_stats.get("total_actions", 0)
            if plan_stats.get("total_actions", 0)
            else None
        )

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
            'plan_alignment_rate': plan_alignment_rate,
            'plan_alignment_baseline_rate': baseline_rate,
            'plan_alignment_variation_rate': variation_rate,
            'plan_alignment_unmatched_rate': unmatched_rate,
            'plan_alignment_avg_match_score': plan_stats.get("avg_match_score"),
            'plan_alignment_plan_samples': plan_samples,
            'plan_alignment_total_actions': plan_stats.get("total_actions", 0),
            'plan_alignment_plan_coverage': coverage_rate,
            'plan_alignment_missing_plans': plan_stats.get("missing", 0),
            'plan_ineligible_tag_rate': plan_feasibility.get("plan_ineligible_tag_rate"),
            'plan_only_tag_rate': plan_feasibility.get("plan_only_tag_rate"),
            'plan_tag_total': plan_feasibility.get("plan_tag_total", 0),
            'plan_feasible_tag_total': plan_feasibility.get("plan_feasible_tag_total", 0),
            'plan_ineligible_tag_count': plan_feasibility.get("plan_ineligible_tag_count", 0),
            'plan_only_tag_count': plan_feasibility.get("plan_only_tag_count", 0),
            'plan_feasibility_samples': plan_feasibility.get("plan_samples", 0),
            'plan_feasibility_missing': plan_feasibility.get("plan_missing", 0),
            'plan_feasibility_missing_reason_counts': plan_feasibility.get(
                "plan_missing_reason_counts",
                {},
            ),
            'agent_code_error_count': error_stats.get("agent_code_error_count", 0),
            'agent_code_error_rate': (
                error_stats.get("agent_code_error_count", 0) / len(round_deltas)
                if round_deltas
                else None
            ),
            'agent_code_error_tag_counts': error_stats.get("agent_code_error_tag_counts", {}),
            'agent_code_error_type_counts': error_stats.get("agent_code_error_type_counts", {}),
        }

        active_ids = {member.id for member in self.current_members}
        memory_keys = set(self.code_memory.keys())
        memory_missing = len(active_ids - memory_keys)
        memory_orphan = len(memory_keys - active_ids)
        memory_active_coverage = (
            len(active_ids & memory_keys) / len(active_ids)
            if active_ids
            else None
        )
        self.execution_history['rounds'][-1]['round_metrics'].update({
            'memory_active_coverage': memory_active_coverage,
            'memory_missing_count': memory_missing,
            'memory_orphan_count': memory_orphan,
        })

        round_record = self.execution_history['rounds'][-1]
        mods_record = round_record.get('mechanism_modifications') or {}
        attempts = mods_record.get('attempts') or []
        executed = mods_record.get('executed') or []
        approved_count = mods_record.get('approved_count')
        if approved_count is None:
            approved_count = len(mods_record.get('approved_ids') or [])
        errors = round_record.get('errors', {}).get('mechanism_errors') or []
        mechanism_error_type_counts = {}
        if isinstance(errors, list) and errors:
            type_counts = Counter()
            for error_info in errors:
                error_category = None
                if isinstance(error_info, dict):
                    error_category = error_info.get("error_category")
                if error_category:
                    type_counts[error_category] += 1
                else:
                    type_counts["unknown"] += 1
            mechanism_error_type_counts = dict(type_counts)
        self.execution_history['rounds'][-1]['round_metrics'].update({
            'mechanism_attempted_count': len(attempts) if isinstance(attempts, list) else 0,
            'mechanism_approved_count': int(approved_count or 0),
            'mechanism_executed_count': len(executed) if isinstance(executed, list) else 0,
            'mechanism_error_count': len(errors) if isinstance(errors, list) else 0,
            'mechanism_error_type_counts': mechanism_error_type_counts,
        })

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

        # End-of-round deltas are recorded in _update_round_end_metrics()
        for member_id, delta in round_deltas.items():
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
            member_index = self._resolve_member_index(member_id)
            if member_index is None:
                continue  # Skip dead or unmapped members
            member = self.current_members[member_index]
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
        sender_key = sender_id
        if not any(m.id == sender_id for m in self.current_members):
            sender_index = self._resolve_member_index(sender_id)
            if sender_index is not None:
                sender_key = self.current_members[sender_index].id

        recipient_key = None
        if any(m.id == recipient_id for m in self.current_members):
            recipient_key = recipient_id
        else:
            recipient_index = self._resolve_member_index(recipient_id)
            if recipient_index is not None:
                recipient_key = self.current_members[recipient_index].id

        display_recipient = recipient_key if recipient_key is not None else recipient_id
        print(f"[MSG] Member {sender_key} -> Member {display_recipient}: {message!r}")
        if recipient_key is not None:
            if recipient_key not in self.messages:
                self.messages[recipient_key] = []
            self.messages[recipient_key].append(f"From member_{sender_key}: {message}")
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
                    
    def execute_mechanism_modifications(self, approved: Optional[List[dict]] = None):
        """Execute ratified modifications (optionally restricted to approved proposals)."""
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
        exec_env = {
            'execution_engine': self,
            'np': np,
        }

        if not self.execution_history.get('rounds'):
            return

        if approved is not None:
            round_record = self.execution_history['rounds'][-1]
            mods_record = round_record.get('mechanism_modifications')
            if isinstance(mods_record, dict):
                approved_ids = [
                    mod.get('member_id')
                    for mod in approved
                    if isinstance(mod, dict)
                ]
                mods_record['approved_ids'] = approved_ids
                mods_record['approved_count'] = len(approved_ids)
            mods = [
                mod for mod in approved
                if isinstance(mod, dict) and mod.get('code')
            ]
        else:
            mods = self.execution_history['rounds'][-1]['mechanism_modifications'].get('attempts', [])
            mods = [
                mod for mod in mods
                if isinstance(mod, dict) and mod.get('code')
            ]

        if not mods:
            print("\n[Mechanisms] No mechanism modifications to execute")
            return

        mod = None
        if len(mods) == 1:
            mod = mods[0]
        else:
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
                aggregated_code = ensure_non_empty_response(
                    completion.choices[0].message.content,
                    context="mechanism_aggregate",
                )
                code_result = self.clean_code_string(aggregated_code)
                
                self.save_generated_code(code_result, '-1', 'aggregated_mechanism')
                # print(f"Aggregated Mechanism Code: {code_result}")
                
                mod = {
                    'member_id': 'aggregated',
                    'code': code_result,
                    'round': current_round
                }
                
            except Exception as e:
                print(f"Error getting aggregated modification from O1: {e}")
                # Fall back to first modification if aggregation fails
                mod = mods[0] if mods else None

        if not mod or not mod.get('code'):
            return

        member_id = mod.get('member_id', 'unknown')
        mod.setdefault('round', current_round)

        try:
            print(f"\nExecuting modification code for Member {member_id}:")
            # print(mod['code'])

            # Execute modification code
            exec(mod['code'], exec_env)
            exec_env['propose_modification'](self)
            print("Mechanism modification code executed successfully.")
            
            # Get execution class attributes
            class_attrs = self.get_execution_class_attributes(member_id)
            # Define a function to save mechanism execution data to JSON
            def save_mechanism_execution_to_json(mechanism_data, json_path):
                with open(json_path, 'w') as f:
                    json.dump(mechanism_data, f, indent=4, default=str)
            
            # Prepare the mechanism execution data
            mechanism_data = {
                'member_id': member_id,
                'round': current_round,
                'code': mod['code'],
                'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'class_attributes': class_attrs
            }
            
            # Create a filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_filename = f'mechanism_execution_{member_id}_round_{current_round}_{timestamp}.json'
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
                'error_category': 'mechanism_execution',
                'traceback': traceback.format_exc(),
                'code': mod.get('code'),
                'member_id': member_id
            }
            self.execution_history['rounds'][-1]['errors']['mechanism_errors'].append(error_info)
            print(f"Error executing code for member {member_id}:")
            print(traceback.format_exc())
            self._logger.error(f"Error executing code for member {member_id}: {e}")

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
            ('propose_mechanisms', 'judge', 'proposals', 'proposals'),
            ('judge', 'execute_mechanisms', 'approved', 'approved'),
            ('execute_mechanisms', 'agent_decisions'),
            ('agent_decisions', 'execute_actions'),
            ('execute_actions', 'contracts'),
            ('contracts', 'produce'),
            ('produce', 'consume'),
            ('consume', 'environment'),
            ('environment', 'log_status'),
        ]

        for connection in connections:
            if len(connection) == 2:
                from_name, to_name = connection
                self.graph.connect(from_name, to_name)
            else:
                from_name, to_name, output_key, input_key = connection
                self.graph.connect(
                    from_name,
                    to_name,
                    output_key=output_key,
                    input_key=input_key,
                )

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
            exec._update_round_end_metrics()

            print(f"\nSurviving members at end of round: {len(exec.current_members)}")

            for member in exec.current_members:
                survival_chance = exec.compute_survival_chance(member)
                status = f"Member {member.id}: Vitality={member.vitality:.2f}, Cargo={member.cargo:.2f}, Survival={survival_chance:.2f}"
                print(status)

if __name__ == "__main__":
    asyncio.run(main())
