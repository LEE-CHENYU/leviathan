from Leviathan.Island import Island
import Leviathan.api_key
from Leviathan.Member import Member
from Leviathan.islandExecution_agent_code_generation_analysis import IslandExecutionAnalysisMixin
from Leviathan.islandExecution_agent_code_generation_diversity import IslandExecutionDiversityMixin
from Leviathan.islandExecution_agent_code_generation_strategy import IslandExecutionStrategyMixin
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import openai
import json
from datetime import datetime
import traceback
import os

class IslandExecution(Island, IslandExecutionAnalysisMixin, IslandExecutionDiversityMixin, IslandExecutionStrategyMixin):
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

        # Diversity controller for adaptive exploration pressure
        self._diversity_controller = {
            "alpha": 0.25,
            "target_diversity": 0.45,
            "target_entropy": 0.6,
            "adjustment": 0.0,
            "min_adjust": -0.15,
            "max_adjust": 0.25,
        }

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
        population_state_summary = self._summarize_population_state()

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
            current_stats=current_stats,
            current_round_context=round_context,
            current_relation_context=relationship_context,
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

        Population state snapshot:
        {population_state_summary}

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
                "gini_wealth": round_context.get("gini_wealth"),
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
