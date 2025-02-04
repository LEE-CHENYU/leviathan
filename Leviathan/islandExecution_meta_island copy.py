from Leviathan.Island import Island
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
from collections import defaultdict
import inspect

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
            'survival_tracking': {},
            'modification_attempts': []
        }

        # Add message storage
        self.messages = {}  # {member_id: [list_of_messages]}
        
        # New: Allow agents to revise the prompt passed to them in future rounds.
        self.agent_prompt_revisions = {}

    def offer(self, member_1, member_2, parameter_influence):
        super()._offer(member_1, member_2, parameter_influence)
        
    def offer_land(self, member_1, member_2, parameter_influence):
        super()._offer_land(member_1, member_2, parameter_influence)
        
    def attack(self, member_1, member_2):
        super()._attack(member_1, member_2)

    def bear(self, member_1, member_2):
        super()._bear(member_1, member_2)
    
    def expand(self, member_1, member_2):
        super()._expand(member_1, member_2)

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

    def get_code_memory_summary(self, member_id):
        """Generate a summary of previous code performances for the agent."""
        if member_id not in self.code_memory:
            return "No previous code history."
            
        memory = self.code_memory[member_id]
        if not memory:
            return "No previous code history."
            
        summary = ["Previous code strategies and their outcomes:"]
        
        # Sort memories by performance
        sorted_memories = sorted(memory, key=lambda x: x['performance'], reverse=True)
        
        for i, mem in enumerate(sorted_memories[-5:]):  # Show last 5 memories
            summary.append(f"\nStrategy {i+1} (Performance: {mem['performance']:.2f}):")
            summary.append(f"Context: {mem['context']}")
            summary.append("Code:")
            summary.append(mem['code'])
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
            past_performance = f"Previous actions resulted in average performance change of {avg_perf:.2f}"

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
        message_context = "\n".join(received_messages) if received_messages else "No messages received"
        
        # New: Incorporate agent-specific prompt revision if available.
        revision_text = self.agent_prompt_revisions.get(member_id, "").strip()
        revision_section = f"\n[Agent Provided Prompt Revision]\n{revision_text}\n" if revision_text else ""

        # New: Get current game mechanisms and modification attempts
        current_mechanisms, modification_attempts = self.get_game_mechanisms_and_mods()
        
        constrainsAndExamples = """{part0}
            [Core Game Mechanics & Parameters]
            The island simulation has several key systems that agents should understand:

            1. Relationship System (_MIN_MAX_INIT_RELATION):
            - victim: [-50, 100] - Tracks damage received from others
            - benefit: [-50, 100] - Records resources received from others
            - benefit_land: [-3, 3] - Tracks land exchanges
            
            2. Population Mechanics:
            - _REPRODUCE_REQUIREMENT = 100 (Combined vitality/cargo needed)
            - Land must exceed population (land_shape[0] * land_shape[1] > population)
            - _NEIGHBOR_SEARCH_RANGE = 1000 for interaction radius
            
            3. Record Keeping (every _RECORD_PERIOD = 1):
            - Tracks all attacks, benefits, land transfers
            - Monitors births, deaths, land ownership
            - Records production, consumption, and performance metrics

            [Example Mechanism Extensions]
            def pre_init_hook(island):
                # New Social Systems
                island.add_mechanism('parliament', {
                    'voting_power': lambda member: member.land_num + member.vitality/100,
                    'proposals': [],
                    'voting_threshold': 0.66,
                    'term_length': 10
                })
                
                island.add_mechanism('market', {
                    'price_discovery': lambda resource, demand: demand * resource.scarcity,
                    'trade_fee': 0.05,
                    'order_book': defaultdict(list)
                })
                
                island.add_mechanism('justice', {
                    'jury_selection': lambda offense: random.sample(
                        [m for m in island.current_members if m.vitality > 50], 
                        min(5, len(island.current_members))
                    ),
                    'punishment_scale': lambda severity: severity * 10
                })

            def modify_member(member, relationships):
                # Add member capabilities
                member.add_ability('propose_law', 
                    lambda law: member.parliament.submit_proposal(law)
                    if member.voting_power > 10 else None
                )
                
                member.add_ability('trade', 
                    lambda resource, amount, price: member.market.place_order(
                        resource, amount, price, member.id
                    )
                )
                
                member.add_ability('file_complaint',
                    lambda offender, charge: member.justice.submit_case(
                        plaintiff=member.id,
                        defendant=offender,
                        charge=charge
                    ) if member.vitality > 30 else None
                )
                return member

            def modify_land(land, members):
                # Add land mechanics
                land.add_feature('zoning', {
                    'residential': 0.4,
                    'agricultural': 0.4,
                    'commercial': 0.2
                })
                
                land.add_feature('development', {
                    'infrastructure': np.zeros(land.shape),
                    'improvements': np.zeros(land.shape)
                })
                
                land.add_mechanism('property_tax',
                    lambda plot: sum(plot.improvements) * 0.1
                )
                
                land.add_mechanism('urban_planning',
                    lambda zone: zone.calculate_optimal_usage(
                        population=len(members),
                        resources=land.resources
                    )
                )
                return land

            def modify_relationships(relationships):
                # Add relationship mechanics
                relationships.add_metric('trust', 
                    lambda m1, m2: sum([
                        relationships['benefit'][m1.id][m2.id],
                        -relationships['victim'][m1.id][m2.id] * 2,
                        relationships['benefit_land'][m1.id][m2.id] * 5
                    ]) / 100
                )
                
                relationships.add_mechanism('alliance_formation',
                    lambda m1, m2: relationships.trust(m1, m2) > 0.7
                )
                
                relationships.add_mechanism('conflict_resolution',
                    lambda m1, m2: {
                        'mediator': max(members, 
                            key=lambda m: relationships.trust(m1, m) + relationships.trust(m2, m)
                        ),
                        'success_chance': relationships.trust(m1, m2) + 0.3
                    }
                )
                return relationships

            [Suggested Strategies]
            1. Form parliamentary coalitions to pass beneficial laws
            2. Use market mechanisms for efficient resource allocation
            3. Build trust through consistent positive interactions
            4. Leverage justice system to deter attacks
            5. Develop land strategically using zoning laws
            6. Create alliance networks based on trust metrics"""
            
        # Move part2 to a separate mechanism section
        mechanism_section = f"""
        {constrainsAndExamples}
        
        [Current Game Version] 
        {self._VERSION}
        
        [Active Game Mechanisms]
        {current_mechanisms}
        
        [Modification Attempt History]
        {modification_attempts}
        
        [Modification Proposal Guide]
        To propose rule changes, follow this template:

        1. ANALYSIS PHASE:
           - Identify limitation in current systems
           - Check mechanism compatibility using get_game_mechanisms_and_mods()
           - Review past modification attempts for patterns

        2. PROPOSAL PHASE:
        def agent_action(execution_engine, member_id):
            # Example valid modification:
            modification_code = '''
            # Add judicial system
            if not hasattr(execution_engine, 'court'):
                class CourtSystem:
                    MECHANISM_META = {{
                        'type': 'Governance',
                        'rules': 'Handles conflict resolution through jury trials',
                        'version': 1.0
                    }}
                    def __init__(self):
                        self.cases = []
                        
                    def submit_case(self, plaintiff, defendant, charge):
                        self.cases.append({{
                            'plaintiff': plaintiff,
                            'defendant': defendant,
                            'charge': charge,
                            'status': 'pending'
                        }})
            
            execution_engine.court = CourtSystem()
            '''
            
            # Proposal with ratification conditions
            execution_engine.propose_modification(
                member_id=member_id,
                mod_type='post_init',
                code=modification_code,
                ratification_condition={{
                    'type': 'compound',
                    'conditions': [
                        {{'type': 'vote', 'threshold': 0.6}},
                        {{'type': 'resource', 'resource': 'cargo', 'amount': 100}},
                        {{'type': 'time', 'rounds': 2}}
                    ]
                }}
            )

        [Valid Ratification Conditions]
        Choose ONE primary condition + optional safeguards:
        1. Voting: {{'type': 'vote', 'threshold': 0.5-1.0}}
        2. Resource: {{'type': 'resource', 'resource': 'vitality|cargo|land', 'amount': X}}
        3. Temporal: {{'type': 'time', 'rounds': N}} (auto-ratify after N rounds)
        4. Achievement: {{'type': 'achievement', 'metric': 'productivity', 'threshold': X}}

        [Common Errors to Avoid]
        1. Namespace Conflicts: Check existing mechanisms with dir(execution_engine)
        2. Invalid References: Use execution_engine.current_members not global members
        3. Version Mismatches: Increment version when modifying existing systems
        4. Resource Leaks: Include cleanup functions for new mechanisms

        [Best Practices]
        - Propose small, testable changes first
        - Include rollback procedures in code
        - Add version checks to modifications
        - Use execution_engine.add_mechanism() helper
        - Test modifications with 'sandbox=True' parameter
        """

        try:
            # Define iterative prompt parts with specific constraints
            part0 = """
            [Previous code execution context]
        {error_context}

        [Current task]
        You are member_{member.id} in a society that you can help shape.
        Write a Python function named agent_action(execution_engine, member_id) 
        that implements your vision of social organization while ensuring your survival.

        [Critical constraints]
        - Carefully analyze previous errors shown above and avoid repeating them
        - Never target yourself (member_{member.id}) in any action
        - Verify member has land before using bear() action
        - Check member IDs exist before referencing them
        - Ensure all matrix indices are valid
        - current_members is a LIST accessed by index, not a dictionary
        - Access members using execution_engine.current_members[index]
        - Check if index exists: if index < len(execution_engine.current_members)
        
        IMPORTANT: Here are the attributes and methods actually available:

        1) Each member object has:
            • member.id (int): The unique ID for the member
            • member.vitality (float)
            • member.cargo (float)
            • member.overall_productivity (float)
            • member.age (float)
            • member.current_clear_list (List[int]) - IDs of neighbors or cleared adjacents
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
            • execution_engine.expand(member1, member2) - Expands territory
        5) The members are accessed by execution_engine.current_members[id].
            For example, execution_engine.current_members[2] for the member with ID=2.
        6) DO NOT reference 'member.member_id' or 'member.self_vitality'. Use member.id, member.vitality, etc.

        Current status (features of all members):
        {features}

        Relationship summary (parsed from relationship_dict):
        {relations}

        Code Memory and Previous Performance:
        {code_memory}

        Performance history:
        {past_performance}

        Based on the previous code performance, adapt and improve the strategy.
        If a previous strategy worked well (high performance), consider building upon it.
        If it failed, try a different approach.
        
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
        execution_engine.send_message(your_id, recipient_id, "message")
        Example usage:
        - Broadcast to all: 
          for recipient in range(len(execution_engine.current_members)):
              if recipient != your_id:
                  execution_engine.send_message(your_id, recipient, "Let's cooperate!")
        - Message allies:
          for ally_id in ally_list:
              execution_engine.send_message(your_id, ally_id, "Attack target X")
        - Group coordination:
          for member_id in coalition:
              execution_engine.send_message(your_id, member_id, "Vote YES on proposal")

        [Received Messages]
        {message_context}
            """
            part2 = mechanism_section  # Replace old part2 with new mechanism section
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

        round_num = len(self.execution_history['rounds'])
        round_data = {
            'timestamp': datetime.now().isoformat(),
            'actions': [],
            'performance_changes': {},
            'survival_changes': {},
            'messages': {}  # Add message tracking
        }

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
                'survival_chance': old_survival
            }

            error_occurred = None
            try:
                # Track messages for this member
                messages_sent = []
                # Get received messages for this member BEFORE clearing
                received_messages = self.messages.get(member_id, [])
                
                # Modified exec environment with message tracking
                def tracked_send_message(sender, recipient, msg):
                    nonlocal messages_sent
                    messages_sent.append((recipient, msg))
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
                'survival_chance': new_survival
            }
            
            performance_change = new_survival - old_survival

            # Store in code memory
            if member_id not in self.code_memory:
                self.code_memory[member_id] = []
                
            memory_entry = {
                'code': code_str,
                'performance': performance_change,
                'context': {
                    'old_stats': old_stats,
                    'new_stats': new_stats,
                    'round': round_num
                }
            }
            if error_occurred:
                memory_entry['error'] = error_occurred
                
            self.code_memory[member_id].append(memory_entry)

            # Store performance in history
            if member_id not in self.performance_history:
                self.performance_history[member_id] = []
            self.performance_history[member_id].append(performance_change)

            # Log changes for this round
            round_data['actions'].append({
                'member_id': member_id,
                'code_executed': code_str,
                'old_stats': old_stats,
                'new_stats': new_stats,
                'performance_change': performance_change,
                'messages_sent': messages_sent
            })

            # Log messages in round data
            round_data['messages'][member_id] = {
                'received': received_messages,
                'sent': messages_sent
            }

        self.execution_history['rounds'].append(round_data)
        
        # Save execution history after each round
        self.save_execution_history()
        
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

    def get_game_mechanisms_and_mods(self):
        """Return formatted current mechanisms and modification attempts"""
        # Dynamically detect all active mechanisms using introspection
        current_mechanisms = []
        # Check for any registered mechanisms with a standard interface
        for name, obj in inspect.getmembers(self):
            if name.startswith('_'):  # Skip private attributes
                continue
            if hasattr(obj, 'mechanism_meta'):
                current_mechanisms.append(
                    f"- {name}: {obj.mechanism_meta['type']} system\n"
                    f"  Rules: {obj.mechanism_meta['rules']}\n"
                    f"  Version: {obj.mechanism_meta.get('version', 1.0)}"
                )
            elif isinstance(obj, dict) and 'mechanism_type' in obj:
                current_mechanisms.append(
                    f"- {name}: {obj.get('description', 'Custom mechanism')}\n"
                    f"  Type: {obj['mechanism_type']}\n"
                    f"  Last modified: {obj.get('modified_round', 'N/A')}"
                )

        # Format modification attempts with versioning
        mod_entries = []
        for attempt in self.execution_history.get('modification_attempts', []):
            status = "Pending" if not attempt.get('ratified', False) else "Active"
            conditions = attempt.get('ratification_condition', {})
            condition_desc = ", ".join([f"{k}={v}" for k,v in conditions.items()])
            
            mod_entries.append(
                f"Proposed by Member {attempt['member_id']} ({status})\n"
                f"Type: {attempt['mod_type']}\n"
                f"Condition: {condition_desc}\n"
                f"Code Preview: {attempt['code'][:50]}..."
            )

        return "\n".join(current_mechanisms), "\n\n".join(mod_entries)

    def propose_modification(self, member_id, mod_type, code, ratification_condition):
        """Track modification attempts"""
        self.execution_history['modification_attempts'].append({
            'member_id': member_id,
            'mod_type': mod_type,
            'code': code,
            'ratification_condition': ratification_condition,
            'ratified': False,
            'votes': defaultdict(int),
            'resource_commitments': defaultdict(float)
        })

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