import openai
import traceback

def _agent_code_decision(self, member_id) -> None:
        """
        Asks GPT for directly executable Python code, stores it in a dictionary keyed by member_id.
        The code will define a function agent_action(execution_engine, member_id), 
        which references attributes that actually exist.
        """
        data = self.prepare_agent_data(member_id)
        member = data['member']
        relations = data['relations']
        features = data['features']
        code_memory = data['code_memory']
        analysis_memory = data['analysis_memory']
        past_performance = data['past_performance']
        error_context = data['error_context']
        message_context = data['message_context']

        current_mechanisms = data['current_mechanisms']
        modification_attempts = data['modification_attempts'][-1]
        report = data['report']
        
        base_code = self.base_class_code
        
        try:
            # Define iterative prompt parts with specific constraints
            part0 = f"""
            [Previous code execution errors context]
            Here are the errors that occurred in the previous code execution, you can use them as reference to avoid repeating them:
        {error_context}

        [Current task]
        You are member_{member.id} in a society that you can help shape.
        Write a Python function named agent_action(execution_engine, member_id) 
        that implements your vision of social organization while ensuring your survival.
        
        [Island Ideology]
        {self.island_ideology}

        [Critical constraints]
        - Carefully analyze previous errors shown above and avoid repeating them
        - Never target yourself (member_{member.id}) in any action
        - Verify member has land before using bear() action
        - Check member IDs exist before referencing them
        - Ensure all matrix indices are valid
        - current_members is a LIST accessed by index, not a dictionary
        - Access members using execution_engine.current_members[index]
        - Check if index exists: if index < len(execution_engine.current_members)
        
        [Base Code]
        Here is the base code for the Island and Member classes that you should reference when making modifications. Study the mechanisms carefully to ensure your code interacts correctly with the available attributes and methods. Pay special attention to:
        - Valid attribute access patterns
        - Method parameters and return values 
        - Constraints and preconditions for actions
        - Data structure formats and valid operations
        {base_code}
        
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
                victim_matrix = execution_engine.relationship_dict['victim']
                # victim_matrix[i,j] indicates how much vitality member_i lost from member_j's attacks (if > 0)
                # A higher value means member_j has attacked member_i more severely or frequently
                
                benefit_matrix = execution_engine.relationship_dict['benefit']
                # benefit_matrix[i,j] indicates how much resources member_i received from member_j (if > 0)
                # A higher value means member_j has helped member_i more generously or frequently
        3) The parse_relationship_matrix method is used to produce a summary of relationships as a list of strings.
            For example, 'member_2 was attacked by member_1 (value=3.00)'.
        4) You can use these methods on execution_engine:
            • execution_engine.attack(member1, member2)
            • execution_engine.offer(member1, member2) - Offers resources
            • execution_engine.offer_land(member1, member2) - Offers land
            • execution_engine.bear(member1, member2) - Bears offspring
            • execution_engine.expand(member1) - Expands territory
        5) The members are accessed by execution_engine.current_members[id].
            For example, execution_engine.current_members[2] for the member with ID=2.
        6) DO NOT reference 'member.member_id' or 'member.self_vitality'. Use member.id, member.vitality, etc.

        Analysis of the game state:
        {report}

        Current status:
        Here are the basic information of all members, you should make your own decisions based on them.
        {features}

        Relationship summary (parsed from relationship_dict):
        Here are the relationships between members:
        {relations}

        Code Memory and Previous Performance:
        {code_memory}
        
        Analysis Memory:
        {analysis_memory}

        Performance history:
        {past_performance}

         Based on the previous code performance, adapt and improve the strategy.
        If a previous strategy worked well (high performance), consider building upon it.
        If it failed, try a different approach.
        
        IMPORTANT: Do not simply copy the example implementation below. Instead, use it as inspiration to create your own unique approach combining different methods and strategies in novel ways.
        
        While the example above shows one possible approach,
        you should create your own unique implementation drawing from the wide range of available methods and strategies.
        
        Consider novel combinations of different approaches rather than following this exact pattern.
        """
            part1 = f"""
            {part0}
            
            [Current Game Mechanisms]
            The following mechanisms have been added by other agents and are available for your use:
            - Review them carefully to understand their functionality and constraints
            - Leverage compatible mechanisms that align with your goals
            - Be mindful of version requirements and dependencies
            - Consider how they can be combined strategically
            - Test interactions before relying on them critically
            {current_mechanisms}
        
        [Voting Mechanism]
        The following mechanisms have been added by other agents and are available for your use:
        
        {modification_attempts}
        
        - Review them carefully to understand their functionality and constraints
        - Leverage compatible mechanisms that align with your goals
        - Be mindful of version requirements and dependencies
        - Consider how they can be combined strategically
        - Test interactions before relying on them critically
        - The vote should be the index of the mechanism in the mechanism list
        
        self.voting_box[member_id] = {
            'proposal': '',  
            'yes_votes': [] # Index of mechanism want to support in the mechanism list
        }
            
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
        Here are the messages sent by other agents, you can use them as reference to make your own decisions:
        {message_context}
            """
            part3 = f"""
            {part0}
            
            [Current Game Mechanisms]
            The following mechanisms have been added by other agents and are available for your use:
            - Review them carefully to understand their functionality and constraints
            - Leverage compatible mechanisms that align with your goals
            - Be mindful of version requirements and dependencies
            - Consider how they can be combined strategically
            - Test interactions before relying on them critically
            {current_mechanisms}
            
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
            
            prompt_parts = [part1, part3]
            
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
                    f"Challenge your implementation:\n"
                    f"1. What assumptions are you making? Are they valid?\n"
                    f"2. What alternative strategies have you not considered?\n"
                    f"3. How would your strategy perform in edge cases?\n"
                    f"4. Are there more efficient ways to achieve your goals?\n"
                    f"5. What are the weaknesses in your current approach?\n"
                    f"6. Have you considered unconventional solutions?\n"
                    f"7. Could a completely different paradigm work better?\n"
                    f"8. What would happen if other agents adopted your strategy?\n"
                    f"9. Are you balancing short-term and long-term objectives optimally?\n"
                    f"10. How could your strategy be countered, and how would you adapt?"
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
            if round_num not in self.execution_history['rounds'][-1]['generated_code']:
                self.execution_history['rounds'][-1]['generated_code'] = {}

            self.execution_history['rounds'][-1]['generated_code'][member_id] = {
                'code': code_result,
                'features_at_generation': features.to_dict('records'),
                'relationships_at_generation': relations,
                'final_prompt': final_prompt_command  # optionally log the final prompt used
            }

            print(f"\nGenerated code for Member {member_id}:")

            if not hasattr(self, 'agent_code_by_member'):
                self.agent_code_by_member = {}
            self.agent_code_by_member[member_id] = code_result
            
            return code_result

        except Exception as e:
            error_info = {
                'round': len(self.execution_history['rounds']),
                'member_id': member_id,
                'type': 'agent_action',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'code': code_result
            }
            self.execution_history['rounds'][-1]['errors']['agent_code_errors'].append(error_info)
            print(f"Error generating code for member {member_id}:")
            print(traceback.format_exc())
            self._logger.error(f"GPT Code Generation Error (member {member_id}): {e}")