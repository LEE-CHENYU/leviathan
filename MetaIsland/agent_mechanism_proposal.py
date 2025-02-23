from itertools import tee
import openai
import traceback
import ast

def _agent_mechanism_proposal(self, member_id) -> None:
    """ 
    Asks GPT for directly executable Python code, stores it in a dictionary keyed by member_id.
    The code will define a function propose_modification(execution_engine), 
    which references attributes that actually exist.
    """
    # Prepare data for the proposal
    data = self.prepare_agent_data(member_id)
    
    # Use the prepared data
    member = data['member']
    relations = data['relations']
    features = data['features']
    code_memory = data['code_memory']
    analysis_memory = data['analysis_memory']
    past_performance = data['past_performance']
    error_context = data['error_context']
    message_context = data['message_context']
    
    current_mechanisms = data['current_mechanisms']
    modification_attempts = []
    for round_num in data['modification_attempts'].keys():
        round_attempts = [attempt for attempt in data['modification_attempts'][round_num] if attempt.get('member_id') == member_id]
        modification_attempts.extend(round_attempts)
    report = data['report']
    
    base_code = self.base_class_code
    
    base_code = f"""
            [Base Code]
            Here is the base code for the Island and Member classes that you should reference when making modifications. Study the mechanisms carefully to ensure your code interacts correctly with the available attributes and methods. Pay special attention to:
            - Valid attribute access patterns
            - Method parameters and return values 
            - Constraints and preconditions for actions
            - Data structure formats and valid operations
            {base_code}
            """
    part0 = f"""
        [Previous code execution errors context]
        Here are the errors that occurred in the previous code execution, you can use them as reference to avoid repeating them:
    {error_context}
    
    [Current Task]
    You are member in a society that you can help shape.
    Write a Python function named propose_modification(execution_engine) 
    that implements your vision of social organization while ensuring your survival.
    
    [Island Ideology]
    {self.island_ideology}
    
    [Critical constraints]
    - Carefully analyze previous errors shown above and inspect._void repeating them
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

    Analysis of the game state:
    {report}
    
    Current status:
    Here are the basic information of all members, you should make your own decisions based on them:
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

    Based on the previous code performance, propose a modification to the game mechanics.
    If a previous proposal worked well (high performance), consider building upon it.
    If it failed, try a different approach.
    
    IMPORTANT: Do not simply copy the example implementation below. Instead, use it as inspiration to create your own unique approach combining different methods and strategies in novel ways.
    
    While the example above shows one possible approach,
    you should create your own unique implementation drawing from the wide range of available methods and strategies.
    
    Consider novel combinations of different approaches rather than following this exact pattern.
    """
    constrainsAndExamples = f"""
    {part0}
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
    # Basic mechanism template
    if not hasattr(island, 'new_system'):
        class CustomMechanism:
            def __init__(self):
                self.data = {{}}
                self.meta = {{'version': 1.0, 'type': 'custom'}}
        
        island.new_system = CustomMechanism()

    # Member capability example
    def modify_member(member, relationships):
        if not hasattr(member, 'custom_abilities'):
            member.custom_abilities = {{}}
        
        # Add trading capability
        member.custom_abilities['trade'] = lambda resource: (
            print(f"Trading {{resource}}") if member.vitality > 20 else None
        )
        return member

    # Land modification example  
    def modify_land(land, members):
        if not hasattr(land, 'zoning'):
            land.zoning = {{
                'residential': 0.4,
                'agricultural': 0.4,
                'commercial': 0.2
            }}
        
        # Add development tracking
        land.development_level = np.zeros(land.shape)
        return land

    # Relationship system extension
    def modify_relationships(relationships):
        relationships.trust_matrix = np.zeros_like(relationships['benefit'])
        return relationships

    [Implementation Patterns]
    1. Check existence first: if not hasattr(obj, 'feature')
    2. Add attributes directly: obj.new_feature = ...
    3. Use simple data structures: dicts, lists, numpy arrays
    4. Include version metadata in new systems
    5. Add cleanup methods for complex systems:

    [Error Prevention]
    - Use try-except when accessing new features
    - Check attribute existence before use
    - Maintain backward compatibility
    - Use version checks for existing systems:

    if (hasattr(island, 'market') and 
        getattr(island.market, 'version', 0) < 2):
        # Add compatibility layer
        island.market.legacy_support = True
    """
    
    mechanism_section = f"""
    {constrainsAndExamples}

    [Active Game Mechanisms]
    The following mechanisms have been added by agents and can be referenced when making your own modifications. Review them carefully to:
    1. Understand existing functionality and avoid conflicts
    2. Build upon successful patterns and improvements
    3. Identify opportunities for optimization or extension
    4. Remove or deprecate mechanisms that are detrimental to your survival
    
    When proposing changes, ensure they:
    - Align with your agent's goals and survival strategy
    - Maintain compatibility with other active mechanisms
    - Include proper versioning and rollback procedures
    - Follow best practices for stability and performance
    {current_mechanisms}
    
    [Modification Attempt History]
    [Previous Modification History]
    Review your past modification attempts below to inform future proposals:
    - Learn from successful patterns and approaches
    - Avoid repeating failed strategies
    - Build upon and extend working mechanisms
    - Identify opportunities for optimization
    {modification_attempts}
    
    [Message Context]
    Here are the messages sent by other agents, you can use them as reference to make your own decisions:
    {message_context}
    
    [Modification Proposal Guide]
    To propose rule changes, follow this template:
    
    1. ANALYSIS PHASE:
        - Identify limitation in current systems
        - Review past modification attempts for patterns
    
    2. PROPOSAL PHASE:
    # This method will be added to the IslandExecution class
    def propose_modification(self):
        \"""
        Include clear reasoning for each modification to help other agents
        understand tee intended benefits and evaluate the proposal.
        \"""
        # Example modification:
        if not hasattr(self, 'mechanism'):
            class Mechanism:
                MECHANISM_META = {{
                    'type': 'Basic',
                    'rules': 'Generic mechanism template',
                    'version': 1.0
                }}
                def __init__(self):
                    self.data = []
                    
                def add_data(self, data):
                    self.data.append(data)
        
        self.mechanism = Mechanism()
    
    [Common Errors to Avoid]
    1. Namespace Conflicts: Check existing mechanisms with dir(execution_engine)
    2. Invalid References: Use execution_engine.current_members not global members
    3. Version Mismatches: Increment version when modifying existing systems
    4. Resource Leaks: Include cleanup functions for new mechanisms
    
    [Best Practices]
    - Propose small, testable changes first
    - Include rollback procedures in code
    - Add version checks to modifications
    """
    
    def make_class_picklable(class_name, class_dict):
        """Make dynamically created classes picklable by adding them to globals"""
        # Create the class and add it to the global namespace
        new_class = type(class_name, (), class_dict)
        globals()[class_name] = new_class
        return new_class

    try:
        # Define iterative prompt parts with specific constraints
        
        prompt_parts = [mechanism_section]
        
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
        final_prompt_command = final_prompt + "\n\nUsing the above comprehensive prompt with all integrated constraints, produce a unique propose_modification() implementation that reflects your individual needs, beliefs and circumstances. The implementation should be tailored to your specific situation rather than following a generic template. Your code should demonstrate a deep understanding of the game mechanics and implement sophisticated methods to achieve both survival and prosperity. Consider both immediate tactical actions and long-term strategic planning, as well as how to effectively allow other symmetric agents interact with each other to achieve both individual and collective goals. Return only the code."
        
        completion = openai.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": final_prompt_command}]
        )
        code_result = completion.choices[0].message.content.strip()

        # Clean and store the code
        code_result = self.clean_code_string(code_result)

        # Log the generated code
        round_num = len(self.execution_history['rounds'])
        mod_proposal = {
            'round': round_num,
            'member_id': member_id,
            'code': code_result,
            'features_at_generation': features.to_dict('records'),
            'relationships_at_generation': relations,
            'final_prompt': final_prompt_command,
            'ratified': False
        }
        self.execution_history['rounds'][-1]['mechanism_modifications']['attempts'].append(mod_proposal)

        print(f"\nGenerated code for Member {member_id}:")

        # Extract class definitions from the code
        tree = ast.parse(code_result)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Make the class picklable before it's instantiated
                class_dict = {'__module__': '__main__'}  # Set the module to __main__
                make_class_picklable(node.name, class_dict)
                
        return code_result
        
    except Exception as e:
        error_info = {
            'round': len(self.execution_history['rounds']),
            'member_id': member_id,
            'type': 'propose_modification',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'code': code_result
        }
        self.execution_history['rounds'][-1]['errors']['mechanism_errors'].append(error_info)
        print(f"Error generating code for member {member_id}:")
        print(traceback.format_exc())
        self._logger.error(f"GPT Code Generation Error (member {member_id}): {e}")