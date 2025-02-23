import numpy as np
import openai
import traceback

def _analyze(self, member_id):
    """
    Analyze member data for strategic insights
    """
    
    member = self.current_members[member_id]
    
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
    modification_attempts = data['modification_attempts'][max(data['modification_attempts'].keys())]
    report = data['report']
    
    base_code = self.base_class_code
    
    """Analyze member data for strategic insights"""
    previous_errors = self.execution_history['rounds'][-1]['errors']['analyze_code_errors'].get(member_id, {})
    analysis_prompt = f"""
    {previous_errors}
    
    [Current task]
    You are member_{member.id} in a society that you can help shape. Use plain text to describe your situation and propose strategies. All your words should be based on concrete facts and data. Strategies should be practical and effective, if you can't find any effective strategies, just say so. If your analysis and strategies are not fact-oriented, you will not be competitive and you will die.
    
    [Base Code]
    {base_code}

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
    
    [Current Mechanisms]
    {current_mechanisms}
    
    Previous Analysis of the game state:
    {report} 
    
    [Data-Driven Survival Framework]
    Collect and analyze ALL available game variables:
    
    [Core Member Metrics]
    - Vitality: [m.vitality for m in execution_engine.current_members]
    - Cargo: [m.cargo for m in execution_engine.current_members]
    - Land_num: [m.land_num for m in execution_engine.current_members]
    - Age: [m.age for m in execution_engine.current_members]
    - Surviver_id: [m.surviver_id for m in execution_engine.current_members]
    - Owned_land: [m.owned_land for m in execution_engine.current_members]
    - Parameters: attack_decision=[getattr(m, 'attack_decision', 0.5) for m in execution_engine.current_members], 
        offer_decision=[getattr(m, 'offer_decision', 0.5) for m in execution_engine.current_members]
    
    [Social Dynamics]
    - Relationship matrices: victim=np.array(execution_engine.relationship_dict['victim']),
        benefit=np.array(execution_engine.relationship_dict['benefit']),
        benefit_land=np.array(execution_engine.relationship_dict['benefit_land'])
    - Action logs: attacks=execution_engine.record_action_dict['attack'],
        benefits=execution_engine.record_action_dict['benefit'],
        land_transfers=execution_engine.record_action_dict['benefit_land']
    
    [Territory Analysis] 
    - Land ownership map: execution_engine.land.owner_id
    - Land shape: execution_engine.land.shape
    - Strategic positions: [m.center_of_land(execution_engine.land) for m in execution_engine.current_members]
    
    [Economic Indicators]
    - Total production: execution_engine.record_total_production
    - Total consumption: execution_engine.record_total_consumption
    - Resource flow rates: [m.cargo/max(m.vitality,1) for m in execution_engine.current_members]
    
    [Evolutionary Trends]
    - Birth/death rates: float(len(execution_engine.record_born))/max(len(execution_engine.record_death), 1)
    - Parameter drift: [getattr(m, '_parameter_drift', 0.0) for m in execution_engine.current_members]
    """
    
    implementation_prompt = """
    
    Write a Python function named analyze_game_state(execution_engine, member_id) to analyze the game state and propose strategic actions.]
    
    Example Implementation:
    IMPORTANT: Do not simply copy the example implementation. Instead, use it as inspiration to create your own unique approach combining different methods and strategies in novel ways.
    
    [Analysis Protocol]
    1. Calculate resource inequality using Gini coefficient on cargo+land
    2. Perform network analysis on benefit relationships to identify power brokers
    3. Compute territory clustering coefficients
    4. Track parameter evolution vs survival outcomes
    5. Model vitality-cargo-land survival probability surface
    
    def analyze_game_state(execution_engine, member_id):
        me = execution_engine.current_members[member_id]
        
        # Collect raw data
        data = {{ 
            'members': execution_engine.current_members,
            'relationships': {{k: np.array(v) if isinstance(v, list) else v 
                                for k,v in execution_engine.relationship_dict.items()}},
            'land': execution_engine.land,
            'economics': (execution_engine.record_total_production, 
                            execution_engine.record_total_consumption),
            'actions': execution_engine.record_action_dict
        }}
        
        # Perform analysis
        ## Try not to copy the example implementation below.
        analysis = {{
            'gini': float(np.std([m.cargo + m.land_num*10 for m in data['members']]))/float(np.mean([m.cargo + m.land_num*10 for m in data['members']])),
            'central_nodes': [m.id for m in data['members'] if float(np.nanmean(data['relationships']['benefit'][m.surviver_id])) > float(np.nanpercentile(data['relationships']['benefit'], 75))],
            'cluster_density': float(len(me.owned_land))/float(execution_engine.land.shape[0]**2),
            'optimal_vitality': float(max(50, np.percentile([m.vitality for m in data['members']], 75))),
            'survival_prob': float(me.vitality)/(float(me.vitality) + float(me.cargo) + 1.0)
        }}
        
        # Generate report
        ## Try not to copy the example implementation below.
        report = {{
            **analysis,
            'top_holders': sorted(data['members'], key=lambda x: x.cargo, reverse=True)[:3],
            'rich_areas': [loc for loc in me.owned_land],  # Removed resource_yield check
            'rec_action1': "Secure defensive alliances" if analysis['survival_prob'] < 0.5 else "Expand territory",
            'rec_action2': "Diversify resource streams" if analysis['gini'] > 0.4 else "Consolidate assets",
            'rec_action3': "Optimize parameters"
        }}
        
        # Store in member memory
        if not hasattr(me, 'data_memory'):
            me.data_memory = {{'reports': [], 'strategies': []}}
        me.data_memory['reports'].append(report)
        
        print(f"\nStrategic Analysis Report:\n{{report}}")
        return analysis  
        
    If a previous strategy worked well (high performance), consider building upon it.
    If it failed, try a different approach.
    
    Return only the code, no extra text or explanation. While the example above shows one possible approach,
    you should create your own unique implementation drawing from the wide range of available methods and strategies.
    
    Consider novel combinations of different approaches rather than following this exact pattern.
        """
        
    # # Iteratively build the final prompt from the parts
    # final_prompt = ""
    # update_message = (
    #     f"Current prompt:\n{analysis_prompt}\n"
    #     f"Ensuring all previous constraints are preserved and adding these new constraints.\n\n"
    #     f"Return the updated full prompt, emphasizing that agents should implement solutions "
    #     f"according to their individual needs, beliefs, and circumstances.\n\n"
    #     f"Additionally, analyze the game mechanics to understand:\n"
    #     f"1. The core objective - Is it pure survival, domination, or cooperation?\n" 
    #     f"2. Key success metrics - What truly determines 'winning'?\n"
    #     f"3. Potential improvements - What mechanics could be added/modified?\n\n"
    #     f"Challenge your implementation:\n"
    #     f"1. What assumptions are you making? Are they valid?\n"
    #     f"2. What alternative strategies have you not considered?\n"
    #     f"3. How would your strategy perform in edge cases?\n"
    #     f"4. Are there more efficient ways to achieve your goals?\n"
    #     f"5. What are the weaknesses in your current approach?\n"
    #     f"6. Have you considered unconventional solutions?\n"
    #     f"7. Could a completely different paradigm work better?\n"
    #     f"8. What would happen if other agents adopted your strategy?\n"
    #     f"9. Are you balancing short-term and long-term objectives optimally?\n"
    #     f"10. How could your strategy be countered, and how would you adapt?"
    # )
    # completion = openai.chat.completions.create(
    #     model="o3-mini", 
    #     messages=[{"role": "user", "content": update_message}]
    # )
    # final_prompt = completion.choices[0].message.content.strip()
    
    # # Append a final instruction to generate the code function
    # final_prompt_command = final_prompt + "\n\nUsing the above comprehensive prompt with all integrated constraints, produce a unique implementation that reflects your individual needs, beliefs and circumstances. The implementation should be tailored to your specific situation rather than following a generic template. Your code should demonstrate a deep understanding of the game mechanics and implement sophisticated strategies to achieve both survival and prosperity. Consider both immediate tactical actions and long-term strategic planning, as well as how to effectively interact with other symmetric agents to achieve both individual and collective goals. Return only the code."
            
    completion = openai.chat.completions.create(
                model="o3-mini", 
                messages=[{"role": "user", "content": analysis_prompt}]
            )
    result = completion.choices[0].message.content.strip()
    # analysis_code = self.clean_code_string(analysis_code)
    
    # print(f"\nStrategic Analysis Code:\n{analysis_code}")
    
    # Execute the code in a new environment
    exec_env = {}
    exec_env['execution_engine'] = self
    exec_env['member_id'] = member_id
    exec_env['me'] = self.current_members[member_id]
    exec_env['np'] = np  # Make numpy available in the environment

    if member_id not in self.analysis_reports:
        self.analysis_reports[member_id] = []
        self.analysis_reports[member_id].append(result)
    else:
        print("No analysis result returned")
        return None
    
    # try:
    #     # First execute the code to define the functions
    #     exec(analysis_code, exec_env)
        
    #     # Then call the analysis function
    #     result = exec_env['analyze_game_state'](self, member_id)
        
    #     if result:
    #         print(f"Analysis result: {result}")
    #         if member_id not in self.analysis_reports:
    #             self.analysis_reports[member_id] = []
    #         self.analysis_reports[member_id].append(result)
    #     else:
    #         print("No analysis result returned")
    #         return None
        
    # except Exception as e:
    #     print(f"Error executing analysis code: {e}")
    #     print(f"Traceback:\n{traceback.format_exc()}")
    #     print(f"Analysis code that failed:\n{analysis_code}")
    #     print(f"Member ID: {member_id}")
    #     print(f"Current round: {len(self.execution_history['rounds'])}")
    #     self.execution_history['rounds'][-1]['errors']['analyze_code_errors'][member_id] = {
    #         'round': len(self.execution_history['rounds']),
    #         'member_id': member_id,
    #         'type': 'analyze_game_state',
    #         'error': str(e),
    #         'code': analysis_code
    #     }
    #     return None