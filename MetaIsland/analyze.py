import numpy as np
import openai
import traceback

def _analyze(self, member_id):
    """
    Analyze member data for strategic insights
    """
    
    member = self.current_members[member_id]
    
    """Analyze member data for strategic insights"""
    previous_errors = self.execution_history['analysis_code_errors'].get(member_id, {})
    analysis_prompt = f"""
    {previous_errors}
    
    [Current task]
    [You are member_{member.id} in a society that you can help shape.
    Write a Python function named analyze_game_state(execution_engine, member_id)
    to analyze the game state and propose strategic actions.]
    
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
    - Color inheritance: [getattr(m, '_current_color', None) for m in execution_engine.current_members]

    [Analysis Protocol]
    1. Calculate resource inequality using Gini coefficient on cargo+land
    2. Perform network analysis on benefit relationships to identify power brokers
    3. Compute territory clustering coefficients
    4. Track parameter evolution vs survival outcomes
    5. Model vitality-cargo-land survival probability surface
    
    Example Implementation:
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
        analysis = {{
            'gini': float(np.std([m.cargo + m.land_num*10 for m in data['members']]))/float(np.mean([m.cargo + m.land_num*10 for m in data['members']])),
            'central_nodes': [m.id for m in data['members'] if float(np.nanmean(data['relationships']['benefit'][m.surviver_id])) > float(np.nanpercentile(data['relationships']['benefit'], 75))],
            'cluster_density': float(len(me.owned_land))/float(execution_engine.land.shape[0]**2),
            'optimal_vitality': float(max(50, np.percentile([m.vitality for m in data['members']], 75))),
            'survival_prob': float(me.vitality)/(float(me.vitality) + float(me.cargo) + 1.0)
        }}
        
        # Generate report
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
        
        # Update strategy parameters
        if hasattr(me, 'attack_decision'):
            me.attack_decision = float(me.attack_decision) * (1.0 + float(analysis['survival_prob'])/10.0)
        if hasattr(me, 'offer_decision'):
            me.offer_decision = float(me.offer_decision) * (1.0 - float(analysis['gini'])/5.0)
        
        print(f"\nStrategic Analysis Report:\n{{report}}")
        return analysis  
        
    If a previous strategy worked well (high performance), consider building upon it.
    If it failed, try a different approach.
    
    IMPORTANT: Do not simply copy the example implementation below. Instead, use it as inspiration to create your own unique approach combining different methods and strategies in novel ways.
    
    Return only the code, no extra text or explanation. While the example above shows one possible approach,
    you should create your own unique implementation drawing from the wide range of available methods and strategies.
    
    Consider novel combinations of different approaches rather than following this exact pattern.
        """
    completion = openai.chat.completions.create(
                model="o3-mini", 
                messages=[{"role": "user", "content": analysis_prompt}]
            )
    analysis_code = completion.choices[0].message.content.strip()
    analysis_code = self.clean_code_string(analysis_code)
    
    # print(f"\nStrategic Analysis Code:\n{analysis_code}")
    
    # Execute the code in a new environment
    exec_env = {}
    exec_env['execution_engine'] = self
    exec_env['member_id'] = member_id
    exec_env['me'] = self.current_members[member_id]
    exec_env['np'] = np  # Make numpy available in the environment
    
    try:
        # First execute the code to define the functions
        exec(analysis_code, exec_env)
        
        # Then call the analysis function
        result = exec_env['analyze_game_state'](self, member_id)
        
        if result:
            print(f"Analysis result: {result}")
            if member_id not in self.analysis_reports:
                self.analysis_reports[member_id] = []
            self.analysis_reports[member_id].append(result)
        else:
            print("No analysis result returned")
            return None
        
    except Exception as e:
        print(f"Error executing analysis code: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        print(f"Analysis code that failed:\n{analysis_code}")
        print(f"Member ID: {member_id}")
        print(f"Current round: {len(self.execution_history['rounds'])}")
        self.execution_history['analysis_code_errors'][member_id] = {
            'member_id': member_id,
            'round': len(self.execution_history['rounds']),
            'error': str(e),
            'code': analysis_code,
            'type': 'analysis'
        }
        return None