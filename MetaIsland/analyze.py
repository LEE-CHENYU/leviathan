import json
import numpy as np
import openai
import traceback

from dotenv import load_dotenv

from MetaIsland.llm_client import get_llm_client
from MetaIsland.model_router import model_router
from MetaIsland.llm_utils import build_chat_kwargs, classify_llm_error

load_dotenv()

client = get_llm_client()

provider, model_id = model_router("default")


def _fallback_analysis_text(member_id: int) -> str:
    baseline = ["expand"] if member_id % 2 == 0 else ["offer"]
    variation = ["offer"] if member_id % 2 == 0 else ["expand"]
    card = {
        "hypothesis": "Fallback analysis: keep plan tags available when LLM is unavailable.",
        "baseline_signature": baseline,
        "variation_signature": variation,
        "success_metrics": ["delta_survival", "delta_vitality"],
        "guardrails": ["avoid negative survival deltas"],
        "coordination": [],
        "memory_note": f"fallback_stub_{member_id}",
        "diversity_note": "Rotate tags across members to avoid monoculture.",
        "confidence": 0.2,
    }
    return "\n".join([
        "Situation summary:",
        "- Fallback analysis stub (no external LLM call).",
        "Risks & opportunities:",
        "- Treat results as pipeline validation only.",
        "Strategy plan:",
        f"- Baseline tags: {', '.join(baseline)}",
        f"- Variation tags: {', '.join(variation)}",
        "Coordination asks: none.",
        "Memory note: fallback stub.",
        "```json",
        json.dumps(card, indent=2),
        "```",
    ])

async def _analyze(self, member_id):
    """
    Analyze member data for strategic insights
    """
    
    member = self.current_members[member_id]
    
    data = self.prepare_agent_data(member_id, error_context_type="analysis")
    member = data['member']
    relations = data['relations']
    features = data['features']
    code_memory = data['code_memory']
    analysis_memory = data['analysis_memory']
    past_performance = data['past_performance']
    error_context = data['error_context']
    message_context = data['message_context']
    communication_summary = data['communication_summary']
    strategy_profile = data['strategy_profile']
    population_strategy_profile = data['population_strategy_profile']
    population_exploration_summary = data['population_exploration_summary']
    strategy_recommendations = data['strategy_recommendations']
    experiment_summary = data.get(
        'experiment_summary',
        'No experiment outcomes available.'
    )
    contextual_strategy_summary = data.get(
        'contextual_strategy_summary',
        'No contextual strategy data.'
    )
    population_state_summary = data.get(
        'population_state_summary',
        'No population state summary available.'
    )

    current_mechanisms = data['current_mechanisms']
    current_mechanisms_text = self.format_mechanisms_for_prompt(current_mechanisms)
    modification_attempts = []
    modification_attempts_map = data.get('modification_attempts') or {}
    if modification_attempts_map:
        try:
            latest_round = max(modification_attempts_map.keys())
            modification_attempts = modification_attempts_map.get(latest_round, [])
        except Exception:
            modification_attempts = []
    report = data['report']
    
    base_code = self.base_class_code
    
    """Analyze member data for strategic insights"""
    previous_errors = error_context
    analysis_prompt = f"""
    {previous_errors}
    
    [Active Mechanisms Modifications]
    You should use following mechanisms have been added by other agents in your code:
    - Review them carefully to understand their functionality and constraints
    - Leverage compatible mechanisms that align with your goals
    - Be mindful of version requirements and dependencies
    - Consider how they can be combined strategically
    - Test interactions before relying on them critically
    {current_mechanisms_text}
    
    [Current task]
    You are member_{member.id} in a society that you can help shape. Use plain text to describe your situation and propose strategies. All your words should be based on concrete facts and data below. You should also think about how to use the mechanisms defined in [Active Mechanisms Modifications] to make your strategies more effective. Strategies should be practical and effective, if you can't find any effective strategies, just say so. If your analysis and strategies are not fact-oriented, you will not be competitive and you will die.
    
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

    Strategy profile:
    {strategy_profile}

    Population strategy diversity snapshot:
    {population_strategy_profile}

    Population exploration signals:
    {population_exploration_summary}

    Strategy recommendations:
    {strategy_recommendations}

    Experiment outcomes (baseline vs variation):
    {experiment_summary}

    Contextual strategy cues:
    {contextual_strategy_summary}

    Population state snapshot:
    {population_state_summary}
    
    Analysis Memory:
    {analysis_memory}

    Performance history:
    {past_performance}
    
    Previous Analysis of the game state:
    {report} 

    [Communication Summary]
    {communication_summary}

    [Received Messages]
    {message_context}

    [Output Format]
    Use plain text. Be concise and evidence-based.
    Provide:
    1) Situation summary (<=5 bullets, cite member IDs/metrics).
    2) Risks & opportunities (<=5 bullets).
    3) Strategy plan with two distinct action signatures:
       - Baseline (safe): include action tags from
         [attack, offer, offer_land, bear, expand, message, contracts, market, resources, businesses].
       - Variation (bounded-risk): different tags or combinations.
       - Guardrails / stop conditions.
       - Success metrics using available deltas (delta_survival, delta_vitality, delta_cargo, delta_relation_balance, delta_land).
    4) Coordination asks (message drafts with opt-in roles).
    5) Memory note (<=120 chars).
    End with a JSON block in a ```json``` fence (required) using keys:
    hypothesis, baseline_signature, variation_signature, success_metrics,
    guardrails, coordination, memory_note, diversity_note, confidence.
    Keep diversity: avoid monoculture or single-equilibrium recommendations.
    
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
            
    stable_id = None
    try:
        stable_id = self._resolve_member_stable_id(member_id)
    except Exception:
        stable_id = None

    try:
        completion = client.chat.completions.create(
                    model=f'{provider}:{model_id}',
                    messages=[{"role": "user", "content": analysis_prompt}],
                    **build_chat_kwargs()
                )
        result = completion.choices[0].message.content.strip()
    except Exception as e:
        fallback_member_id = stable_id if stable_id is not None else member_id
        try:
            fallback_member_id = int(fallback_member_id)
        except (TypeError, ValueError):
            fallback_member_id = member_id
        result = _fallback_analysis_text(fallback_member_id)
        analysis_key = stable_id if stable_id is not None else member_id
        error_info = {
            'round': len(self.execution_history['rounds']),
            'member_id': analysis_key,
            'member_index': member_id,
            'type': 'analysis',
            'error': str(e),
            'error_category': classify_llm_error(e),
            'traceback': traceback.format_exc(),
            'code': "",
            'fallback_used': True,
            'fallback_source': 'analysis_stub'
        }
        self.execution_history['rounds'][-1]['errors']['analyze_code_errors'][analysis_key] = error_info
        print(f"Error generating analysis for member {member_id}:")
        print(traceback.format_exc())
        if hasattr(self, "_logger"):
            try:
                self._logger.error(f"Analysis generation error (member {member_id}): {e}")
            except Exception:
                pass
    # analysis_code = self.clean_code_string(analysis_code)
    
    # print(f"\nStrategic Analysis Code:\n{analysis_code}")
    
    # Execute the code in a new environment
    exec_env = {}
    exec_env['execution_engine'] = self
    exec_env['member_id'] = member_id
    exec_env['me'] = self.current_members[member_id]
    exec_env['np'] = np  # Make numpy available in the environment

    # print(f"Analysis result: {result}")
    # Store analysis in execution history using stable member id when possible
    analysis_key = stable_id if stable_id is not None else member_id
    self.execution_history['rounds'][-1]['analysis'][analysis_key] = result
    try:
        self._record_analysis_card(member_id, result)
    except Exception:
        pass
    return result
    
    # try:
    #     # First execute the code to define the functions
    #     exec(analysis_code, exec_env)
        
    #     # Then call the analysis function
    #     result = exec_env['analyze_game_state'](self, member_id)
        
    #     if result:
    #         print(f"Analysis result: {result}")
            # # Store analysis in execution history
            # if member_id not in self.execution_history['rounds'][-1]['analysis']:
            #     self.execution_history['rounds'][-1]['analysis'][member_id] = []
            # self.execution_history['rounds'][-1]['analysis'][member_id].append(result)
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
