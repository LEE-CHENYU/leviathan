import traceback
import ast

from dotenv import load_dotenv
import aisuite as ai

from MetaIsland.model_router import model_router
from MetaIsland.prompt_loader import get_prompt_loader

load_dotenv()

client = ai.Client()

provider, model_id = model_router("deepseek")


async def _agent_mechanism_proposal(self, member_id) -> None:
    """ 
    Asks GPT for directly executable Python code, stores it in a dictionary keyed by member_id.
    The code will define a function propose_modification(execution_engine),
    which references attributes that actually exist.
    """
    # Prepare data for the proposal
    data = self.prepare_agent_data(member_id, error_context_type="mechanism")

    # Use the prepared data
    member = data['member']
    stable_id = getattr(member, "id", None)
    relations = data['relations']
    features = data['features']
    code_memory = data['code_memory']
    analysis_memory = data['analysis_memory']
    past_performance = data['past_performance']
    error_context = data['error_context']
    message_context = data['message_context']
    communication_summary = data['communication_summary']
    contextual_strategy_summary = data.get(
        'contextual_strategy_summary',
        'No contextual strategy data.'
    )
    population_state_summary = data.get(
        'population_state_summary',
        'No population state summary available.'
    )

    current_mechanisms = data['current_mechanisms']
    modification_attempts = []
    for round_num in data['modification_attempts'].keys():
        round_attempts = [
            attempt
            for attempt in data['modification_attempts'][round_num]
            if attempt.get('member_id') == stable_id
            or attempt.get('member_index') == member_id
            or attempt.get('member_id') == member_id
        ]
        modification_attempts.extend(round_attempts)
    report = data['report']

    base_code = self.base_class_code
    code_result = ""

    def make_class_picklable(class_name, class_dict):
        """Make dynamically created classes picklable by adding them to globals"""
        new_class = type(class_name, (), class_dict)
        globals()[class_name] = new_class
        return new_class

    try:
        loader = get_prompt_loader()
        current_mechanisms_text = self.format_mechanisms_for_prompt(current_mechanisms)
        modification_attempts_text = self.format_modification_attempts_for_prompt(modification_attempts)

        prompt = loader.build_mechanism_prompt(
            member_id=member.id,
            island_ideology=self.island_ideology,
            error_context=error_context,
            current_mechanisms=current_mechanisms_text,
            modification_attempts=modification_attempts_text,
            report=report if report else "No analysis available",
            features=features,
            relations=relations,
            code_memory=code_memory,
            past_performance=past_performance,
            analysis_memory=analysis_memory,
            analysis_card_summary=data.get(
                'analysis_card_summary',
                'No analysis cards available.'
            ),
            experiment_summary=data.get(
                'experiment_summary',
                'No experiment outcomes available.'
            ),
            strategy_profile=data.get('strategy_profile', 'No strategy profile available.'),
            population_strategy_profile=data.get(
                'population_strategy_profile',
                'No population strategy data.'
            ),
            population_exploration_summary=data.get(
                'population_exploration_summary',
                'No population exploration data.'
            ),
            strategy_recommendations=data.get(
                'strategy_recommendations',
                'No strategy recommendations available.'
            ),
            contextual_strategy_summary=contextual_strategy_summary,
            population_state_summary=population_state_summary,
            message_context=message_context,
            communication_summary=communication_summary,
            base_code=base_code
        )

        completion = client.chat.completions.create(
            model=f'{provider}:{model_id}',
            messages=[{"role": "user", "content": prompt}]
        )
        code_result = completion.choices[0].message.content.strip()

        # Clean and store the code
        code_result = self.clean_code_string(code_result)

        # Log the generated code
        round_num = len(self.execution_history['rounds'])
        mod_proposal = {
            'round': round_num,
            'member_id': stable_id if stable_id is not None else member_id,
            'member_index': member_id,
            'code': code_result,
            'features_at_generation': features.to_dict('records'),
            'relationships_at_generation': relations,
            'final_prompt': prompt,
            'ratified': False
        }
        self.execution_history['rounds'][-1]['mechanism_modifications']['attempts'].append(mod_proposal)

        print(f"\nGenerated code for Member {member_id}:")

        # Extract class definitions from the code
        tree = ast.parse(code_result)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_dict = {'__module__': '__main__'}
                make_class_picklable(node.name, class_dict)

        return code_result

    except Exception as e:
        stable_id = stable_id if stable_id is not None else None
        error_info = {
            'round': len(self.execution_history['rounds']),
            'member_id': stable_id if stable_id is not None else member_id,
            'member_index': member_id,
            'type': 'propose_modification',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'code': code_result
        }
        self.execution_history['rounds'][-1]['errors']['mechanism_errors'].append(error_info)
        print(f"Error generating code for member {member_id}:")
        print(traceback.format_exc())
        self._logger.error(f"GPT Code Generation Error (member {member_id}): {e}")
