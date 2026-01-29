import traceback

from dotenv import load_dotenv
import aisuite as ai

from MetaIsland.model_router import model_router
from MetaIsland.prompt_loader import get_prompt_loader

load_dotenv()

client = ai.Client()

provider, model_id = model_router("deepseek")


async def _agent_code_decision(self, member_id) -> None:
    """
    Asks GPT for directly executable Python code, stores it in a dictionary keyed by member_id.
    The code will define a function agent_action(execution_engine, member_id),
    which references attributes that actually exist.
    """
    data = self.prepare_agent_data(member_id, error_context_type="agent_action")
    member = data['member']
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
    report = data['report']

    base_code = self.base_class_code
    code_result = ""

    try:
        loader = get_prompt_loader()
        current_mechanisms_text = self.format_mechanisms_for_prompt(current_mechanisms)

        prompt = loader.build_action_prompt(
            member_id=member.id,
            island_ideology=self.island_ideology,
            error_context=error_context,
            current_mechanisms=current_mechanisms_text,
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
            message_context=message_context,
            communication_summary=communication_summary,
            base_code=base_code,
            population_state_summary=population_state_summary
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
        if round_num not in self.execution_history['rounds'][-1]['generated_code']:
            self.execution_history['rounds'][-1]['generated_code'] = {}

        self.execution_history['rounds'][-1]['generated_code'][member_id] = {
            'code': code_result,
            'features_at_generation': features.to_dict('records'),
            'relationships_at_generation': relations,
            'final_prompt': prompt
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
