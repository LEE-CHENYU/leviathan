import traceback
import ast

from dotenv import load_dotenv

from MetaIsland.llm_client import get_llm_client, get_offline_client
from MetaIsland.model_router import model_router
from MetaIsland.prompt_loader import get_prompt_loader
from MetaIsland.llm_utils import (
    build_chat_kwargs,
    build_prompt_diagnostics,
    classify_llm_error,
    ensure_non_empty_response,
    extract_completion_metadata,
    extract_request_metadata,
    should_use_offline_fallback,
    merge_prompt_sections,
)

load_dotenv()

client = get_llm_client()

provider, model_id = model_router("default")

FALLBACK_MECHANISM_CODE = """def propose_modification(execution_engine):
    \"\"\"Fallback: no-op when LLM output is empty.\"\"\"
    return None
"""


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
    analysis_card_summary = data.get(
        'analysis_card_summary',
        'No analysis cards available.'
    )
    experiment_summary = data.get(
        'experiment_summary',
        'No experiment outcomes available.'
    )
    strategy_profile = data.get('strategy_profile', 'No strategy profile available.')
    population_strategy_profile = data.get(
        'population_strategy_profile',
        'No population strategy data.'
    )
    population_exploration_summary = data.get(
        'population_exploration_summary',
        'No population exploration data.'
    )
    strategy_recommendations = data.get(
        'strategy_recommendations',
        'No strategy recommendations available.'
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
    primary_llm_metadata = {}
    fallback_llm_metadata = {}
    report_text = report if report else "No analysis available"

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
            report=report_text,
            features=features,
            relations=relations,
            code_memory=code_memory,
            past_performance=past_performance,
            analysis_memory=analysis_memory,
            analysis_card_summary=analysis_card_summary,
            experiment_summary=experiment_summary,
            strategy_profile=strategy_profile,
            population_strategy_profile=population_strategy_profile,
            population_exploration_summary=population_exploration_summary,
            strategy_recommendations=strategy_recommendations,
            contextual_strategy_summary=contextual_strategy_summary,
            population_state_summary=population_state_summary,
            message_context=message_context,
            communication_summary=communication_summary,
            base_code=base_code
        )
        prompt_sections = {
            "base_code": base_code,
            "current_mechanisms": current_mechanisms_text,
            "modification_attempts": modification_attempts_text,
            "error_context": error_context,
            "report": report_text,
            "features": features,
            "relations": relations,
            "code_memory": code_memory,
            "analysis_memory": analysis_memory,
            "analysis_card_summary": analysis_card_summary,
            "experiment_summary": experiment_summary,
            "population_state_summary": population_state_summary,
            "strategy_profile": strategy_profile,
            "population_strategy_profile": population_strategy_profile,
            "population_exploration_summary": population_exploration_summary,
            "strategy_recommendations": strategy_recommendations,
            "contextual_strategy_summary": contextual_strategy_summary,
            "past_performance": past_performance,
            "message_context": message_context,
            "communication_summary": communication_summary,
            "island_ideology": self.island_ideology,
        }
        prompt_sections = merge_prompt_sections(
            prompt_sections,
            getattr(self, "base_code", None),
            "base_code_",
        )
        prompt_diag = build_prompt_diagnostics(prompt, prompt_sections)
        primary_llm_metadata.update(prompt_diag)
        fallback_llm_metadata.update(prompt_diag)

        fallback_used = False
        fallback_error = None
        fallback_traceback = None
        fallback_source = None
        try:
            chat_kwargs = build_chat_kwargs()
            completion = client.chat.completions.create(
                model=f'{provider}:{model_id}',
                messages=[{"role": "user", "content": prompt}],
                **chat_kwargs
            )
            primary_llm_metadata = extract_completion_metadata(completion)
            primary_llm_metadata.update(extract_request_metadata(chat_kwargs))
            primary_llm_metadata.update(prompt_diag)
            code_result = ensure_non_empty_response(
                completion.choices[0].message.content,
                context="mechanism",
            )
        except Exception as e:
            fallback_error = e
            fallback_traceback = traceback.format_exc()
            if should_use_offline_fallback(e):
                fallback_used = True
                fallback_source = "offline_stub"
                offline_client = get_offline_client()
                fallback_kwargs = build_chat_kwargs()
                completion = offline_client.chat.completions.create(
                    model=f'{provider}:{model_id}',
                    messages=[{"role": "user", "content": prompt}],
                    **fallback_kwargs
                )
                fallback_llm_metadata = extract_completion_metadata(completion)
                fallback_llm_metadata.update(extract_request_metadata(fallback_kwargs))
                fallback_llm_metadata.update(prompt_diag)
                code_result = ensure_non_empty_response(
                    completion.choices[0].message.content,
                    context="mechanism_offline",
                )
            else:
                raise

        # Clean and store the code
        code_result = self.clean_code_string(code_result)
        try:
            code_result = ensure_non_empty_response(
                code_result,
                context="mechanism_cleaned",
            )
        except Exception as e:
            fallback_used = True
            if fallback_error is None:
                fallback_error = e
            if fallback_traceback is None:
                fallback_traceback = traceback.format_exc()
            if fallback_source is None:
                fallback_source = "empty_cleaned"
            code_result = self.clean_code_string(FALLBACK_MECHANISM_CODE)

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
        llm_metadata = fallback_llm_metadata or primary_llm_metadata
        if llm_metadata:
            mod_proposal['llm_metadata'] = llm_metadata
        if fallback_used and primary_llm_metadata and fallback_llm_metadata:
            mod_proposal['llm_metadata_primary'] = primary_llm_metadata
        self.execution_history['rounds'][-1]['mechanism_modifications']['attempts'].append(mod_proposal)

        print(f"\nGenerated code for Member {member_id}:")

        # Extract class definitions from the code
        tree = ast.parse(code_result)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_dict = {'__module__': '__main__'}
                make_class_picklable(node.name, class_dict)

        if fallback_used and fallback_error is not None:
            error_info = {
                'round': len(self.execution_history['rounds']),
                'member_id': stable_id if stable_id is not None else member_id,
                'member_index': member_id,
                'type': 'propose_modification',
                'error': str(fallback_error),
                'error_category': classify_llm_error(fallback_error),
                'traceback': fallback_traceback,
                'code': code_result,
                'fallback_used': True,
                'fallback_source': fallback_source,
                'llm_metadata': llm_metadata,
            }
            if fallback_used and primary_llm_metadata and fallback_llm_metadata:
                error_info['llm_metadata_primary'] = primary_llm_metadata
            self.execution_history['rounds'][-1]['errors']['mechanism_errors'].append(error_info)

        return code_result

    except Exception as e:
        stable_id = stable_id if stable_id is not None else None
        error_info = {
            'round': len(self.execution_history['rounds']),
            'member_id': stable_id if stable_id is not None else member_id,
            'member_index': member_id,
            'type': 'propose_modification',
            'error': str(e),
            'error_category': classify_llm_error(e),
            'traceback': traceback.format_exc(),
            'code': code_result,
            'llm_metadata': fallback_llm_metadata or primary_llm_metadata,
        }
        if fallback_used and primary_llm_metadata and fallback_llm_metadata:
            error_info['llm_metadata_primary'] = primary_llm_metadata
        self.execution_history['rounds'][-1]['errors']['mechanism_errors'].append(error_info)
        print(f"Error generating code for member {member_id}:")
        print(traceback.format_exc())
        self._logger.error(f"GPT Code Generation Error (member {member_id}): {e}")
