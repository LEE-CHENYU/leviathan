import traceback

from dotenv import load_dotenv

from MetaIsland.llm_client import get_llm_client
from MetaIsland.model_router import model_router
from MetaIsland.prompt_loader import get_prompt_loader
from MetaIsland.llm_utils import (
    build_chat_kwargs,
    build_prompt_diagnostics,
    build_code_stats,
    classify_llm_error,
    describe_syntax_error,
    ensure_non_empty_response,
    extract_completion_metadata,
    extract_request_metadata,
    merge_prompt_sections,
)

load_dotenv()

client = get_llm_client()

provider, model_id = model_router("default")


DEFAULT_FALLBACK_CODE = """def agent_action(execution_engine, member_id):
    \"\"\"Fallback strategy when LLM output is unavailable.\"\"\"
    members = getattr(execution_engine, "current_members", [])
    if not members or member_id >= len(members):
        return
    me = members[member_id]
    history = getattr(execution_engine, "execution_history", {})
    round_num = 0
    if isinstance(history, dict):
        round_num = len(history.get("rounds", []))
    policy = (member_id + round_num) % 3
    partner = None
    if len(members) > 1:
        partner_idx = (member_id + 1) % len(members)
        if partner_idx == member_id:
            partner_idx = (member_id - 1) % len(members)
        partner = members[partner_idx]
    if policy == 0 and partner is not None and getattr(me, "cargo", 0) > 0:
        if hasattr(execution_engine, "offer"):
            execution_engine.offer(me, partner)
            return
    if policy == 1 and partner is not None and hasattr(execution_engine, "attack"):
        execution_engine.attack(me, partner)
        return
    if hasattr(execution_engine, "expand"):
        execution_engine.expand(me)
"""


FALLBACK_TEMPLATES = [
    {
        "name": "expander",
        "code": """def agent_action(execution_engine, member_id):
    \"\"\"Fallback strategy: expand bias.\"\"\"
    members = getattr(execution_engine, "current_members", [])
    if not members or member_id >= len(members):
        return
    me = members[member_id]
    if hasattr(execution_engine, "expand"):
        execution_engine.expand(me)
""",
    },
    {
        "name": "diplomat",
        "code": """def agent_action(execution_engine, member_id):
    \"\"\"Fallback strategy: cooperative trade.\"\"\"
    members = getattr(execution_engine, "current_members", [])
    if not members or member_id >= len(members):
        return
    me = members[member_id]
    partner = None
    if len(members) > 1:
        partner = members[(member_id + 1) % len(members)]
    if partner is not None and hasattr(execution_engine, "offer"):
        execution_engine.offer(me, partner)
    sender_id = getattr(me, "id", member_id)
    recipient_id = getattr(partner, "id", None) if partner is not None else None
    if recipient_id is not None and hasattr(execution_engine, "send_message"):
        execution_engine.send_message(sender_id, recipient_id, "Open to trade or alliance.")
    if hasattr(execution_engine, "expand"):
        execution_engine.expand(me)
""",
    },
    {
        "name": "raider",
        "code": """def agent_action(execution_engine, member_id):
    \"\"\"Fallback strategy: aggressive pressure.\"\"\"
    members = getattr(execution_engine, "current_members", [])
    if not members or member_id >= len(members):
        return
    me = members[member_id]
    target = None
    if len(members) > 1:
        target = members[(member_id + 1) % len(members)]
    if target is not None and hasattr(execution_engine, "attack"):
        execution_engine.attack(me, target)
    elif target is not None and hasattr(execution_engine, "bear"):
        execution_engine.bear(me, target)
    if hasattr(execution_engine, "expand"):
        execution_engine.expand(me)
""",
    },
    {
        "name": "broker",
        "code": """def agent_action(execution_engine, member_id):
    \"\"\"Fallback strategy: land broker.\"\"\"
    members = getattr(execution_engine, "current_members", [])
    if not members or member_id >= len(members):
        return
    me = members[member_id]
    partner = None
    if len(members) > 1:
        partner = members[(member_id + 1) % len(members)]
    if partner is not None and hasattr(execution_engine, "offer_land"):
        execution_engine.offer_land(me, partner)
    if partner is not None and hasattr(execution_engine, "offer"):
        execution_engine.offer(me, partner)
    if hasattr(execution_engine, "expand"):
        execution_engine.expand(me)
""",
    },
]


def _select_fallback_code(self, member_id):
    """Select a fallback action implementation from memory if possible."""
    try:
        _, memory = self._get_member_history(self.code_memory, member_id)
    except Exception:
        return None, {}
    if not memory:
        return None, {}
    candidates = []
    for mem in memory:
        if not isinstance(mem, dict):
            continue
        code = mem.get("code")
        if code:
            candidates.append(mem)
    if not candidates:
        return None, {}
    clean_candidates = [mem for mem in candidates if not mem.get("error")]
    if clean_candidates:
        candidates = clean_candidates

    current_tags = {}
    try:
        current_tags = self._get_member_context_tags(member_id)
    except Exception:
        current_tags = {}
    match_idx, match_score = None, 0.0
    try:
        match_idx, match_score = self._find_contextual_memory_match(candidates, current_tags)
    except Exception:
        match_idx, match_score = None, 0.0

    if match_idx is not None and match_score > 0:
        mem = candidates[match_idx]
        return mem.get("code"), {
            "source": "memory_context_match",
            "match_score": match_score,
            "memory_round": mem.get("context", {}).get("round"),
        }

    best_mem = max(
        candidates,
        key=lambda mem: (
            self._get_memory_performance(mem),
            mem.get("context", {}).get("round", -1),
        ),
    )
    return best_mem.get("code"), {
        "source": "memory_best_performance",
        "memory_round": best_mem.get("context", {}).get("round"),
    }


def _select_fallback_template(self, member_id):
    """Select a deterministic fallback template to preserve diversity."""
    if not FALLBACK_TEMPLATES:
        return None, {}
    round_num = 0
    try:
        round_num = len(self.execution_history.get("rounds", []))
    except Exception:
        round_num = 0
    stable_id = None
    try:
        stable_id = self._resolve_member_stable_id(member_id)
    except Exception:
        stable_id = None
    selector = (stable_id if stable_id is not None else member_id) + round_num
    template_idx = selector % len(FALLBACK_TEMPLATES)
    template = FALLBACK_TEMPLATES[template_idx]
    return template.get("code"), {
        "source": "template_variant",
        "variant": template.get("name"),
        "template_index": template_idx,
        "round": round_num,
    }


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
    contract_summary = data.get(
        'contract_summary',
        'Contract activity: unavailable.'
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
    report = data['report']

    base_code = self.base_class_code
    code_result = ""
    raw_code = None
    completion_metadata = {}
    report_text = report if report else "No analysis available"

    try:
        loader = get_prompt_loader()
        current_mechanisms_text = self.format_mechanisms_for_prompt(current_mechanisms)

        prompt = loader.build_action_prompt(
            member_id=member.id,
            island_ideology=self.island_ideology,
            error_context=error_context,
            current_mechanisms=current_mechanisms_text,
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
            message_context=message_context,
            communication_summary=communication_summary,
            base_code=base_code,
            population_state_summary=population_state_summary,
            contract_summary=contract_summary
        )
        prompt_sections = {
            "base_code": base_code,
            "current_mechanisms": current_mechanisms_text,
            "error_context": error_context,
            "report": report_text,
            "features": features,
            "relations": relations,
            "code_memory": code_memory,
            "past_performance": past_performance,
            "analysis_memory": analysis_memory,
            "analysis_card_summary": analysis_card_summary,
            "experiment_summary": experiment_summary,
            "strategy_profile": strategy_profile,
            "population_strategy_profile": population_strategy_profile,
            "population_exploration_summary": population_exploration_summary,
            "strategy_recommendations": strategy_recommendations,
            "contextual_strategy_summary": contextual_strategy_summary,
            "message_context": message_context,
            "communication_summary": communication_summary,
            "population_state_summary": population_state_summary,
            "contract_summary": contract_summary,
            "island_ideology": self.island_ideology,
        }
        prompt_sections = merge_prompt_sections(
            prompt_sections,
            getattr(self, "base_code", None),
            "base_code_",
        )
        prompt_diag = build_prompt_diagnostics(prompt, prompt_sections)
        completion_metadata.update(prompt_diag)

        chat_kwargs = build_chat_kwargs()
        completion = client.chat.completions.create(
            model=f'{provider}:{model_id}',
            messages=[{"role": "user", "content": prompt}],
            **chat_kwargs
        )
        completion_metadata = extract_completion_metadata(completion)
        completion_metadata.update(extract_request_metadata(chat_kwargs))
        completion_metadata.update(prompt_diag)
        raw_code = completion.choices[0].message.content
        raw_code = ensure_non_empty_response(
            raw_code,
            context="agent_action",
        )

        # Clean and store the code
        code_result = self.clean_code_string(raw_code)
        code_result = ensure_non_empty_response(
            code_result,
            context="agent_action_cleaned",
        )

        # Log the generated code
        round_num = len(self.execution_history['rounds'])
        if round_num not in self.execution_history['rounds'][-1]['generated_code']:
            self.execution_history['rounds'][-1]['generated_code'] = {}

        self.execution_history['rounds'][-1]['generated_code'][member_id] = {
            'code': code_result,
            'features_at_generation': features.to_dict('records'),
            'relationships_at_generation': relations,
            'final_prompt': prompt,
            'llm_metadata': completion_metadata,
        }

        print(f"\nGenerated code for Member {member_id}:")

        if not hasattr(self, 'agent_code_by_member'):
            self.agent_code_by_member = {}
        self.agent_code_by_member[member_id] = code_result

        return code_result

    except Exception as e:
        stable_id = None
        try:
            stable_id = self._resolve_member_stable_id(member_id)
        except Exception:
            stable_id = None
        fallback_code, fallback_meta = _select_fallback_code(self, member_id)
        if not fallback_code:
            fallback_code, fallback_meta = _select_fallback_template(self, member_id)
        if not fallback_code:
            fallback_code = DEFAULT_FALLBACK_CODE
            fallback_meta = {"source": "template_default"}
        try:
            fallback_code = self.clean_code_string(fallback_code)
        except Exception:
            pass

        if not hasattr(self, 'agent_code_by_member'):
            self.agent_code_by_member = {}
        self.agent_code_by_member[member_id] = fallback_code

        try:
            features_payload = features.to_dict('records')
        except Exception:
            features_payload = features

        round_record = self.execution_history['rounds'][-1]
        if not isinstance(round_record.get('generated_code'), dict):
            round_record['generated_code'] = {}
        round_record['generated_code'][member_id] = {
            'code': fallback_code,
            'features_at_generation': features_payload,
            'relationships_at_generation': relations,
            'final_prompt': None,
            'fallback': fallback_meta,
            'llm_metadata': completion_metadata,
        }

        error_info = {
            'round': len(self.execution_history['rounds']),
            'member_id': stable_id if stable_id is not None else member_id,
            'member_index': member_id,
            'type': 'agent_action',
            'error': str(e),
            'error_category': classify_llm_error(e),
            'traceback': traceback.format_exc(),
            'code': code_result,
            'llm_metadata': completion_metadata,
            'fallback_used': True,
            'fallback_source': fallback_meta.get('source'),
            'fallback_meta': fallback_meta,
            'fallback_code': fallback_code
        }
        raw_code_for_error = locals().get("raw_code")
        code_result_for_error = locals().get("code_result")
        error_details = describe_syntax_error(
            e,
            code_result_for_error or raw_code_for_error,
        )
        if error_details:
            error_info['error_details'] = error_details
        code_stats = build_code_stats(raw_code, code_result)
        if code_stats:
            error_info['code_stats'] = code_stats
        self.execution_history['rounds'][-1]['errors']['agent_code_errors'].append(error_info)
        print(f"Error generating code for member {member_id}:")
        print(traceback.format_exc())
        print(f"Using fallback code for member {member_id}: {fallback_meta.get('source')}")
        self._logger.error(f"GPT Code Generation Error (member {member_id}): {e}")
        return fallback_code
