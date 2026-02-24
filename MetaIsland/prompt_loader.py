"""
Prompt Loader Utility
Loads and formats prompts from YAML files
"""

import yaml
import os
from typing import Dict, Any


class PromptLoader:
    """Loads and manages prompts from YAML files"""

    def __init__(self, prompts_dir: str = None):
        """
        Initialize the prompt loader

        Args:
            prompts_dir: Directory containing prompt YAML files
        """
        if prompts_dir is None:
            # Default to prompts directory in same folder as this file
            prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')

        self.prompts_dir = prompts_dir
        self._cache = {}

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file and cache it"""
        if filename not in self._cache:
            filepath = os.path.join(self.prompts_dir, filename)
            with open(filepath, 'r') as f:
                self._cache[filename] = yaml.safe_load(f)
        return self._cache[filename]

    def get_base_prompts(self) -> Dict[str, str]:
        """Load base prompts"""
        return self._load_yaml('base_prompts.yaml')

    def get_action_templates(self) -> Dict[str, Any]:
        """Load agent action templates"""
        return self._load_yaml('agent_action_templates.yaml')

    def get_mechanism_templates(self) -> Dict[str, Any]:
        """Load mechanism proposal templates"""
        return self._load_yaml('mechanism_templates.yaml')

    def format_action_templates(self) -> str:
        """Format action templates into a prompt string"""
        templates = self.get_action_templates()

        sections = []

        # Introduction
        sections.append(templates['introduction'])

        # Action templates
        sections.append("\n" + "="*66)
        sections.append("ACTION TEMPLATES")
        sections.append("="*66 + "\n")

        for i, template in enumerate(templates['action_templates'], 1):
            sections.append(f"[Action Template {i}: {template['name']}]")
            sections.append(template['code'])
            sections.append("")

        # Economic strategies
        sections.append("="*66)
        sections.append("ECONOMIC STRATEGY PATTERNS")
        sections.append("="*66 + "\n")

        for strategy in templates['economic_strategies']:
            sections.append(f"[Pattern: {strategy['name']}]")
            sections.append(f"- {strategy['description']}\n")

        # Implementation guide
        sections.append("IMPLEMENTATION GUIDE:")
        for step in templates['implementation_guide']['steps']:
            sections.append(f"{step}")

        sections.append("\nREMEMBER:")
        for reminder in templates['implementation_guide']['reminders']:
            sections.append(f"- {reminder}")

        return "\n".join("" if s is None else str(s) for s in sections)

    def format_mechanism_templates(self, member_id: int) -> str:
        """Format mechanism templates into a prompt string"""
        templates = self.get_mechanism_templates()

        sections = []

        # Introduction
        sections.append(templates['introduction'])

        # Contract templates
        sections.append("\n" + "="*66)
        sections.append("CONTRACT TEMPLATES")
        sections.append("="*66 + "\n")

        for i, template in enumerate(templates['contract_templates'], 1):
            sections.append(f"[Template {i}: {template['name']}]")
            sections.append(template['code'])
            sections.append("")

        # Physics templates
        sections.append("="*66)
        sections.append("PHYSICS CONSTRAINT TEMPLATES")
        sections.append("="*66 + "\n")

        for i, template in enumerate(templates['physics_templates'], 1):
            # Replace MEMBER_ID placeholder with actual member_id
            code = template['code'].replace('MEMBER_ID', str(member_id))
            sections.append(f"[Template {5+i}: {template['name']}]")
            sections.append(code)
            sections.append("")

        # Usage guidelines
        sections.append("="*66)
        sections.append("USAGE GUIDELINES")
        sections.append("="*66 + "\n")

        sections.append("To use these systems in your propose_modification():\n")

        for guideline in templates['usage_guidelines']['steps']:
            sections.append(f"{guideline['title']}")
            for item in guideline['items']:
                sections.append(f"   - {item}")
            sections.append("")

        sections.append("REMEMBER:")
        for reminder in templates['usage_guidelines']['reminders']:
            sections.append(f"- {reminder}")

        return "\n".join("" if s is None else str(s) for s in sections)

    def build_action_prompt(
        self,
        member_id: int,
        island_ideology: str,
        error_context: str,
        current_mechanisms: str,
        report: str,
        features: str,
        relations: str,
        code_memory: str,
        past_performance: str,
        analysis_memory: str,
        analysis_card_summary: str,
        experiment_summary: str,
        strategy_profile: str,
        population_strategy_profile: str,
        population_exploration_summary: str,
        strategy_recommendations: str,
        contextual_strategy_summary: str,
        message_context: str,
        communication_summary: str,
        base_code: str,
        population_state_summary: str = "No population state summary available.",
        contract_summary: str = "Contract activity: unavailable."
    ) -> str:
        """
        Build complete action decision prompt

        Args:
            member_id: ID of the agent
            island_ideology: Island ideology text
            error_context: Previous errors
            current_mechanisms: Available mechanisms
            report: Analysis report
            features: Member features
            relations: Relationship summary
            code_memory: Code memory
            past_performance: Performance history
            analysis_memory: Analysis history
            analysis_card_summary: Recent analysis strategy cards
            experiment_summary: Recent baseline/variation experiment outcomes
            strategy_profile: Summary of action signature coverage
            population_strategy_profile: Population-level strategy diversity summary
            population_exploration_summary: Population-level exploration signals
            strategy_recommendations: Strategy suggestion summary
            contextual_strategy_summary: Contextual strategy performance summary
            population_state_summary: Snapshot of current population state
            contract_summary: Contract activity summary for this agent
            message_context: Received messages
            communication_summary: Recent communication summary
            base_code: Base class code

        Returns:
            Complete formatted prompt
        """
        base = self.get_base_prompts()

        sections = [
            "[Base Code]",
            f"Here is the base code for the Island and Member classes that you should reference when making your actions. Study the mechanisms carefully to ensure your code interacts correctly with the available attributes and methods. Pay special attention to:",
            "- Valid attribute access patterns",
            "- Method parameters and return values",
            "- Constraints and preconditions for actions",
            "- Data structure formats and valid operations",
            base_code,
            "",
            "[Previous code execution errors context]",
            "Here are the errors that occurred in the previous code execution, you can use them as reference to avoid repeating them:",
            error_context,
            "",
            "[Current task]",
            f"You are member_{member_id} in a society that you can help shape.",
            "Write a Python function named agent_action(execution_engine, member_id) that implements your vision of social organization while ensuring your survival.",
            "You should use methods and methods of objects defined in in [Active Mechanisms Modifications] section to make your actions in agent_action(execution_engine, member_id), but DO NOT define another propose_modification(execution_engine) itself in your code.",
            "",
            "[Active Mechanisms Modifications]",
            "You should use following mechanisms have been added by other agents in your code:",
            "- Review them carefully to understand their functionality and constraints",
            "- Leverage compatible mechanisms that align with your goals",
            "- Be mindful of version requirements and dependencies",
            "- Consider how they can be combined strategically",
            "- Test interactions before relying on them critically",
            current_mechanisms,
            "",
            "[Island Ideology]",
            island_ideology,
            "",
            base['critical_constraints'].format(member_id=member_id),
            "",
            base['system_attributes'],
            "",
        ]

        if base.get('execution_flow'):
            sections.append(base['execution_flow'])
            sections.append("")

        if base.get('contracts_and_physics'):
            sections.append(base['contracts_and_physics'])
            sections.append("")

        sections += [
            f"Analysis of the game state:",
            report if report else "No analysis available",
            "",
            "Current status:",
            "Here are the basic information of all members, you should make your own decisions based on them.",
            features,
            "",
            "Relationship summary (parsed from relationship_dict):",
            "Here are the relationships between members:",
            relations,
            "",
            "Code Memory and Previous Performance:",
            code_memory,
            "",
            "Performance history:",
            past_performance,
            "",
            "Population state snapshot:",
            population_state_summary,
            "",
            "Contract activity (your participation):",
            contract_summary,
            "",
            "Analysis Memory:",
            analysis_memory,
            "",
            "Analysis strategy cards:",
            analysis_card_summary,
            "",
            "Experiment outcomes (baseline vs variation):",
            experiment_summary,
            "",
            "Strategy profile:",
            strategy_profile,
            "",
            "Population strategy diversity snapshot:",
            population_strategy_profile,
            "",
            "Population exploration signals:",
            population_exploration_summary,
            "",
            "Strategy recommendations:",
            strategy_recommendations,
            "",
            "Contextual strategy cues:",
            contextual_strategy_summary,
            ""
        ]

        if base.get('self_improvement_loop'):
            sections.append(base['self_improvement_loop'])
            sections.append("")

        if base.get('planning_and_evaluation'):
            sections.append(base['planning_and_evaluation'])
            sections.append("")

        if base.get('memory_guidance'):
            sections.append(base['memory_guidance'])
            sections.append("")

        sections += [
            "Based on the previous code performance, adapt and improve the strategy.",
            "If a previous strategy worked well (high performance), consider building upon it.",
            "If it failed, try a different approach.",
            "",
            "IMPORTANT: Do not simply copy the example implementation below. Instead, use it as inspiration to create your own unique approach combining different methods and strategies in novel ways.",
            "",
            "Your code should include agent_action() function. Do not define another propose_modification() from the [Active Mechanisms Modifications] section instead use the methods and methods of objects defined in the [Active Mechanisms Modifications] section.",
            'def agent_action(execution_engine, member_id):',
            '    """',
            '    Include clear reasoning for each modification to help other agents',
            '    understand the intended benefits and evaluate the proposal.',
            '    You should use methods and methods of objects defined in in [Active Mechanisms Modifications] section to make your actions in agent_action(execution_engine, member_id), but DO NOT define another propose_modification(execution_engine) itself in your code.',
            '    State if you used any of the mechanisms defined in [Active Mechanisms Modifications] in your code.',
            '    """',
            '    <Write your own implementation here>',
            "",
            "While the example above shows one possible approach,",
            "you should create your own unique implementation drawing from the wide range of available methods and strategies.",
            "",
            "Consider novel combinations of different approaches rather than following this exact pattern.",
            "",
            "==================================================================",
            "CONTRACT ACTIONS - USE THE NEW SYSTEMS",
            "==================================================================",
            "",
            self.format_action_templates(),
            ""
        ]

        if base.get('strategy_diversity'):
            sections.append(base['strategy_diversity'])
            sections.append("")

        if base.get('survival_metrics'):
            sections.append(base['survival_metrics'])
            sections.append("")

        if base.get('challenge_questions'):
            sections.append(base['challenge_questions'])
            sections.append("")

        if base.get('coordination_protocols'):
            sections.append(base['coordination_protocols'])
            sections.append("")

        sections.append(base['communication_section'])
        sections.append("")
        sections.append("[Communication Summary]")
        sections.append("Recent communication context for coordination:")
        sections.append(communication_summary)
        sections.append("")
        sections.append("[Received Messages]")
        sections.append("Here are the messages sent by other agents, you can use them as reference to make your own decisions:")
        sections.append(message_context)

        if base.get('final_instruction_action'):
            sections.append("")
            sections.append(base['final_instruction_action'])

        return "\n".join("" if s is None else str(s) for s in sections)

    def build_mechanism_prompt(
        self,
        member_id: int,
        island_ideology: str,
        error_context: str,
        current_mechanisms: str,
        modification_attempts: str,
        report: str,
        features: str,
        relations: str,
        code_memory: str,
        past_performance: str,
        analysis_memory: str,
        analysis_card_summary: str,
        experiment_summary: str,
        strategy_profile: str,
        population_strategy_profile: str,
        population_exploration_summary: str,
        strategy_recommendations: str,
        contextual_strategy_summary: str,
        message_context: str,
        communication_summary: str,
        base_code: str,
        population_state_summary: str = "No population state summary available."
    ) -> str:
        """
        Build complete mechanism proposal prompt

        Args:
            member_id: ID of the agent
            (similar params as build_action_prompt, including experiment_summary)
            population_state_summary: Snapshot of current population state

        Returns:
            Complete formatted prompt
        """
        base = self.get_base_prompts()

        sections = [
            "[Base Code]",
            f"Here is the base code for the Island and Member classes that you should reference when making modifications. Study the mechanisms carefully to ensure your code interacts correctly with the available attributes and methods and objects defined in [Active Mechanisms Modifications]. Pay special attention to:",
            "- Valid attribute access patterns",
            "- Method parameters and return values",
            "- Constraints and preconditions for actions",
            "- Data structure formats and valid operations",
            base_code,
            "",
            "[Previous code execution errors context]",
            "Here are the errors that occurred in the previous code execution, you can use them as reference to avoid repeating them:",
            error_context,
            "",
            "[Current Task]",
            "Island is a mechanical environment that every agent would interact with and get impacted.",
            "As an agent, you can propose modifications to the game mechanics to improve your survival chance.",
            "Write a Python function named propose_modification(execution_engine) that implements your proposal of modifications to the game mechanics.",
            "",
            "[Island Ideology]",
            island_ideology,
            "",
            base['critical_constraints'].format(member_id=member_id),
            "",
            base['system_attributes'],
            "",
        ]

        if base.get('execution_flow'):
            sections.append(base['execution_flow'])
            sections.append("")

        if base.get('contracts_and_physics'):
            sections.append(base['contracts_and_physics'])
            sections.append("")

        sections += [
            f"Analysis of the game state:",
            report if report else "No analysis available",
            "",
            "Current status:",
            "Here are the basic information of all members, you should make your own decisions based on them:",
            features,
            "",
            "Relationship summary (parsed from relationship_dict):",
            "Here are the relationships between members:",
            relations,
            "",
            "Code Memory and Previous Performance:",
            code_memory,
            "",
            "Analysis Memory:",
            analysis_memory,
            "",
            "Analysis strategy cards:",
            analysis_card_summary,
            "",
            "Experiment outcomes (baseline vs variation):",
            experiment_summary,
            "",
            "Population state snapshot:",
            population_state_summary,
            "",
            "Strategy profile:",
            strategy_profile,
            "",
            "Population strategy diversity snapshot:",
            population_strategy_profile,
            "",
            "Population exploration signals:",
            population_exploration_summary,
            "",
            "Strategy recommendations:",
            strategy_recommendations,
            "",
            "Contextual strategy cues:",
            contextual_strategy_summary,
            "",
            "Performance history:",
            past_performance,
            ""
        ]

        if base.get('self_improvement_loop'):
            sections.append(base['self_improvement_loop'])
            sections.append("")

        if base.get('planning_and_evaluation'):
            sections.append(base['planning_and_evaluation'])
            sections.append("")

        if base.get('memory_guidance'):
            sections.append(base['memory_guidance'])
            sections.append("")

        sections += [
            "Based on the previous code performance, propose a modification to the game mechanics.",
            "If a previous proposal worked well (high performance), consider building upon it.",
            "If it failed, try a different approach.",
            "",
            "IMPORTANT: Do not simply copy the example implementation below. Instead, use it as inspiration to create your own unique approach combining different methods and strategies in novel ways.",
            "",
            "While the example above shows one possible approach,",
            "you should create your own unique implementation drawing from the wide range of available methods and strategies.",
            "",
            "Consider novel combinations of different approaches rather than following this exact pattern.",
            "",
            "[NEW SYSTEMS AVAILABLE]",
            "",
            self.format_mechanism_templates(member_id),
            "",
            base['game_mechanics'],
            "",
            "[Active Game Mechanisms]",
            "The following mechanisms have been added by agents and can be referenced when making your own modifications. Review them carefully to:",
            "1. Understand existing functionality and avoid conflicts",
            "2. Build upon successful patterns and improvements",
            "3. Identify opportunities for optimization or extension",
            "4. Remove or deprecate mechanisms that are detrimental to your survival",
            "",
            "When proposing changes, ensure they:",
            "- Align with your agent's goals and survival strategy",
            "- Maintain compatibility with other active mechanisms",
            "- Include proper versioning and rollback procedures",
            "- Follow best practices for stability and performance",
            current_mechanisms,
            "",
            "[Modification Attempt History]",
            "[Previous Modification History]",
            "Review your past modification attempts below to inform future proposals:",
            "- Learn from successful patterns and approaches",
            "- Avoid repeating failed strategies",
            "- Build upon and extend working mechanisms",
            "- Identify opportunities for optimization",
            modification_attempts,
            ""
        ]

        if base.get('mechanism_design_guardrails'):
            sections.append(base['mechanism_design_guardrails'])
            sections.append("")

        if base.get('coordination_protocols'):
            sections.append(base['coordination_protocols'])
            sections.append("")

        sections += [
            "[Message Context]",
            "Recent communication context for coordination:",
            communication_summary,
            "",
            "Here are the messages sent by other agents, you can use them as reference to make your own decisions:",
            message_context,
            "",
            base['implementation_patterns'],
            "",
            base['error_prevention']
        ]

        if base.get('strategy_diversity'):
            sections.append("")
            sections.append(base['strategy_diversity'])

        if base.get('survival_metrics'):
            sections.append("")
            sections.append(base['survival_metrics'])

        if base.get('challenge_questions'):
            sections.append("")
            sections.append(base['challenge_questions'])

        if base.get('final_instruction_mechanism'):
            sections.append("")
            sections.append(base['final_instruction_mechanism'])

        return "\n".join("" if s is None else str(s) for s in sections)


# Global instance
_loader = None

def get_prompt_loader() -> PromptLoader:
    """Get the global prompt loader instance"""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader
