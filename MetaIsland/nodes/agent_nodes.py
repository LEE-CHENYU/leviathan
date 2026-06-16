"""
Agent-related execution nodes.
"""
import asyncio
import traceback
from MetaIsland.graph_engine import ExecutionNode
from MetaIsland.agent_code_decision import _apply_agent_action_fallback


class AnalyzeNode(ExecutionNode):
    """Agents analyze game state in parallel"""

    def __init__(self):
        super().__init__("analyze", "process")

    async def execute(self, context, input_data):
        execution = context['execution']

        print("\n[Analyze] All agents analyzing game state...")
        tasks = []
        for member_id in range(len(execution.current_members)):
            tasks.append(execution.analyze(member_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = len(results) - successes

        print(f"[Analyze] Complete: {successes} successful, {failures} failed")

        return {"analyses": results, "successes": successes, "failures": failures}


class ProposeMechanismNode(ExecutionNode):
    """Agents propose mechanism modifications in parallel"""

    def __init__(self):
        super().__init__("propose_mechanisms", "process")

    async def execute(self, context, input_data):
        execution = context['execution']

        print("\n[Propose] All agents proposing mechanisms...")
        tasks = []
        for member_id in range(len(execution.current_members)):
            tasks.append(execution.agent_mechanism_proposal(member_id))

        proposals = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures
        valid_proposals = [p for p in proposals if p is not None and not isinstance(p, Exception)]

        print(f"[Propose] Complete: {len(valid_proposals)} proposals generated")

        return {"proposals": valid_proposals, "count": len(valid_proposals)}


class AgentDecisionNode(ExecutionNode):
    """Agents make decisions in parallel (hidden actions)"""

    def __init__(self):
        super().__init__("agent_decisions", "process")

    async def execute(self, context, input_data):
        execution = context['execution']

        print("\n[Decide] All agents making decisions...")
        tasks = []
        member_ids = []
        for member_id in range(len(execution.current_members)):
            tasks.append(execution.agent_code_decision(member_id))
            member_ids.append(member_id)

        decisions = await asyncio.gather(*tasks, return_exceptions=True)

        # Count valid decisions
        valid_decisions = [d for d in decisions if d is not None and not isinstance(d, Exception)]

        # Ensure every member has a decision; fill gaps with fallback templates
        code_map = {}
        try:
            code_map = execution.agent_code_by_member or {}
        except Exception:
            code_map = {}
        missing_members = [
            member_id
            for member_id in member_ids
            if not code_map.get(member_id)
        ]
        if missing_members:
            for member_id in missing_members:
                err = RuntimeError("agent_decision_missing")
                _apply_agent_action_fallback(
                    execution,
                    member_id,
                    err,
                    reason="agent_decision_missing",
                    error_category_override="agent_decision_missing",
                    traceback_str="".join(
                        traceback.format_exception(err.__class__, err, err.__traceback__)
                    ),
                )
            print(f"[Decide] Filled {len(missing_members)} missing decisions with fallback code")

        print(f"[Decide] Complete: {len(valid_decisions)} decisions made")

        return {"decisions": valid_decisions, "count": len(valid_decisions)}


class ExecuteActionsNode(ExecutionNode):
    """Execute all agent actions with conflict resolution"""

    def __init__(self):
        super().__init__("execute_actions", "process")

    def execute(self, context, input_data):
        execution = context['execution']

        print("\n[Execute] Executing agent actions with conflict resolution...")
        execution.execute_code_actions()

        return {"actions_executed": True}
