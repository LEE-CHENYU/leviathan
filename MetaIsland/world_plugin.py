"""
MetaIsland world-plugin boundary for graph execution.

This keeps island-specific phase logic out of ``IslandExecution`` so the
execution shell can depend on a pluggable world adapter.
"""

from __future__ import annotations

import asyncio
from typing import Any

from MetaIsland.nodes import (
    AgentDecisionNode,
    AgentReviewNode,
    AnalyzeNode,
    CanaryNode,
    ConsumeNode,
    ContractNode,
    EnvironmentNode,
    ExecuteActionsNode,
    ExecuteMechanismsNode,
    JudgeNode,
    LogStatusNode,
    NewRoundNode,
    ProduceNode,
    ProposeMechanismNode,
)


class MetaIslandWorldPlugin:
    """Owns the MetaIsland phase graph and world-phase execution logic."""

    def build_default_graph(self, graph) -> None:
        nodes = [
            NewRoundNode(),
            AnalyzeNode(),
            ProposeMechanismNode(),
            CanaryNode(),
            AgentReviewNode(),
            ExecuteMechanismsNode(),
            AgentDecisionNode(),
            ExecuteActionsNode(),
            ContractNode(),
            ProduceNode(),
            ConsumeNode(),
            EnvironmentNode(),
            LogStatusNode(),
        ]

        for node in nodes:
            graph.add_node(node)

        connections = [
            ("new_round", "analyze", "default", "default"),
            ("analyze", "propose_mechanisms", "default", "default"),
            ("propose_mechanisms", "canary", "proposals", "proposals"),
            ("canary", "agent_review", "proposals", "proposals"),
            ("agent_review", "execute_mechanisms", "approved", "approved"),
            ("execute_mechanisms", "agent_decisions", "default", "default"),
            ("agent_decisions", "execute_actions", "default", "default"),
            ("execute_actions", "contracts", "default", "default"),
            ("contracts", "produce", "default", "default"),
            ("produce", "consume", "default", "default"),
            ("consume", "environment", "default", "default"),
            ("environment", "log_status", "default", "default"),
        ]
        for from_name, to_name, output_key, input_key in connections:
            graph.connect(
                from_name,
                to_name,
                output_key=output_key,
                input_key=input_key,
            )

    def build_graph_context(self, execution, round_number: int) -> dict[str, Any]:
        return {
            "execution": execution,
            "round": round_number,
            "world_plugin": self,
        }

    def run_new_round(self, execution, context, input_data):
        execution.new_round()
        execution.get_neighbors()
        return {"round_initialized": True}

    async def run_analyze(self, execution, context, input_data):
        print("\n[Analyze] All agents analyzing game state...")
        tasks = [execution.analyze(member_id) for member_id in range(len(execution.current_members))]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = sum(1 for result in results if not isinstance(result, Exception))
        failures = len(results) - successes
        print(f"[Analyze] Complete: {successes} successful, {failures} failed")
        return {"analyses": results, "successes": successes, "failures": failures}

    async def run_propose_mechanisms(self, execution, context, input_data):
        print("\n[Propose] All agents proposing mechanisms...")
        tasks = [
            execution.agent_mechanism_proposal(member_id)
            for member_id in range(len(execution.current_members))
        ]
        proposals = await asyncio.gather(*tasks, return_exceptions=True)
        current_round = execution.execution_history["rounds"][-1] if execution.execution_history["rounds"] else {}
        attempts = current_round.get("mechanism_modifications", {}).get("attempts", [])
        valid_proposals = [
            proposal
            for proposal in attempts
            if isinstance(proposal, dict) and proposal.get("code")
        ]
        if not valid_proposals:
            valid_proposals = [
                proposal
                for proposal in proposals
                if isinstance(proposal, dict) and proposal.get("code")
            ]
        if execution.execution_history["rounds"]:
            execution.execution_history["rounds"][-1]["mechanism_modifications"]["attempts"] = valid_proposals
        print(f"[Propose] Complete: {len(valid_proposals)} proposals generated")
        return {"proposals": valid_proposals, "count": len(valid_proposals)}

    def run_judge(self, execution, context, input_data):
        proposals = input_data.get("proposals") or []
        if not proposals:
            return {"approved": [], "rejected": []}

        print(f"\n[Judge] Evaluating {len(proposals)} proposals...")
        approved = []
        rejected = []

        current_round = execution.execution_history["rounds"][-1] if execution.execution_history["rounds"] else {}
        attempts = current_round.get("mechanism_modifications", {}).get("attempts", [])
        for proposal in attempts:
            code = proposal.get("code", "")
            member_id = proposal.get("member_id", -1)
            if not code:
                continue

            is_approved, reason = execution.judge.judge_proposal(code, member_id, "mechanism")
            if is_approved:
                approved.append(proposal)
                print(f"  ✓ Member {member_id}: APPROVED")
            else:
                rejected.append({"proposal": proposal, "reason": reason})
                print(f"  ✗ Member {member_id}: REJECTED - {reason}")

        print(f"[Judge] Complete: {len(approved)} approved, {len(rejected)} rejected")
        return {"approved": approved, "rejected": rejected}

    def run_execute_mechanisms(self, execution, context, input_data):
        approved = input_data.get("approved") or []
        if not approved:
            print("\n[Mechanisms] No approved mechanisms to execute")
            return {"mechanisms_executed": 0}

        print(f"\n[Mechanisms] Executing {len(approved)} approved mechanisms...")
        execution.execute_mechanism_modifications(approved=approved)
        return {"mechanisms_executed": len(approved)}

    async def run_agent_decisions(self, execution, context, input_data):
        print("\n[Decide] All agents making decisions...")
        tasks = [
            execution.agent_code_decision(member_id)
            for member_id in range(len(execution.current_members))
        ]
        decisions = await asyncio.gather(*tasks, return_exceptions=True)
        valid_decisions = [
            decision
            for decision in decisions
            if decision is not None and not isinstance(decision, Exception)
        ]
        print(f"[Decide] Complete: {len(valid_decisions)} decisions made")
        return {"decisions": valid_decisions, "count": len(valid_decisions)}

    def run_execute_actions(self, execution, context, input_data):
        print("\n[Execute] Executing agent actions with conflict resolution...")
        execution.execute_code_actions()
        return {"actions_executed": True}

    def run_contracts(self, execution, context, input_data):
        if not hasattr(execution, "contracts"):
            return {"contracts_processed": 0}

        print("\n[Contracts] Processing contracts...")
        signed = []
        for contract_id, contract in list(execution.contracts.pending.items()):
            if len(contract.get("signatures", {})) == len(contract.get("parties", [])):
                signed.append(contract_id)

        executed = []
        for contract_id in list(execution.contracts.active.keys()):
            result = execution.contracts.execute_contract(contract_id, execution, context)
            if result.get("status") == "success":
                executed.append(contract_id)

        print(f"[Contracts] {len(signed)} signed, {len(executed)} executed")
        return {
            "contracts_signed": len(signed),
            "contracts_executed": len(executed),
        }

    def run_produce(self, execution, context, input_data):
        execution.produce()
        return {"production_complete": True}

    def run_consume(self, execution, context, input_data):
        execution.consume()
        return {"consumption_complete": True}

    def run_environment(self, execution, context, input_data):
        print("\n[Environment] Applying environmental effects...")
        constraints_applied = 0
        if hasattr(execution, "physics"):
            constraints_applied = len(execution.physics.constraints)
        return {
            "environment_updated": True,
            "constraints_applied": constraints_applied,
        }

    def run_log_status(self, execution, context, input_data):
        print("\n=== Round Summary ===")
        if hasattr(execution, "log_status"):
            execution.log_status(action=True, log_instead_of_print=True)
        if hasattr(execution, "_update_round_end_metrics"):
            execution._update_round_end_metrics()
        if (
            execution.execution_history.get("rounds")
            and hasattr(execution, "contracts")
            and hasattr(execution.contracts, "get_statistics")
        ):
            round_record = execution.execution_history["rounds"][-1]
            round_record["contract_stats"] = execution.contracts.get_statistics()
        if hasattr(execution, "save_execution_history"):
            execution.save_execution_history()
        surviving_members = len(getattr(execution, "current_members", []))
        return {"status_logged": True, "surviving_members": surviving_members}
