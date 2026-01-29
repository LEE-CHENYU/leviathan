"""
System-level execution nodes (judge, contracts, environment).
"""
from MetaIsland.graph_engine import ExecutionNode


class JudgeNode(ExecutionNode):
    """Judge evaluates proposals"""

    def __init__(self):
        super().__init__("judge", "decision")

    def execute(self, context, input_data):
        execution = context['execution']
        proposals = input_data.get("proposals", [])

        if not proposals:
            return {"approved": [], "rejected": []}

        print(f"\n[Judge] Evaluating {len(proposals)} proposals...")

        approved = []
        rejected = []

        # Get the current round's mechanism proposals
        if execution.execution_history['rounds']:
            current_round = execution.execution_history['rounds'][-1]
            attempts = current_round.get('mechanism_modifications', {}).get('attempts', [])

            for proposal in attempts:
                code = proposal.get('code', '')
                member_id = proposal.get('member_id', -1)

                if not code:
                    continue

                # Judge the proposal
                is_approved, reason = execution.judge.judge_proposal(
                    code, member_id, "mechanism"
                )

                if is_approved:
                    approved.append(proposal)
                    print(f"  ✓ Member {member_id}: APPROVED")
                else:
                    rejected.append({"proposal": proposal, "reason": reason})
                    print(f"  ✗ Member {member_id}: REJECTED - {reason}")

        print(f"[Judge] Complete: {len(approved)} approved, {len(rejected)} rejected")

        return {"approved": approved, "rejected": rejected}


class ExecuteMechanismsNode(ExecutionNode):
    """Execute approved mechanism modifications"""

    def __init__(self):
        super().__init__("execute_mechanisms", "process")

    def execute(self, context, input_data):
        execution = context['execution']
        approved = input_data.get("approved", [])

        if not approved:
            print("\n[Mechanisms] No approved mechanisms to execute")
            return {"mechanisms_executed": 0}

        if execution.execution_history.get('rounds'):
            round_record = execution.execution_history['rounds'][-1]
            mods_record = round_record.get('mechanism_modifications')
            if isinstance(mods_record, dict):
                mods_record['approved_ids'] = [
                    mod.get('member_id') for mod in approved if isinstance(mod, dict)
                ]
                mods_record['approved_count'] = len(approved)

        print(f"\n[Mechanisms] Executing {len(approved)} approved mechanisms...")
        execution.execute_mechanism_modifications(approved=approved)

        return {"mechanisms_executed": len(approved)}


class ContractNode(ExecutionNode):
    """Process and execute contracts"""

    def __init__(self):
        super().__init__("contracts", "process")

    def execute(self, context, input_data):
        execution = context['execution']

        if not hasattr(execution, 'contracts'):
            return {"contracts_processed": 0}

        print("\n[Contracts] Processing contracts...")

        # Sign pending contracts (agents would have requested signatures in their actions)
        signed = []
        for contract_id, contract in list(execution.contracts.pending.items()):
            # Auto-sign would happen through agent actions
            # This just checks for fully signed contracts
            if len(contract.get('signatures', {})) == len(contract.get('parties', [])):
                signed.append(contract_id)

        # Execute active contracts
        executed = []
        for contract_id in list(execution.contracts.active.keys()):
            result = execution.contracts.execute_contract(
                contract_id, execution, context
            )
            if result.get('status') == 'success':
                executed.append(contract_id)

        print(f"[Contracts] {len(signed)} signed, {len(executed)} executed")

        return {
            "contracts_signed": len(signed),
            "contracts_executed": len(executed)
        }


class EnvironmentNode(ExecutionNode):
    """Apply environmental effects and constraints"""

    def __init__(self):
        super().__init__("environment", "process")

    def execute(self, context, input_data):
        execution = context['execution']

        print("\n[Environment] Applying environmental effects...")

        # Apply physics constraints if they exist
        constraints_applied = 0
        if hasattr(execution, 'physics'):
            constraints_applied = len(execution.physics.constraints)

        return {
            "environment_updated": True,
            "constraints_applied": constraints_applied
        }
