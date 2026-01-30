"""
Basic execution nodes for MetaIsland rounds.
"""
from MetaIsland.graph_engine import ExecutionNode


class NewRoundNode(ExecutionNode):
    """Initialize a new round"""

    def __init__(self):
        super().__init__("new_round", "process")

    def execute(self, context, input_data):
        execution = context['execution']
        execution.new_round()
        execution.get_neighbors()

        return {"round_initialized": True}


class ProduceNode(ExecutionNode):
    """Production phase - agents produce resources"""

    def __init__(self):
        super().__init__("produce", "process")

    def execute(self, context, input_data):
        execution = context['execution']
        execution.produce()

        return {"production_complete": True}


class ConsumeNode(ExecutionNode):
    """Consumption phase - agents consume resources"""

    def __init__(self):
        super().__init__("consume", "process")

    def execute(self, context, input_data):
        execution = context['execution']
        execution.consume()

        return {"consumption_complete": True}


class LogStatusNode(ExecutionNode):
    """Log status at end of round"""

    def __init__(self):
        super().__init__("log_status", "process")

    def execute(self, context, input_data):
        execution = context['execution']

        print("\n=== Round Summary ===")
        execution.log_status(action=True, log_instead_of_print=True)

        # Update end-of-round metrics for learning/evaluation
        execution._update_round_end_metrics()

        # Record end-of-round system stats for evaluation
        if execution.execution_history.get('rounds'):
            round_record = execution.execution_history['rounds'][-1]
            if hasattr(execution, 'contracts'):
                round_record['contract_stats'] = execution.contracts.get_statistics()
                partner_stats = execution._collect_contract_partner_stats()
                partner_counts = partner_stats.get("partner_counts", {})
                round_record['contract_partner_counts'] = {
                    party_id: len(partners) for party_id, partners in partner_counts.items()
                }
                round_record['contract_partner_top_share'] = partner_stats.get(
                    "partner_top_share", {}
                )
                round_record['contract_partner_top_partner'] = partner_stats.get(
                    "partner_top_partner", {}
                )
                round_record['contract_status_by_party'] = {
                    party_id: dict(counter)
                    for party_id, counter in partner_stats.get("status_counts", {}).items()
                }

                if isinstance(round_record.get('round_metrics'), dict):
                    stats = round_record['contract_stats']
                    round_record['round_metrics'].update({
                        'contract_total': stats.get('total_contracts', 0),
                        'contract_pending': stats.get('pending', 0),
                        'contract_active': stats.get('active', 0),
                        'contract_completed': stats.get('completed', 0),
                        'contract_failed': stats.get('failed', 0),
                        'contract_partner_unique_avg': partner_stats.get('avg_unique_partners', 0.0),
                        'contract_partner_top_share_avg': partner_stats.get('avg_top_partner_share', 0.0),
                    })
            if hasattr(execution, 'physics'):
                physics_stats = execution.physics.get_statistics()
                round_record['physics_stats'] = physics_stats
                if isinstance(round_record.get('round_metrics'), dict):
                    round_record['round_metrics'].update({
                        'physics_active_constraints': physics_stats.get('active_constraints', 0),
                        'physics_domain_count': len(physics_stats.get('domains', []) or []),
                    })

        # Print survival stats
        print(f"Surviving members: {len(execution.current_members)}")
        for member in execution.current_members:
            survival_chance = execution.compute_survival_chance(member)
            print(f"  Member {member.id}: Vitality={member.vitality:.2f}, "
                  f"Cargo={member.cargo:.2f}, Survival={survival_chance:.2f}")

        execution.save_execution_history()

        return {"status_logged": True}
