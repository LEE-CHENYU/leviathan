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

        # Print survival stats
        print(f"Surviving members: {len(execution.current_members)}")
        for member in execution.current_members:
            survival_chance = execution.compute_survival_chance(member)
            print(f"  Member {member.id}: Vitality={member.vitality:.2f}, "
                  f"Cargo={member.cargo:.2f}, Survival={survival_chance:.2f}")

        return {"status_logged": True}
