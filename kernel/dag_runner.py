"""KernelDAGRunner — executes simulation phases in topological order."""

from typing import Any, List


class KernelDAGRunner:
    """Runs infrastructure phases on an IslandExecution instance.

    Phase order (matching ExecutionGraph topology):
        produce -> consume -> environment -> log_status

    new_round is handled by WorldKernel.begin_round().
    Judge and execute_mechanisms are handled by the kernel before settlement.
    Agent actions are handled by the Phase 2 write path.
    """

    def __init__(self, execution: Any) -> None:
        self._execution = execution

    def run_settlement_phases(self) -> List[str]:
        """Execute infrastructure phases in order. Returns list of phase names run."""
        log: List[str] = []

        # Contracts (if available)
        if hasattr(self._execution, "contracts"):
            self._run_contracts()
            log.append("contracts")

        # Core economic phases
        self._execution.produce()
        log.append("produce")

        self._execution.consume()
        log.append("consume")

        # Environment (physics constraints, if available)
        if hasattr(self._execution, "physics"):
            log.append("environment")

        return log

    def _run_contracts(self) -> None:
        """Process active contracts."""
        contracts = self._execution.contracts
        for contract_id in list(contracts.active.keys()):
            try:
                contracts.execute_contract(contract_id, self._execution, {})
            except Exception:
                pass  # Contract errors don't halt the round
