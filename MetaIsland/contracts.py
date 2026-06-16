"""
Contract system for agent-to-agent agreements.
Agents write contract code that gets executed when conditions are met.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback
import hashlib


class ContractEngine:
    """Engine for managing and executing agent-written contracts"""

    def __init__(self):
        self.contracts: Dict[str, Dict] = {}
        self.pending: Dict[str, Dict] = {}  # Awaiting signatures
        self.active: Dict[str, Dict] = {}  # Being executed
        self.completed: List[Dict] = []  # Finished contracts
        self.failed: List[Dict] = []  # Failed contracts
        self.history: List[Dict] = []  # All contract events

    def generate_contract_id(self, code: str, proposer_id: int, parties: List[int]) -> str:
        """Generate unique contract ID"""
        content = f"{code}{proposer_id}{sorted(parties)}{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def propose_contract(self, code: str, proposer_id: int, parties: List[int],
                        terms: Dict[str, Any]) -> str:
        """
        Agent proposes a contract to other parties.

        Args:
            code: Python code defining the contract
            proposer_id: ID of the proposing agent
            parties: List of all party IDs (including proposer)
            terms: Dictionary describing contract terms

        Returns:
            contract_id: Unique identifier for this contract
        """
        contract_id = self.generate_contract_id(code, proposer_id, parties)

        contract = {
            'id': contract_id,
            'code': code,
            'proposer': proposer_id,
            'parties': parties,
            'terms': terms,
            'signatures': {proposer_id: True},
            'created_at': datetime.now().isoformat(),
            'status': 'pending',
            'execution_count': 0,
            'last_executed': None
        }

        self.pending[contract_id] = contract
        self.contracts[contract_id] = contract

        # Log event
        self._log_event({
            'type': 'contract_proposed',
            'contract_id': contract_id,
            'proposer': proposer_id,
            'parties': parties,
            'timestamp': datetime.now().isoformat()
        })

        return contract_id

    def sign_contract(self, contract_id: str, party_id: int) -> bool:
        """
        Party signs a pending contract.

        Args:
            contract_id: Contract to sign
            party_id: ID of the signing party

        Returns:
            True if signing successful, False otherwise
        """
        if contract_id not in self.pending:
            print(f"Contract {contract_id} not found in pending")
            return False

        contract = self.pending[contract_id]

        # Verify party is part of contract
        if party_id not in contract['parties']:
            print(f"Party {party_id} not authorized to sign contract {contract_id}")
            return False

        # Add signature
        contract['signatures'][party_id] = True

        # Log event
        self._log_event({
            'type': 'contract_signed',
            'contract_id': contract_id,
            'party_id': party_id,
            'timestamp': datetime.now().isoformat()
        })

        # Check if all parties have signed
        if len(contract['signatures']) == len(contract['parties']):
            # Move to active
            contract['status'] = 'active'
            contract['activated_at'] = datetime.now().isoformat()
            self.active[contract_id] = contract
            del self.pending[contract_id]

            self._log_event({
                'type': 'contract_activated',
                'contract_id': contract_id,
                'timestamp': datetime.now().isoformat()
            })

            print(f"Contract {contract_id} fully signed and activated")

        return True

    def execute_contract(self, contract_id: str, execution_engine: Any,
                        context: Optional[Dict] = None) -> Dict:
        """
        Execute an active contract.

        Args:
            contract_id: Contract to execute
            execution_engine: Reference to main execution engine
            context: Additional context for execution

        Returns:
            Result dictionary with status and any outputs
        """
        if contract_id not in self.active:
            return {'status': 'error', 'message': 'Contract not active'}

        contract = self.active[contract_id]
        context = context or {}

        result = {
            'contract_id': contract_id,
            'status': 'success',
            'outputs': None,
            'error': None
        }

        try:
            # Create execution environment
            exec_env = {
                'execution_engine': execution_engine,
                'contract': contract,
                'context': context,
                '__builtins__': __builtins__
            }

            # Execute contract code
            exec(contract['code'], exec_env)

            # Check if contract defines an execute function
            if 'execute_contract' in exec_env and callable(exec_env['execute_contract']):
                output = exec_env['execute_contract'](execution_engine, context)
                result['outputs'] = output
            else:
                result['status'] = 'warning'
                result['message'] = 'No execute_contract function defined'

            # Update execution stats
            contract['execution_count'] += 1
            contract['last_executed'] = datetime.now().isoformat()

            # Log successful execution
            self._log_event({
                'type': 'contract_executed',
                'contract_id': contract_id,
                'timestamp': datetime.now().isoformat(),
                'result': result
            })

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()

            # Move to failed
            contract['status'] = 'failed'
            contract['error'] = str(e)
            self.failed.append(contract)
            del self.active[contract_id]

            print(f"Contract {contract_id} failed: {e}")
            self._log_event({
                'type': 'contract_failed',
                'contract_id': contract_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

        return result

    def complete_contract(self, contract_id: str, reason: str = "fulfilled") -> bool:
        """
        Mark a contract as completed.

        Args:
            contract_id: Contract to complete
            reason: Reason for completion

        Returns:
            True if successful
        """
        if contract_id not in self.active:
            return False

        contract = self.active[contract_id]
        contract['status'] = 'completed'
        contract['completed_at'] = datetime.now().isoformat()
        contract['completion_reason'] = reason

        self.completed.append(contract)
        del self.active[contract_id]

        self._log_event({
            'type': 'contract_completed',
            'contract_id': contract_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })

        return True

    def cancel_contract(self, contract_id: str, party_id: int, reason: str) -> bool:
        """
        Cancel a pending or active contract.

        Args:
            contract_id: Contract to cancel
            party_id: Party requesting cancellation
            reason: Reason for cancellation

        Returns:
            True if cancellation successful
        """
        contract = None
        if contract_id in self.pending:
            contract = self.pending[contract_id]
            del self.pending[contract_id]
        elif contract_id in self.active:
            contract = self.active[contract_id]
            del self.active[contract_id]
        else:
            return False

        # Verify party is authorized
        if party_id not in contract['parties']:
            return False

        contract['status'] = 'cancelled'
        contract['cancelled_at'] = datetime.now().isoformat()
        contract['cancelled_by'] = party_id
        contract['cancellation_reason'] = reason

        self.failed.append(contract)

        self._log_event({
            'type': 'contract_cancelled',
            'contract_id': contract_id,
            'party_id': party_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })

        return True

    def get_contracts_for_party(self, party_id: int, status: Optional[str] = None) -> List[Dict]:
        """Get all contracts involving a specific party"""
        results = []

        sources = []
        if status is None or status == 'pending':
            sources.append(self.pending)
        if status is None or status == 'active':
            sources.append(self.active)

        for source in sources:
            for contract in source.values():
                if party_id in contract['parties']:
                    results.append(contract)

        return results

    def _log_event(self, event: Dict) -> None:
        """Log a contract event to history"""
        self.history.append(event)

    def get_statistics(self) -> Dict:
        """Get contract system statistics"""
        return {
            'total_contracts': len(self.contracts),
            'pending': len(self.pending),
            'active': len(self.active),
            'completed': len(self.completed),
            'failed': len(self.failed),
            'events': len(self.history)
        }

    def __repr__(self):
        stats = self.get_statistics()
        return (f"<ContractEngine: {stats['active']} active, "
                f"{stats['pending']} pending, {stats['completed']} completed>")
