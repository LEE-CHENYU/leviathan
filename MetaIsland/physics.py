"""
Physics engine for agent-defined physical constraints and laws.
Agents propose constraints that govern resource extraction, transformation, etc.
"""
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import traceback


class PhysicsEngine:
    """Engine for managing agent-defined physical constraints"""

    def __init__(self):
        self.constraints: List[Dict] = []  # Active constraints
        self.proposed_constraints: List[Dict] = []  # Pending approval
        self.constraint_history: List[Dict] = []  # All constraint events

    def propose_constraint(self, code: str, proposer_id: int,
                          domain: str, description: str) -> str:
        """
        Agent proposes a physical constraint for their domain.

        Args:
            code: Python code defining the constraint
            proposer_id: ID of the proposing agent
            domain: Domain this constraint applies to (e.g., 'agriculture', 'manufacturing')
            description: Human-readable description

        Returns:
            constraint_id: Unique identifier for this constraint
        """
        constraint_id = f"constraint_{proposer_id}_{len(self.constraints)}_{datetime.now().timestamp()}"

        constraint = {
            'id': constraint_id,
            'code': code,
            'proposer': proposer_id,
            'domain': domain,
            'description': description,
            'status': 'proposed',
            'created_at': datetime.now().isoformat(),
            'votes': {'approve': [], 'reject': []},
            'applied': False
        }

        self.proposed_constraints.append(constraint)

        self._log_event({
            'type': 'constraint_proposed',
            'constraint_id': constraint_id,
            'proposer': proposer_id,
            'domain': domain,
            'timestamp': datetime.now().isoformat()
        })

        return constraint_id

    def approve_constraint(self, constraint_id: str) -> bool:
        """
        Approve a proposed constraint (typically done by judge).

        Args:
            constraint_id: Constraint to approve

        Returns:
            True if successful
        """
        constraint = self._find_constraint(constraint_id, self.proposed_constraints)
        if not constraint:
            return False

        # Try to instantiate the constraint
        try:
            exec_env = {'__builtins__': __builtins__}
            exec(constraint['code'], exec_env)

            # Store the constraint class or function
            constraint['status'] = 'active'
            constraint['applied'] = True
            constraint['activated_at'] = datetime.now().isoformat()
            constraint['exec_env'] = exec_env

            # Move to active constraints
            self.constraints.append(constraint)
            self.proposed_constraints.remove(constraint)

            self._log_event({
                'type': 'constraint_approved',
                'constraint_id': constraint_id,
                'timestamp': datetime.now().isoformat()
            })

            print(f"Constraint {constraint_id} approved and activated")
            return True

        except Exception as e:
            print(f"Error activating constraint {constraint_id}: {e}")
            constraint['status'] = 'failed'
            constraint['error'] = str(e)
            return False

    def reject_constraint(self, constraint_id: str, reason: str) -> bool:
        """
        Reject a proposed constraint.

        Args:
            constraint_id: Constraint to reject
            reason: Reason for rejection

        Returns:
            True if successful
        """
        constraint = self._find_constraint(constraint_id, self.proposed_constraints)
        if not constraint:
            return False

        constraint['status'] = 'rejected'
        constraint['rejected_at'] = datetime.now().isoformat()
        constraint['rejection_reason'] = reason

        self.proposed_constraints.remove(constraint)

        self._log_event({
            'type': 'constraint_rejected',
            'constraint_id': constraint_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })

        print(f"Constraint {constraint_id} rejected: {reason}")
        return True

    def apply_constraints(self, action: Dict, execution_engine: Any) -> Dict:
        """
        Apply all active constraints to an action.

        Args:
            action: Action to constrain
            execution_engine: Reference to main execution engine

        Returns:
            Modified action after applying constraints
        """
        modified_action = action.copy()

        for constraint in self.constraints:
            if not constraint.get('applied'):
                continue

            try:
                # Check if constraint has an apply function
                exec_env = constraint.get('exec_env', {})

                if 'apply_constraint' in exec_env and callable(exec_env['apply_constraint']):
                    modified_action = exec_env['apply_constraint'](
                        modified_action, execution_engine
                    )

            except Exception as e:
                print(f"Error applying constraint {constraint['id']}: {e}")
                traceback.print_exc()

        return modified_action

    def check_conservation(self, before_state: Dict, after_state: Dict) -> tuple[bool, str]:
        """
        Check basic conservation laws.

        Args:
            before_state: State before action
            after_state: State after action

        Returns:
            (is_valid, message): Whether conservation is maintained and explanation
        """
        # Basic resource conservation check
        # Allow some increase due to production, but not arbitrary amounts
        tolerance = 1.2  # Allow 20% increase from production

        try:
            total_before = sum(
                member.get('cargo', 0) + member.get('vitality', 0)
                for member in before_state.get('members', [])
            )

            total_after = sum(
                member.get('cargo', 0) + member.get('vitality', 0)
                for member in after_state.get('members', [])
            )

            if total_after > total_before * tolerance:
                return False, f"Violates conservation: {total_before} -> {total_after} (>{tolerance}x increase)"

            return True, "Conservation maintained"

        except Exception as e:
            return True, f"Could not verify conservation: {e}"

    def get_constraints_by_domain(self, domain: str) -> List[Dict]:
        """Get all active constraints for a specific domain"""
        return [c for c in self.constraints if c.get('domain') == domain]

    def _find_constraint(self, constraint_id: str, constraint_list: List[Dict]) -> Optional[Dict]:
        """Find a constraint by ID in a list"""
        for constraint in constraint_list:
            if constraint['id'] == constraint_id:
                return constraint
        return None

    def _log_event(self, event: Dict) -> None:
        """Log a constraint event to history"""
        self.constraint_history.append(event)

    def get_statistics(self) -> Dict:
        """Get physics engine statistics"""
        return {
            'active_constraints': len(self.constraints),
            'proposed_constraints': len(self.proposed_constraints),
            'total_events': len(self.constraint_history),
            'domains': list(set(c.get('domain') for c in self.constraints))
        }

    def __repr__(self):
        stats = self.get_statistics()
        return (f"<PhysicsEngine: {stats['active_constraints']} active constraints, "
                f"{stats['proposed_constraints']} proposed>")
