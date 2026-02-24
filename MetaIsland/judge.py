"""
Judge system for validating agent-proposed code.
Prevents unrealistic physics, reward hacking, and exploits.
"""
from typing import Tuple, Dict, Any
from datetime import datetime

from MetaIsland.llm_client import get_llm_client
from MetaIsland.model_router import model_router
from MetaIsland.llm_utils import build_chat_kwargs


class Judge:
    """Simple LLM judge for validating agent code"""

    def __init__(self, model_name: str = "default"):
        self.client = get_llm_client()
        self.provider, self.model_id = model_router(model_name)
        self.judgment_history = []

    def judge_proposal(self, code: str, proposer_id: int,
                      proposal_type: str, context: Dict = None) -> Tuple[bool, str]:
        """
        Judge whether a code proposal should be approved.

        Args:
            code: The proposed code
            proposer_id: ID of the agent proposing
            proposal_type: Type of proposal ('mechanism', 'action', 'constraint')
            context: Additional context for judging

        Returns:
            (approved, reason): Whether approved and explanation
        """
        context = context or {}

        judge_prompt = self._build_judge_prompt(code, proposer_id, proposal_type, context)

        try:
            kwargs = build_chat_kwargs()
            kwargs["temperature"] = 0  # Deterministic judging
            response = self.client.chat.completions.create(
                model=f'{self.provider}:{self.model_id}',
                messages=[{"role": "user", "content": judge_prompt}],
                **kwargs
            )

            result = response.choices[0].message.content.strip()

            # Parse result
            approved = result.upper().startswith("APPROVE")
            reason = result.split(":", 1)[1].strip() if ":" in result else result

            # Log judgment
            self._log_judgment({
                'proposer_id': proposer_id,
                'proposal_type': proposal_type,
                'approved': approved,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            })

            return approved, reason

        except Exception as e:
            print(f"Error in judge: {e}")
            # Default to rejecting on error
            return False, f"Judge error: {str(e)}"

    def _build_judge_prompt(self, code: str, proposer_id: int,
                           proposal_type: str, context: Dict) -> str:
        """Build the judge prompt based on proposal type"""

        base_checks = """
CHECK FOR VIOLATIONS:
1. Unrealistic physics (e.g., creating energy/resources from nothing)
2. Reward hacking (e.g., giving self free resources without cost)
3. Breaking other agents unfairly (e.g., arbitrary theft without mechanism)
4. Infinite loops or exploits
5. Violations of conservation laws

APPROVE if code:
- Follows conservation laws (energy, mass, resources)
- Has realistic costs and tradeoffs
- Respects other agents' autonomy
- Makes economic sense
- Has legitimate transformations (inputs → outputs)

REJECT if code:
- Creates resources from nothing
- Has no costs or constraints
- Unfairly advantages one agent
- Exploits system mechanics
- Violates basic physics
"""

        if proposal_type == 'mechanism':
            specific_guidance = """
This is a MECHANISM PROPOSAL - code that modifies game rules.

Additional checks for mechanisms:
- Does it create fair rules that apply to all?
- Are there proper constraints and limits?
- Does it enable new interactions without breaking existing ones?
- Is it well-defined and not ambiguous?
"""
        elif proposal_type == 'action':
            specific_guidance = """
This is an AGENT ACTION - code for a single agent's turn.

Additional checks for actions:
- Does agent try to modify themselves without cost?
- Are all resource transfers legitimate?
- Does it respect ownership and permissions?
- Are all method calls to valid interfaces?
"""
        elif proposal_type == 'constraint':
            specific_guidance = """
This is a PHYSICS CONSTRAINT - code defining physical laws.

Additional checks for constraints:
- Is it realistic for the domain?
- Does it have diminishing returns where appropriate?
- Are there limits and bounds?
- Does it model scarcity correctly?
"""
        else:
            specific_guidance = ""

        prompt = f"""
You are judging agent code for a realistic economic simulation.

PROPOSAL TYPE: {proposal_type}
PROPOSER: Agent {proposer_id}

CODE TO JUDGE:
```python
{code}
```

{base_checks}

{specific_guidance}

Reply with ONLY one of:
APPROVE: [brief reason]
REJECT: [specific violation found]
"""
        return prompt

    def judge_batch(self, proposals: list) -> list:
        """
        Judge multiple proposals efficiently.

        Args:
            proposals: List of (code, proposer_id, type, context) tuples

        Returns:
            List of (approved, reason) tuples
        """
        results = []
        for code, proposer_id, prop_type, context in proposals:
            approved, reason = self.judge_proposal(code, proposer_id, prop_type, context)
            results.append((approved, reason))
        return results

    def _log_judgment(self, judgment: Dict) -> None:
        """Log a judgment to history"""
        self.judgment_history.append(judgment)

    def get_statistics(self) -> Dict:
        """Get judge statistics"""
        total = len(self.judgment_history)
        if total == 0:
            return {
                'total_judgments': 0,
                'approved': 0,
                'rejected': 0,
                'approval_rate': 0.0
            }

        approved = sum(1 for j in self.judgment_history if j['approved'])
        rejected = total - approved

        return {
            'total_judgments': total,
            'approved': approved,
            'rejected': rejected,
            'approval_rate': approved / total if total > 0 else 0.0
        }

    def get_recent_rejections(self, limit: int = 5) -> list:
        """Get recent rejection reasons for agent learning"""
        rejections = [
            j for j in self.judgment_history
            if not j['approved']
        ]
        return rejections[-limit:]

    def __repr__(self):
        stats = self.get_statistics()
        return (f"<Judge: {stats['approved']}/{stats['total_judgments']} approved "
                f"({stats['approval_rate']:.1%})>")
