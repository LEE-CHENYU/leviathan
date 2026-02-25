"""Canary testing and agent review nodes for the graph engine.

CanaryNode runs mechanism proposals against a state clone.
AgentReviewNode collects votes across rounds and activates approved mechanisms.
"""

from MetaIsland.graph_engine import ExecutionNode
from kernel.canary import CanaryRunner


class CanaryNode(ExecutionNode):
    """Run canary tests on all proposed mechanisms."""

    def __init__(self):
        super().__init__("canary", "process")

    def execute(self, context, input_data):
        execution = context["execution"]
        proposals = input_data.get("proposals", [])

        if not proposals:
            return {"canary_reports": [], "proposals": []}

        print(f"\n[Canary] Testing {len(proposals)} proposals...")

        runner = CanaryRunner()
        judge = getattr(execution, "judge", None)
        reports = []

        for idx, proposal in enumerate(proposals):
            code = proposal.get("code", "")
            member_id = proposal.get("member_id", -1)
            proposal_id = proposal.get("proposal_id", f"prop_{member_id}_{idx}")

            if not code:
                continue

            report = runner.run_canary(
                execution_engine=execution,
                mechanism_code=code,
                proposal_id=proposal_id,
                proposer_id=member_id,
                judge=judge,
            )
            reports.append(report)

            # Annotate proposal with canary results
            proposal["canary_report"] = report.to_dict()

            if report.execution_error:
                print(f"  ! Proposal {proposal_id} (Member {member_id}): ERROR")
            elif report.divergence_flags:
                print(
                    f"  ~ Proposal {proposal_id} (Member {member_id}): "
                    f"FLAGGED {report.divergence_flags}"
                )
            else:
                print(
                    f"  + Proposal {proposal_id} (Member {member_id}): "
                    f"CLEAN (vitality {report.vitality_change_pct:+.1f}%)"
                )

        # Store canary reports in execution history
        if execution.execution_history.get("rounds"):
            current_round = execution.execution_history["rounds"][-1]
            mods_record = current_round.get("mechanism_modifications")
            if isinstance(mods_record, dict):
                mods_record["canary_reports"] = [r.to_dict() for r in reports]

        print(f"[Canary] Complete: {len(reports)} tested")

        return {"canary_reports": [r.to_dict() for r in reports], "proposals": proposals}


class AgentReviewNode(ExecutionNode):
    """Agents review canary results and vote on pending mechanisms.

    Maintains a pending proposals queue across rounds. A proposal enters the
    queue when its canary runs. It leaves when either (a) majority votes yes,
    (b) majority votes no, or (c) proposer withdraws.
    """

    def __init__(self):
        super().__init__("agent_review", "decision")

    def execute(self, context, input_data):
        execution = context["execution"]
        proposals = input_data.get("proposals", [])

        # Initialize pending proposals queue on execution engine if needed
        if not hasattr(execution, "pending_proposals"):
            execution.pending_proposals = {}  # proposal_id -> {proposal, votes, round_submitted}

        # Add new proposals to the pending queue
        for proposal in proposals:
            proposal_id = proposal.get("proposal_id", f"prop_{proposal.get('member_id', '?')}")
            if proposal_id not in execution.pending_proposals:
                execution.pending_proposals[proposal_id] = {
                    "proposal": proposal,
                    "votes": {},  # member_id -> True/False
                    "round_submitted": context.get("round", 0),
                }

        # Count living agents for majority calculation
        living_count = len(execution.current_members)
        majority_threshold = (living_count // 2) + 1

        approved = []
        rejected = []
        still_pending = {}

        for prop_id, entry in execution.pending_proposals.items():
            votes = entry["votes"]
            proposal = entry["proposal"]

            # Auto-vote: In the current implementation without LLM agent voting,
            # proposals without canary errors and without divergence flags pass
            # with automatic approval. This default can be overridden when the
            # full voting system is active.
            canary_report = proposal.get("canary_report", {})
            has_error = canary_report.get("execution_error") is not None
            has_divergence = bool(canary_report.get("divergence_flags"))

            if has_error:
                # Auto-reject proposals that failed canary
                rejected.append({"proposal": proposal, "reason": "Canary execution error"})
                continue

            if not votes:
                # Default voting behavior: approve unless divergent
                if has_divergence:
                    # Flagged proposals stay pending (need explicit votes)
                    still_pending[prop_id] = entry
                    continue
                else:
                    # Clean canary -> auto-approve as default behavior
                    approved.append(proposal)
                    continue

            # Count actual votes
            yes_count = sum(1 for v in votes.values() if v)
            no_count = sum(1 for v in votes.values() if not v)

            if yes_count >= majority_threshold:
                approved.append(proposal)
            elif no_count >= majority_threshold:
                rejected.append({"proposal": proposal, "reason": "Rejected by majority vote"})
            else:
                still_pending[prop_id] = entry

        execution.pending_proposals = still_pending

        # Store review results in execution history
        if execution.execution_history.get("rounds"):
            current_round = execution.execution_history["rounds"][-1]
            mods_record = current_round.get("mechanism_modifications")
            if isinstance(mods_record, dict):
                mods_record["review_results"] = {
                    "approved_count": len(approved),
                    "rejected_count": len(rejected),
                    "still_pending_count": len(still_pending),
                }

        print(
            f"\n[Review] {len(approved)} approved, {len(rejected)} rejected, "
            f"{len(still_pending)} still pending"
        )

        return {"approved": approved, "rejected": rejected}
