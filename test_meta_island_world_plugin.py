from __future__ import annotations

import asyncio
from types import SimpleNamespace

from MetaIsland.graph_engine import ExecutionGraph
from MetaIsland.world_plugin import MetaIslandWorldPlugin


class FakeExecution:
    def __init__(self):
        self.current_members = [
            SimpleNamespace(id=0, vitality=10.0, cargo=5.0),
            SimpleNamespace(id=1, vitality=8.0, cargo=6.0),
        ]
        self.execution_history = {"rounds": []}
        self.contracts = SimpleNamespace(
            pending={},
            active={},
            execute_contract=lambda contract_id, execution, context: {"status": "success"},
        )
        self.physics = SimpleNamespace(constraints=["scarcity", "conservation"])
        self.judge_calls = []
        self.judge = SimpleNamespace(judge_proposal=self._judge_proposal)
        self.new_round_calls = 0
        self.neighbor_calls = 0
        self.actions_executed = 0
        self.mechanisms_executed = 0
        self.production_calls = 0
        self.consumption_calls = 0
        self.status_logged = 0

    def _judge_proposal(self, code, proposer_id, proposal_type):
        self.judge_calls.append((proposer_id, proposal_type))
        return True, "ok"

    def new_round(self):
        self.new_round_calls += 1
        self.execution_history["rounds"].append(
            {
                "round_number": len(self.execution_history["rounds"]) + 1,
                "analysis": {},
                "agent_actions": [],
                "agent_messages": {},
                "mechanism_modifications": {
                    "attempts": [],
                    "executed": [],
                },
                "errors": {
                    "agent_code_errors": [],
                    "mechanism_errors": [],
                    "analyze_code_errors": {},
                },
            }
        )

    def get_neighbors(self):
        self.neighbor_calls += 1

    async def analyze(self, member_id):
        return {"member_id": member_id, "analysis": "ok"}

    async def agent_mechanism_proposal(self, member_id):
        return {
            "member_id": member_id,
            "code": "def propose_modification(self):\n    return None\n",
        }

    async def agent_code_decision(self, member_id):
        return {"member_id": member_id, "decision": "wait"}

    def execute_code_actions(self):
        self.actions_executed += 1

    def execute_mechanism_modifications(self):
        self.mechanisms_executed += 1

    def produce(self):
        self.production_calls += 1

    def consume(self):
        self.consumption_calls += 1

    def log_status(self, action=True, log_instead_of_print=True):
        self.status_logged += 1


def test_meta_island_world_plugin_runs_one_graph_round():
    graph = ExecutionGraph()
    plugin = MetaIslandWorldPlugin()
    plugin.build_default_graph(graph)
    execution = FakeExecution()
    graph.context = plugin.build_graph_context(execution, 1)

    asyncio.run(graph.execute_round())

    assert graph.get_execution_order()[0] == ["new_round"]
    assert "propose_mechanisms" in graph.nodes
    assert "agent_decisions" in graph.nodes

    assert execution.new_round_calls == 1
    assert execution.neighbor_calls == 1
    assert len(execution.execution_history["rounds"]) == 1
    assert len(execution.execution_history["rounds"][-1]["mechanism_modifications"]["attempts"]) == 2
    assert len(execution.judge_calls) == 2
    assert execution.mechanisms_executed == 1
    assert execution.actions_executed == 1
    assert execution.production_calls == 1
    assert execution.consumption_calls == 1
    assert execution.status_logged == 1

    judge_outputs = graph.nodes["judge"].outputs
    assert len(judge_outputs["approved"]) == 2
    assert graph.nodes["execute_mechanisms"].outputs["mechanisms_executed"] == 2
