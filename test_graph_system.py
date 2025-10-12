"""
Simple test script for the new graph-based execution system.
Tests basic functionality without running full LLM prompts.
"""
import asyncio
import sys
import os

# Add MetaIsland to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MetaIsland.graph_engine import ExecutionGraph, ExecutionNode
from MetaIsland.contracts import ContractEngine
from MetaIsland.physics import PhysicsEngine
from MetaIsland.judge import Judge


class SimpleTestNode(ExecutionNode):
    """Test node that just prints and passes data"""
    def __init__(self, name):
        super().__init__(name, "test")
        self.executed = False

    def execute(self, context, input_data):
        print(f"  [{self.name}] Executing with inputs: {list(input_data.keys())}")
        self.executed = True
        return {f"{self.name}_output": f"Data from {self.name}"}


async def test_graph_execution():
    """Test basic graph execution"""
    print("\n" + "="*60)
    print("TEST 1: Basic Graph Execution")
    print("="*60)

    graph = ExecutionGraph()

    # Create simple test nodes
    node1 = SimpleTestNode("node1")
    node2 = SimpleTestNode("node2")
    node3 = SimpleTestNode("node3")

    # Add nodes
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    # Connect: node1 -> node2 -> node3
    graph.connect("node1", "node2")
    graph.connect("node2", "node3")

    print("\nGraph structure:")
    print(graph.visualize())

    # Execute
    print("\nExecuting graph...")
    graph.context = {"test": "data"}
    await graph.execute_round()

    # Verify
    assert node1.executed, "Node1 should have executed"
    assert node2.executed, "Node2 should have executed"
    assert node3.executed, "Node3 should have executed"

    print("\n✓ Basic graph execution works!")


def test_contract_engine():
    """Test contract system"""
    print("\n" + "="*60)
    print("TEST 2: Contract Engine")
    print("="*60)

    contracts = ContractEngine()

    # Propose a simple contract
    contract_code = """
def execute_contract(execution_engine, context):
    return {"status": "executed", "result": "simple trade"}
"""

    contract_id = contracts.propose_contract(
        code=contract_code,
        proposer_id=0,
        parties=[0, 1],
        terms={"type": "trade", "goods": "food"}
    )

    print(f"\nContract proposed: {contract_id}")
    print(f"Status: {contracts.get_statistics()}")

    # Sign contract
    contracts.sign_contract(contract_id, 1)
    print(f"\nContract signed by party 1")
    print(f"Status: {contracts.get_statistics()}")

    # Execute contract
    result = contracts.execute_contract(contract_id, execution_engine=None)
    print(f"\nContract executed: {result}")

    print("\n✓ Contract engine works!")


def test_physics_engine():
    """Test physics constraints"""
    print("\n" + "="*60)
    print("TEST 3: Physics Engine")
    print("="*60)

    physics = PhysicsEngine()

    # Propose a constraint
    constraint_code = """
class TestConstraint:
    def __init__(self):
        self.name = "test_constraint"

    def apply_constraint(self, action, execution_engine):
        # Modify action based on physics
        action['constrained'] = True
        return action

def apply_constraint(action, execution_engine):
    action['constrained'] = True
    return action
"""

    constraint_id = physics.propose_constraint(
        code=constraint_code,
        proposer_id=0,
        domain="test",
        description="Simple test constraint"
    )

    print(f"\nConstraint proposed: {constraint_id}")
    print(f"Status: {physics.get_statistics()}")

    # Approve constraint
    physics.approve_constraint(constraint_id)
    print(f"\nConstraint approved")
    print(f"Status: {physics.get_statistics()}")

    # Test applying constraints
    action = {"type": "test_action"}
    modified = physics.apply_constraints(action, None)
    print(f"\nAction modified: {modified}")

    assert modified.get('constrained'), "Constraint should have been applied"

    print("\n✓ Physics engine works!")


def test_judge_basic():
    """Test judge system (basic, without actual LLM call)"""
    print("\n" + "="*60)
    print("TEST 4: Judge System (Structure Only)")
    print("="*60)

    judge = Judge()

    print(f"\nJudge initialized: {judge}")
    print(f"Statistics: {judge.get_statistics()}")

    print("\n✓ Judge system structure works!")
    print("  (Note: Full judge testing requires LLM API)")


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" TESTING NEW METAISLAND GRAPH SYSTEM")
    print("="*70)

    try:
        await test_graph_execution()
        test_contract_engine()
        test_physics_engine()
        test_judge_basic()

        print("\n" + "="*70)
        print(" ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nThe new graph-based system is working correctly.")
        print("Next steps:")
        print("  1. Update agent prompts to include contract/physics coding")
        print("  2. Run full simulation with: python MetaIsland/metaIsland.py")
        print("  3. Monitor judge decisions and agent adaptations")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
