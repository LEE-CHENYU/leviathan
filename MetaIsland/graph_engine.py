"""
Graph-based execution engine for MetaIsland.
Allows flexible, modular execution flow with parallel node execution.
"""
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict
import inspect


class ExecutionNode:
    """Base class for all execution nodes in the graph"""

    def __init__(self, name: str, node_type: str = "process"):
        self.name = name
        self.node_type = node_type  # process, decision, storage
        self.inputs = {}  # {input_name: (source_node, output_key)}
        self.outputs = {}  # {output_name: data}
        self.next_nodes = []  # List of next nodes to execute
        self.enabled = True

    def execute(self, context: Dict, input_data: Dict) -> Dict:
        """
        Override this in subclasses. Can be sync or async.

        Args:
            context: Shared context across all nodes
            input_data: Data from connected input nodes

        Returns:
            Dictionary of outputs from this node
        """
        raise NotImplementedError(f"Node {self.name} must implement execute()")

    def can_execute(self) -> bool:
        """Check if all required inputs are available and node is enabled"""
        return self.enabled

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})>"


class ExecutionGraph:
    """Main graph execution engine with support for parallel execution"""

    def __init__(self):
        self.nodes: Dict[str, ExecutionNode] = {}
        self.execution_layers: List[List[ExecutionNode]] = []
        self.context: Dict = {}

    def add_node(self, node: ExecutionNode) -> None:
        """Add a node to the graph"""
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' already exists in graph")
        self.nodes[node.name] = node
        self._update_execution_order()

    def remove_node(self, node_name: str) -> None:
        """Remove a node and update connections"""
        if node_name not in self.nodes:
            return

        node = self.nodes[node_name]

        # Reconnect predecessors to successors
        for pred_node in self.nodes.values():
            if node in pred_node.next_nodes:
                pred_node.next_nodes.remove(node)
                pred_node.next_nodes.extend(node.next_nodes)

        # Remove from nodes dict
        del self.nodes[node_name]
        self._update_execution_order()

    def connect(self, from_node_name: str, to_node_name: str,
                output_key: str = "default", input_key: str = "default") -> None:
        """Connect two nodes"""
        if from_node_name not in self.nodes:
            raise ValueError(f"Source node '{from_node_name}' not found")
        if to_node_name not in self.nodes:
            raise ValueError(f"Target node '{to_node_name}' not found")

        from_node = self.nodes[from_node_name]
        to_node = self.nodes[to_node_name]

        from_node.next_nodes.append(to_node)
        to_node.inputs[input_key] = (from_node, output_key)
        self._update_execution_order()

    def _update_execution_order(self) -> None:
        """Compute execution layers using topological sort"""
        # Build in-degree map
        in_degree = {name: 0 for name in self.nodes}

        for node in self.nodes.values():
            for next_node in node.next_nodes:
                in_degree[next_node.name] += 1

        # Find nodes with no dependencies (first layer)
        layers = []
        current_layer = [
            self.nodes[name] for name, degree in in_degree.items()
            if degree == 0
        ]

        visited = set()

        while current_layer:
            layers.append(current_layer)
            visited.update(node.name for node in current_layer)

            # Find next layer
            next_layer = []
            for node in current_layer:
                for next_node in node.next_nodes:
                    if next_node.name not in visited and next_node.name not in [n.name for n in next_layer]:
                        # Check if all dependencies are satisfied
                        deps_satisfied = all(
                            source_node.name in visited
                            for source_node, _ in next_node.inputs.values()
                        )
                        if deps_satisfied:
                            next_layer.append(next_node)

            current_layer = next_layer

        self.execution_layers = layers

    def _gather_inputs(self, node: ExecutionNode) -> Dict:
        """Gather input data for a node from its connected sources"""
        input_data = {}
        for key, (source_node, output_key) in node.inputs.items():
            if output_key in source_node.outputs:
                input_data[key] = source_node.outputs[output_key]
            else:
                input_data[key] = None
        return input_data

    async def _execute_node_async(self, node: ExecutionNode, input_data: Dict) -> Dict:
        """Execute a single node asynchronously"""
        try:
            if asyncio.iscoroutinefunction(node.execute):
                result = await node.execute(self.context, input_data)
            else:
                result = node.execute(self.context, input_data)
            return result
        except Exception as e:
            print(f"Error in node {node.name}: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    async def execute_round(self) -> None:
        """Execute one complete round through the graph"""
        for layer_idx, layer in enumerate(self.execution_layers):
            if not layer:
                continue

            print(f"\n[Graph] Executing layer {layer_idx + 1}/{len(self.execution_layers)}: "
                  f"{[n.name for n in layer if n.enabled]}")

            # Filter enabled nodes
            enabled_nodes = [n for n in layer if n.enabled and n.can_execute()]

            if not enabled_nodes:
                continue

            # Execute all nodes in this layer in parallel
            tasks = []
            for node in enabled_nodes:
                input_data = self._gather_inputs(node)
                tasks.append(self._execute_node_async(node, input_data))

            # Wait for all nodes in layer to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Store outputs
            for node, result in zip(enabled_nodes, results):
                if isinstance(result, Exception):
                    print(f"[Graph] Node {node.name} raised exception: {result}")
                    node.outputs = {"error": str(result)}
                elif isinstance(result, dict):
                    node.outputs = result
                else:
                    node.outputs = {"result": result}

    def enable_node(self, node_name: str) -> None:
        """Enable a disabled node"""
        if node_name in self.nodes:
            self.nodes[node_name].enabled = True

    def disable_node(self, node_name: str) -> None:
        """Disable a node without removing it"""
        if node_name in self.nodes:
            self.nodes[node_name].enabled = False

    def get_execution_order(self) -> List[List[str]]:
        """Get the current execution order as list of layers"""
        return [[node.name for node in layer] for layer in self.execution_layers]

    def visualize(self) -> str:
        """Return a text visualization of the graph"""
        lines = ["Execution Graph:"]
        for idx, layer in enumerate(self.execution_layers):
            lines.append(f"\nLayer {idx + 1}:")
            for node in layer:
                status = "✓" if node.enabled else "✗"
                lines.append(f"  {status} {node.name} ({node.node_type})")
        return "\n".join(lines)
