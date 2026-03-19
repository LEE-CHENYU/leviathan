# Agent-Centric Parallel Execution Model

## Current Model vs. Agent-Parallel Model

### Current: Layer-Based (Lockstep)
```
Timeline:
[All agents] → Layer 1: new_round
[All agents] → Layer 2: analyze (parallel within layer)
[All agents] → Layer 3: propose_mechanisms (parallel within layer)
[All agents] → Layer 4: judge
[All agents] → Layer 5: execute_mechanisms
[All agents] → Layer 6: agent_decisions
[All agents] → Layer 7: execute_actions
...
```

**Characteristics:**
- ✓ Simple synchronization points
- ✓ Easy to reason about state
- ✓ Judge can see all proposals at once
- ✗ Faster agents wait for slower ones
- ✗ Not truly simultaneous

### Proposed: Agent-Centric (True Parallel)
```
Timeline:
Agent 0: [new_round → analyze → propose → judge → decide → execute] ━━━━━━━━━━
Agent 1:   [new_round → analyze → propose → judge → decide → execute] ━━━━━━━
Agent 2:     [new_round → analyze → propose → judge → decide → execute] ━━━━
Agent 3:       [new_round → analyze → propose → judge → decide → execute] ━

All agents progress independently through their own graph!
```

**Characteristics:**
- ✓ True simultaneous execution
- ✓ No waiting for slow agents
- ✓ More realistic (real-world parallel)
- ✗ Complex synchronization needed
- ✗ Race conditions possible
- ✗ Judge needs to handle streaming proposals

## Implementation Options

### Option 1: Agent-Specific Graphs (Isolated)

Each agent gets their own graph instance:

```python
class AgentExecutionPipeline:
    """Individual execution pipeline per agent"""
    def __init__(self, agent_id, execution_engine):
        self.agent_id = agent_id
        self.execution_engine = execution_engine
        self.graph = self._create_agent_graph()

    def _create_agent_graph(self):
        """Create graph for this specific agent"""
        graph = ExecutionGraph()

        # Agent-specific nodes
        nodes = {
            'analyze': AnalyzeSelfNode(self.agent_id),
            'propose': ProposeMechanismSelfNode(self.agent_id),
            'decide': AgentDecisionSelfNode(self.agent_id),
            'execute': ExecuteActionsSelfNode(self.agent_id)
        }

        for node in nodes.values():
            graph.add_node(node)

        # Connect nodes
        graph.connect('analyze', 'propose')
        graph.connect('propose', 'decide')
        graph.connect('decide', 'execute')

        return graph

    async def run_round(self):
        """Execute one round for this agent"""
        await self.graph.execute_round()

# In IslandExecution
class IslandExecution:
    def __init__(self, ...):
        # Create pipeline per agent
        self.agent_pipelines = {}
        for member in self.current_members:
            self.agent_pipelines[member.id] = AgentExecutionPipeline(
                member.id, self
            )

    async def run_round_with_parallel_agents(self):
        """All agents execute their pipelines in parallel"""
        tasks = [
            pipeline.run_round()
            for pipeline in self.agent_pipelines.values()
        ]
        await asyncio.gather(*tasks)
```

### Option 2: Shared Graph with Agent Tokens

Agents traverse the shared graph with "tokens":

```python
class AgentToken:
    """Represents an agent's position in the graph"""
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.current_node = 'start'
        self.state = {}

    async def advance(self, graph, execution_engine):
        """Move to next node and execute"""
        current = graph.get_node(self.current_node)

        # Execute node for this agent only
        output = await current.execute_for_agent(
            execution_engine,
            self.agent_id,
            self.state
        )

        # Move to next node
        self.state.update(output)
        self.current_node = current.next_node

        return self.current_node != 'end'

class IslandExecution:
    async def run_round_with_tokens(self):
        """Each agent traverses graph independently"""
        tokens = [
            AgentToken(member.id)
            for member in self.current_members
        ]

        async def traverse_agent(token):
            """One agent's journey through graph"""
            while await token.advance(self.graph, self):
                pass  # Keep going until end

        # All agents traverse in parallel
        await asyncio.gather(*[
            traverse_agent(token)
            for token in tokens
        ])
```

### Option 3: Hybrid (Sync Points)

Mostly parallel with occasional synchronization:

```python
async def run_round_with_sync_points(self):
    """
    Parallel execution with synchronization points
    """
    # Phase 1: Each agent analyzes independently
    await asyncio.gather(*[
        self.analyze_agent(agent_id)
        for agent_id in range(len(self.current_members))
    ])

    # Phase 2: Each agent proposes independently
    await asyncio.gather(*[
        self.propose_mechanism_agent(agent_id)
        for agent_id in range(len(self.current_members))
    ])

    # SYNC POINT: Judge evaluates all proposals
    await self.judge_all_proposals()

    # Phase 3: Each agent decides independently
    await asyncio.gather(*[
        self.agent_decision(agent_id)
        for agent_id in range(len(self.current_members))
    ])

    # Phase 4: Each agent executes independently
    # (with conflict resolution)
    await asyncio.gather(*[
        self.execute_agent_action(agent_id)
        for agent_id in range(len(self.current_members))
    ])

    # SYNC POINT: Resolve conflicts
    self.resolve_conflicts()
```

## Handling Shared State

### Challenge: Concurrent Access

When agents run truly in parallel, they might:
- Read outdated state
- Write conflicting changes
- Create race conditions

### Solution 1: Copy-on-Write

```python
class AgentStateView:
    """Read-only view of world for agent"""
    def __init__(self, execution_engine, agent_id):
        self.snapshot = execution_engine.get_snapshot()
        self.agent_id = agent_id
        self.pending_actions = []

    def read(self, attribute):
        """Read from snapshot"""
        return self.snapshot[attribute]

    def propose_action(self, action):
        """Queue action for later execution"""
        self.pending_actions.append(action)

# Later: Apply all actions with conflict resolution
```

### Solution 2: Event Sourcing

```python
class ActionEvent:
    """Immutable event representing an action"""
    def __init__(self, agent_id, action_type, params, timestamp):
        self.agent_id = agent_id
        self.action_type = action_type
        self.params = params
        self.timestamp = timestamp

class EventLog:
    """Ordered log of all actions"""
    def __init__(self):
        self.events = []
        self.lock = asyncio.Lock()

    async def append(self, event):
        async with self.lock:
            self.events.append(event)

    def apply_all(self, execution_engine):
        """Apply events in order to resolve conflicts"""
        for event in sorted(self.events, key=lambda e: e.timestamp):
            execution_engine.apply_event(event)
```

### Solution 3: Actor Model

```python
class AgentActor:
    """Agent as an independent actor"""
    def __init__(self, agent_id, execution_engine):
        self.agent_id = agent_id
        self.execution_engine = execution_engine
        self.inbox = asyncio.Queue()
        self.running = False

    async def run(self):
        """Main agent loop"""
        self.running = True
        while self.running:
            message = await self.inbox.get()
            await self.handle_message(message)

    async def handle_message(self, message):
        """Process message"""
        if message['type'] == 'tick':
            await self.analyze()
            await self.decide()
            await self.act()
        elif message['type'] == 'interact':
            await self.handle_interaction(message)

    async def send_to(self, other_agent_id, message):
        """Send message to another agent"""
        other = self.execution_engine.agents[other_agent_id]
        await other.inbox.put(message)
```

## Comparison

| Aspect | Current (Layers) | Agent-Parallel |
|--------|------------------|----------------|
| Complexity | Low | High |
| Realism | Medium | High |
| Performance | Good | Better |
| Debugging | Easy | Hard |
| Sync Issues | None | Many |
| State Management | Simple | Complex |

## Recommendation

For MetaIsland's current design:

**Keep layer-based for now, BUT:**
- Within layers, agents are already parallel ✓
- This gives most benefits of parallelism
- Avoids complexity of full agent-parallel model

**Consider agent-parallel if:**
- You want truly simultaneous decision-making
- You're okay with complex conflict resolution
- You want to simulate real-world timing effects

## Migration Path

If you want to move to agent-parallel:

1. **Phase 1:** Add agent-specific context
   ```python
   class AgentContext:
       def __init__(self, agent_id, execution_engine):
           self.agent_id = agent_id
           self.snapshot = execution_engine.get_snapshot()
   ```

2. **Phase 2:** Make nodes agent-aware
   ```python
   async def execute_for_agent(self, context, agent_id):
       # Execute just for this agent
       pass
   ```

3. **Phase 3:** Implement event log
   ```python
   class EventLog:
       # Track all agent actions
       pass
   ```

4. **Phase 4:** Add conflict resolution
   ```python
   def resolve_conflicts(self, events):
       # Deterministic resolution
       pass
   ```

5. **Phase 5:** Switch to parallel traversal
   ```python
   async def run_round(self):
       await asyncio.gather(*[
           agent.traverse_graph()
           for agent in self.agents
       ])
   ```

## Code Example: Quick Implementation

Want me to implement the agent-parallel model? I can create:
1. `AgentPipeline` class
2. Agent-specific node executors
3. Event-based conflict resolution
4. Switch in `metaIsland.py` to enable it

Just say the word!
