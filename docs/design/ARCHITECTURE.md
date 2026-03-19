# MetaIsland Economic Simulation - Architecture Visualization

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MetaIsland Execution                          │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Contract   │  │   Physics    │  │    Judge     │              │
│  │    Engine    │  │   Engine     │  │   System     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                 │                  │                       │
│         └─────────────────┴──────────────────┘                       │
│                           │                                          │
│                  ┌────────▼────────┐                                │
│                  │  Execution Graph │                                │
│                  │   (DAG-based)    │                                │
│                  └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Execution Graph - Round Flow

```
Round Start
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Initialization (Parallel)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │New Round │   │Get State │   │Neighbors │               │
│  └──────────┘   └──────────┘   └──────────┘               │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Analysis Phase (All Agents in Parallel)            │
├─────────────────────────────────────────────────────────────┤
│  Agent 0        Agent 1        Agent 2      ...   Agent N   │
│     │              │              │                   │      │
│  ┌──▼───┐      ┌──▼───┐      ┌──▼───┐           ┌──▼───┐  │
│  │Analyze│      │Analyze│      │Analyze│   ...   │Analyze│  │
│  └──────┘      └──────┘      └──────┘           └──────┘  │
│                                                              │
│  All agents observe SAME pre-round state                    │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Mechanism Proposals (All Agents in Parallel)       │
├─────────────────────────────────────────────────────────────┤
│  Agent 0        Agent 1        Agent 2      ...   Agent N   │
│     │              │              │                   │      │
│  ┌──▼────────┐ ┌──▼────────┐ ┌──▼────────┐      ┌──▼────┐ │
│  │ Propose   │ │ Propose   │ │ Propose   │  ... │Propose│ │
│  │Constraints│ │Contracts  │ │Resources  │      │Physics│ │
│  └───────────┘ └───────────┘ └───────────┘      └───────┘ │
│                                                              │
│  Agents code their own:                                     │
│  • Physical laws & constraints                              │
│  • Resource definitions                                     │
│  • Economic mechanisms                                      │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Judge Validation (Sequential)                      │
├─────────────────────────────────────────────────────────────┤
│                  ┌──────────┐                               │
│     All Proposals│  Master  │                               │
│          ───────►│  Judge   │◄────────                      │
│                  │  (LLM)   │                               │
│                  └─────┬────┘                               │
│                        │                                     │
│           ┌────────────┴────────────┐                       │
│           ▼                         ▼                        │
│      ┌─────────┐              ┌─────────┐                  │
│      │APPROVED │              │REJECTED │                  │
│      │  Code   │              │  Code   │                  │
│      └────┬────┘              └────┬────┘                  │
│           │                         │                       │
│           │                         └──► Log reason         │
│           ▼                                                  │
│    Execute & Apply                                          │
│                                                              │
│  Judge checks:                                              │
│  • Conservation laws (no free energy/resources)             │
│  • Fairness (no unfair advantages)                          │
│  • Realism (economic sense)                                 │
│  • No exploits/infinite loops                               │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Agent Actions (All Agents in Parallel)             │
├─────────────────────────────────────────────────────────────┤
│  Agent 0        Agent 1        Agent 2      ...   Agent N   │
│     │              │              │                   │      │
│  ┌──▼───────┐  ┌──▼───────┐  ┌──▼───────┐       ┌──▼────┐ │
│  │ Generate │  │ Generate │  │ Generate │   ... │Generate│ │
│  │ Actions  │  │ Actions  │  │ Actions  │       │Actions │ │
│  └──────────┘  └──────────┘  └──────────┘       └────────┘ │
│                                                              │
│  Hidden actions - agents don't see each other's choices     │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 6: Conflict Resolution (Sequential)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────┐               │
│  │  Collect All Proposed Actions            │               │
│  └──────────────┬───────────────────────────┘               │
│                 │                                            │
│                 ▼                                            │
│  ┌─────────────────────────────────────────┐               │
│  │  Detect Conflicts:                       │               │
│  │  • Multiple extractions from same land   │               │
│  │  • Competing trades                      │               │
│  │  • Resource competition                  │               │
│  └──────────────┬───────────────────────────┘               │
│                 │                                            │
│                 ▼                                            │
│  ┌─────────────────────────────────────────┐               │
│  │  Apply Resolution Rules:                 │               │
│  │  • Proportional allocation               │               │
│  │  • Mutual agreement requirement          │               │
│  │  • Deterministic ordering (seeded)       │               │
│  └──────────────┬───────────────────────────┘               │
│                 │                                            │
│                 ▼                                            │
│           Execute Resolved                                   │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 7: Contract Processing (Sequential)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Process    │───►│   Execute    │───►│    Update    │ │
│  │   Pending    │    │   Active     │    │   Contract   │ │
│  │  Signatures  │    │  Contracts   │    │   Status     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                              │
│  Contract types agents create:                              │
│  • Resource exchange                                        │
│  • Service provision                                        │
│  • Business partnerships                                    │
│  • Loan agreements                                          │
│  • Labor contracts                                          │
│  • Supply chain agreements                                  │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 8: Physical World Updates (Parallel)                  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │ Produce  │   │ Consume  │   │Environment│              │
│  │Resources │   │Resources │   │  Update   │              │
│  └──────────┘   └──────────┘   └──────────┘               │
│                                                              │
│  Apply agent-defined physics:                               │
│  • Resource extraction with constraints                     │
│  • Soil depletion                                           │
│  • Resource regeneration                                    │
│  • Environmental damage                                     │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 9: Economic Feedback (Sequential)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────┐               │
│  │  Calculate Market Dynamics               │               │
│  │  • Supply/demand per product             │               │
│  │  • Market saturation                     │               │
│  │  • Price discovery                       │               │
│  └──────────────┬───────────────────────────┘               │
│                 │                                            │
│                 ▼                                            │
│  ┌─────────────────────────────────────────┐               │
│  │  Agent Profit Calculation                │               │
│  │  • Revenues - Costs = Profit             │               │
│  │  • Adjusted for competition              │               │
│  │  • Profit rate determines survival       │               │
│  └──────────────┬───────────────────────────┘               │
│                 │                                            │
│                 ▼                                            │
│  ┌─────────────────────────────────────────┐               │
│  │  Business Pivot Detection                │               │
│  │  If profit_rate < threshold:             │               │
│  │    • Analyze alternative markets         │               │
│  │    • Calculate pivot costs               │               │
│  │    • Plan retooling                      │               │
│  └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 10: Logging & Cleanup (Sequential)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │   Log    │   │   Save   │   │  Update  │               │
│  │  Status  │   │ History  │   │ Metrics  │               │
│  └──────────┘   └──────────┘   └──────────┘               │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
                 Round End
```

## Agent Code Writing Flow

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT PERSPECTIVE                     │
└─────────────────────────────────────────────────────────┘

Each Round, Each Agent Writes Code for:

1. PHYSICS & CONSTRAINTS (Mechanism Proposal)
   │
   ├─► def propose_modification(execution_engine):
   │       # Define physical constraints for my domain
   │       class AgriculturalPhysics:
   │           soil_depletion_rate = 0.1
   │           water_requirement = 5
   │           growth_time = 3
   │
   │       execution_engine.physics.add_constraint(AgriculturalPhysics())
   │
   └─► Sent to Judge ──► [APPROVE/REJECT] ──► Apply if approved

2. CONTRACTS (Action Phase)
   │
   ├─► def agent_action(execution_engine, member_id):
   │       # Propose trade contract
   │       trade = ResourceExchangeContract(
   │           party_a=member_id,
   │           party_b=target_id,
   │           offer={'food': 10},
   │           request={'energy': 20}
   │       )
   │       execution_engine.contracts.propose(trade)
   │
   └─► Awaits counterparty signature ──► Execute if signed

3. BUSINESS LOGIC (Action Phase)
   │
   ├─► def agent_action(execution_engine, member_id):
   │       # Calculate profitability
   │       profit_rate = (revenue - costs) / costs
   │
   │       if profit_rate < 0.1:
   │           # Market saturated - PIVOT
   │           new_market = find_best_opportunity()
   │           pivot_business(new_market)
   │
   └─► Executes in parallel with other agents

4. DEPENDENCIES & LABOR (Action Phase)
   │
   ├─► def agent_action(execution_engine, member_id):
   │       # Find suppliers for my factory
   │       needed = {'raw_materials': 10, 'energy': 5}
   │
   │       for resource, quantity in needed.items():
   │           supplier = find_supplier(resource)
   │           contract = ServiceContract(supplier, member_id, ...)
   │           execution_engine.contracts.propose(contract)
   │
   └─► Build supply chain through contracts
```

## Key Infrastructure Components

```
┌─────────────────────────────────────────────────────────────┐
│                   CONTRACT ENGINE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  contracts = {                                               │
│    pending: {contract_id: {...}}   # Awaiting signatures    │
│    active: {contract_id: {...}}    # Being executed         │
│    history: [...]                  # Completed              │
│  }                                                           │
│                                                              │
│  Methods:                                                    │
│  • propose_contract(code, parties, terms)                   │
│  • sign_contract(contract_id, party_id)                     │
│  • execute_contract(contract_id, context)                   │
│  • verify_conditions(contract_id)                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   PHYSICS ENGINE                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  constraints = [                                             │
│    AgriculturalConstraints,                                 │
│    ManufacturingConstraints,                                │
│    EnvironmentalConstraints,                                │
│    ...                                                       │
│  ]                                                           │
│                                                              │
│  Methods:                                                    │
│  • propose_constraint(code, proposer, domain)               │
│  • apply_constraints(action, context)                       │
│  • validate_conservation(before, after)                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     JUDGE SYSTEM                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  def judge_proposal(code, proposer_id, type):                │
│                                                              │
│    prompt = """                                              │
│      Check for:                                              │
│      1. Unrealistic physics (free energy)                   │
│      2. Reward hacking (free resources)                     │
│      3. Unfair advantages                                   │
│      4. Exploits/infinite loops                             │
│                                                              │
│      Reply: APPROVE or REJECT: [reason]                     │
│    """                                                       │
│                                                              │
│    response = llm.call(prompt)                               │
│    return (approved, reason)                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Emergent Economic Behaviors

```
                    Agent-Coded Interactions
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                     │
        ▼                    ▼                     ▼
┌──────────────┐    ┌──────────────┐     ┌──────────────┐
│   Resource   │    │    Labor     │     │   Supply     │
│  Competition │    │   Markets    │     │   Chains     │
└──────┬───────┘    └──────┬───────┘     └──────┬───────┘
       │                   │                     │
       │   Homogenous      │  Specialization     │
       │   Products        │  & Skills           │  Complex
       │      │            │      │              │  Dependencies
       └──────┼────────────┼──────┼──────────────┘
              │            │      │
              ▼            ▼      ▼
       ┌─────────────────────────────┐
       │   MARKET SATURATION         │
       │   profit_rate drops         │
       └──────────┬──────────────────┘
                  │
                  ▼
       ┌─────────────────────────────┐
       │   BUSINESS PIVOTS           │
       │   Agents retool & switch    │
       └──────────┬──────────────────┘
                  │
                  ▼
       ┌─────────────────────────────┐
       │   ECONOMIC CYCLES           │
       │   Boom → Saturation → Bust  │
       └─────────────────────────────┘
```

## Data Flow

```
                    Round N State
                         │
        ┌────────────────┼────────────────┐
        │                │                 │
        ▼                ▼                 ▼
   Agents Read      Agents Read       Agents Read
   Environment      Contracts         Other Agents
        │                │                 │
        └────────────────┼─────────────────┘
                         │
                         ▼
                  Agents Write Code
                  (proposals + actions)
                         │
                         ▼
                    Judge Reviews
                         │
                ┌────────┴────────┐
                ▼                 ▼
           APPROVED           REJECTED
                │                 │
                │                 └─► Log & Learn
                ▼
         Execute in Parallel
                │
                ▼
         Resolve Conflicts
                │
                ▼
         Update State
                │
                ▼
         Calculate Economics
         (profit, saturation)
                │
                ▼
            Round N+1 State
```

## File Structure

```
MetaIsland/
├── metaIsland.py              # Main execution class with graph
├── graph_engine.py            # Graph execution engine (NEW)
│   ├── ExecutionNode (base)
│   ├── ExecutionGraph
│   └── Parallel execution logic
│
├── contracts.py               # Contract system (NEW)
│   ├── ContractEngine
│   ├── Contract types
│   └── Execution logic
│
├── physics.py                 # Physical constraints (NEW)
│   ├── PhysicsEngine
│   ├── Constraint registry
│   └── Conservation laws
│
├── judge.py                   # Judge system (NEW)
│   ├── judge_proposal()
│   ├── Conservation checks
│   └── Validation rules
│
├── nodes/                     # Execution nodes (NEW)
│   ├── analyze_node.py
│   ├── proposal_node.py
│   ├── action_node.py
│   ├── contract_node.py
│   ├── conflict_resolution_node.py
│   └── environment_node.py
│
├── base_island.py             # Base Island class
├── base_member.py             # Base Member class
├── base_land.py               # Land management
│
├── agent_mechanism_proposal.py  # Mechanism proposals (MODIFY)
├── agent_code_decision.py       # Agent actions (MODIFY)
├── analyze.py                   # Analysis (MODIFY)
│
└── model_router.py            # LLM routing
```

## Implementation Phases

```
Phase 1: Core Infrastructure (Week 1)
├─► graph_engine.py
├─► contracts.py (basic)
├─► physics.py (basic)
└─► judge.py

Phase 2: Node Implementation (Week 2)
├─► Create all execution nodes
├─► Modify existing agent prompts
├─► Add conflict resolution
└─► Test parallel execution

Phase 3: Economic Mechanics (Week 3)
├─► Market saturation calculation
├─► Profit rate tracking
├─► Business pivot logic
└─► Supply chain discovery

Phase 4: Advanced Features (Week 4+)
├─► Complex contracts
├─► Environmental feedback
├─► Labor markets
└─► Skill development
```

## Key Design Principles

1. **Minimal Hardcoding**: Agents write all economic logic
2. **Graph-Based Flow**: Easy to modify/extend execution order
3. **Parallel Execution**: Agents act simultaneously within phases
4. **Judge Validation**: Simple LLM prevents exploits
5. **Contract-Centric**: Economy emerges from bilateral agreements
6. **Realistic Constraints**: Agent-proposed physics with voting
7. **Market Forces**: Saturation leads to natural pivots
8. **Emergent Complexity**: Simple rules → complex behaviors
