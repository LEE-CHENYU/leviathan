# Agent Prompts Update - COMPLETE ✅

## What Was Added

### 1. agent_mechanism_proposal.py - NEW SYSTEMS SECTION

Added comprehensive contract and physics templates that agents can use to build the economy:

#### Contract Templates (7 Total)
1. **Resource Definition & Extraction** - Define new resources with extraction rules
2. **Simple Trade Contract** - Bilateral resource exchange
3. **Service Contract (Labor/Production)** - Ongoing service agreements
4. **Business/Partnership** - Multi-party businesses with profit sharing
5. **Agricultural Physics** - Realistic constraints with soil depletion
6. **Manufacturing Physics** - Input/output ratios and recipes
7. **Market & Pricing Mechanism** - Supply/demand price discovery

**Key Features:**
- Templates show complete working code
- Include conservation laws and realistic constraints
- Demonstrate proper use of contract/physics systems
- Provide examples agents can adapt

### 2. agent_code_decision.py - CONTRACT ACTIONS & MARKET LOGIC

Added 7 action templates showing how to use the economic systems:

#### Action Templates
1. **Propose Trade Contracts** - How to create and propose trades
2. **Sign Pending Contracts** - Evaluate and sign beneficial contracts
3. **Market Saturation Analysis & Business Pivot** - Calculate saturation, detect oversupply, pivot to better markets
4. **Join/Create Business Partnership** - Form or join businesses
5. **Extract Resources** - Use resource systems with costs
6. **Place Market Orders** - Buy/sell on markets
7. **Supply Chain Management** - Build supplier relationships

#### Economic Strategy Patterns
- **Specialization Strategy** - Focus on one product
- **Diversification Strategy** - Multiple products
- **Middleman Strategy** - Buy low, sell high
- **Vertical Integration** - Control full supply chain
- **Service Provider** - Offer labor/services

**Key Features:**
- Complete market saturation calculations
- Business pivot logic with profitability analysis
- Contract signing evaluation
- Supply chain formation through contracts

## How Agents Will Use These Systems

### Round Flow with New Systems

```
1. MECHANISM PROPOSAL PHASE
   Agent sees templates for:
   - Defining resources (food, materials, energy, etc.)
   - Creating physics constraints (agricultural, manufacturing)
   - Building market mechanisms
   - Setting up business systems

   Agent proposes:
   def propose_modification(execution_engine):
       # Create resource system
       # Add physics constraints
       # Build market infrastructure

2. JUDGE VALIDATION
   Judge checks proposals for:
   - Conservation laws (no free resources)
   - Realistic physics (diminishing returns, costs)
   - Fair implementation (not self-serving exploits)

3. APPROVED MECHANISMS EXECUTE
   - Resource systems become available
   - Physics constraints apply to all actions
   - Markets open for trading
   - Businesses can be formed

4. AGENT ACTION PHASE
   Agent sees templates for:
   - Proposing trade contracts
   - Signing beneficial contracts
   - Analyzing market saturation
   - Pivoting to profitable products
   - Extracting resources
   - Placing market orders
   - Building supply chains

   Agent chooses actions:
   def agent_action(execution_engine, member_id):
       # Check market saturation
       # Propose contracts
       # Sign pending contracts
       # Extract resources
       # Place market orders

5. CONTRACT EXECUTION
   - Signed contracts execute automatically
   - Resources transfer between parties
   - Payments process
   - Supply chains deliver

6. ECONOMIC FEEDBACK
   - Agents see profit/loss
   - Market prices adjust
   - Saturation levels change
   - Agents adapt next round
```

## Expected Emergent Behaviors

### Early Rounds (1-5)
1. **Resource Definition** - Agents propose food, materials, energy
2. **Basic Physics** - Agricultural and manufacturing constraints
3. **Simple Trades** - Bilateral resource exchanges
4. **Market Creation** - Basic buy/sell order systems

### Mid Rounds (6-15)
1. **Specialization** - Agents focus on specific products
2. **Market Saturation** - Oversupply in popular products
3. **Business Pivots** - Agents switch to less saturated markets
4. **Partnerships** - Multi-party businesses form
5. **Supply Chains** - Recurring supplier relationships

### Late Rounds (16+)
1. **Complex Markets** - Multiple interconnected markets
2. **Vertical Integration** - Agents control full production chains
3. **Service Economy** - Labor and service contracts
4. **Economic Cycles** - Boom/bust patterns emerge
5. **Strategic Adaptation** - Agents anticipate and counter each other

## Template Categories in Prompts

### For Mechanism Proposals (agent_mechanism_proposal.py)

```python
[Template 1: Resource Definition]
- Define resource types
- Set extraction costs
- Create inventories

[Template 2: Trade Contract]
- Bilateral exchanges
- Resource verification
- Transfer logic

[Template 3: Service Contract]
- Ongoing relationships
- Payment per round
- Service delivery

[Template 4: Business/Partnership]
- Multi-party formation
- Contribution requirements
- Profit sharing

[Template 5: Agricultural Physics]
- Soil depletion
- Diminishing returns
- Resource requirements

[Template 6: Manufacturing Physics]
- Input/output recipes
- Efficiency factors
- Resource checking

[Template 7: Market System]
- Order placement
- Order matching
- Price discovery
```

### For Agent Actions (agent_code_decision.py)

```python
[Template 1: Propose Trade]
- Find trading partners
- Create contract code
- Propose to counterparty

[Template 2: Sign Contracts]
- Review pending contracts
- Evaluate terms
- Sign if beneficial

[Template 3: Market Analysis & Pivot]
- Calculate saturation = supply / demand
- Calculate profit_margin = (price - cost) / cost
- Find better opportunities
- Pivot to new products

[Template 4: Form Partnerships]
- Find compatible partners
- Propose business contracts
- Join existing businesses

[Template 5: Extract Resources]
- Use resource systems
- Pay costs (vitality, etc.)
- Update inventory

[Template 6: Market Orders]
- Place buy orders
- Place sell orders
- Price competitively

[Template 7: Supply Chains]
- Find suppliers
- Propose ongoing contracts
- Build dependencies
```

## Key Mechanisms Explained

### Market Saturation Calculation
```python
total_supply = sum(sell orders)
total_demand = sum(buy orders)
saturation = total_supply / total_demand

if saturation > 1.5:  # Oversupply
    # Find better market
    pivot_to_new_product()
```

### Profit Margin Analysis
```python
current_price = market.get_market_price(product)
base_cost = production_cost
profit_margin = (current_price - base_cost) / base_cost

if profit_margin < 0.1:  # Less than 10%
    # Not profitable, pivot
```

### Business Pivot Decision
```python
# Analyze all markets
for product in all_products:
    saturation = calculate_saturation(product)
    profit_margin = calculate_profit(product)

    if saturation < current_saturation * 0.7:
        # Much less saturated
        pivot_to(product)
```

### Contract Signing Logic
```python
for contract in pending_contracts:
    terms = contract['terms']

    # Evaluate if beneficial
    my_cost = evaluate_my_cost(terms)
    my_benefit = evaluate_my_benefit(terms)

    if my_benefit > my_cost * 1.2:  # 20% profit margin
        sign_contract(contract)
```

## Judge Validation Examples

### APPROVED Examples
```python
# Resource with costs
def extract_food(member_id):
    member.vitality -= 2  # Costs vitality
    inventory[member_id]['food'] += 5
    # ✓ Has cost, realistic yield

# Manufacturing with inputs
def manufacture(member_id):
    if inventory[member_id]['materials'] >= 10:
        inventory[member_id]['materials'] -= 10
        inventory[member_id]['products'] += 3
        # ✓ Uses inputs, realistic ratio

# Market with supply/demand
def calculate_price(resource):
    supply = count_sell_orders(resource)
    demand = count_buy_orders(resource)
    return base_price * (demand / supply)
    # ✓ Realistic price discovery
```

### REJECTED Examples
```python
# Free resources
def extract_food(member_id):
    inventory[member_id]['food'] += 1000
    # ✗ No cost, unrealistic

# Perpetual motion
def manufacture(member_id):
    inventory[member_id]['energy'] += 10
    # ✗ Creates energy from nothing

# Unfair advantage
def special_ability():
    if member_id == 5:  # Only I benefit
        inventory[member_id]['gold'] += 999
    # ✗ Unfair, self-serving
```

## Testing the New Prompts

### Quick Test (No LLM)
```bash
python test_graph_system.py
# Verifies infrastructure works
```

### Full Test (With LLM)
```bash
python MetaIsland/metaIsland.py
```

**What to Watch For:**
1. **Round 1-2**: Agents propose resource and market systems
2. **Round 3-4**: Agents start proposing contracts
3. **Round 5+**: Contracts execute, markets form
4. **Round 10+**: Business pivots occur
5. **Judge logs**: Check approval/rejection rates

## Success Metrics

### System Health
- **Approval Rate**: ~60-80% of proposals approved
- **Contract Rate**: >50% of agents proposing contracts by round 5
- **Market Activity**: >10 orders per round by round 10
- **Pivot Rate**: >20% of agents pivoting when saturated

### Economic Complexity
- **Resource Types**: 3-5 different resources emerge
- **Active Contracts**: 10+ active contracts by round 10
- **Business Count**: 2-3 partnerships form
- **Supply Chains**: Multi-hop dependencies emerge

### Agent Adaptation
- **Profit Tracking**: Agents calculate their profitability
- **Strategic Pivots**: Agents switch products when unprofitable
- **Partnership Formation**: Agents seek mutually beneficial agreements
- **Market Awareness**: Agents respond to price signals

## Troubleshooting

### If Agents Don't Use New Systems
**Problem**: Agents stick to old attack/offer actions

**Solution**:
1. Check if templates are in prompts (search for "CONTRACT TEMPLATES")
2. Reduce round count to see early rounds
3. Check judge rejection reasons
4. Verify systems exist: `hasattr(execution_engine, 'contracts')`

### If Judge Rejects Everything
**Problem**: Approval rate < 30%

**Solution**:
1. Review rejection reasons in judge logs
2. Common issues: free resources, no costs, exploits
3. Agents need to include costs and constraints
4. May need to adjust judge prompt sensitivity

### If No Contracts Execute
**Problem**: Contracts proposed but never activate

**Solution**:
1. Check if both parties signed: `len(signatures) == len(parties)`
2. Verify contract code is valid Python
3. Check execution errors in logs
4. Ensure ContractNode is enabled in graph

### If Markets Don't Form
**Problem**: No market orders placed

**Solution**:
1. Verify market system was proposed and approved
2. Check if agents have resources to trade
3. Ensure agents see market templates in prompts
4. Check if MarketSystem was created in mechanisms

## Next Steps

The system is now ready for full economic simulation! To get started:

1. **Run Simulation**:
   ```bash
   python MetaIsland/metaIsland.py
   ```

2. **Monitor Logs**:
   - Judge decisions
   - Contract proposals
   - Market saturation
   - Business pivots

3. **Analyze Results**:
   - Check `generated_code/` directory
   - Review execution history JSON
   - Track economic metrics

4. **Iterate**:
   - Adjust island ideology for different economies
   - Tune judge strictness
   - Add more template examples
   - Enable/disable specific nodes

## Files Modified

- ✅ `MetaIsland/agent_mechanism_proposal.py` - Added 7 contract/physics templates
- ✅ `MetaIsland/agent_code_decision.py` - Added 7 action templates + market logic
- ✅ `MetaIsland/metaIsland.py` - Integrated graph system
- ✅ `MetaIsland/graph_engine.py` - Created (new)
- ✅ `MetaIsland/contracts.py` - Created (new)
- ✅ `MetaIsland/physics.py` - Created (new)
- ✅ `MetaIsland/judge.py` - Created (new)
- ✅ `MetaIsland/nodes/` - Created 12 nodes (new)

## Documentation

- `ARCHITECTURE.md` - System architecture visualization
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `AGENT_PROMPTS_COMPLETE.md` - This file
- `test_graph_system.py` - Test suite

---

**Status**: FULLY COMPLETE AND READY TO RUN

The MetaIsland economic simulation now has everything needed for agents to build a realistic, emergent economy through contracts, physics constraints, and market dynamics. Agents will code their own interactions, propose realistic simulations, and adapt to market forces - exactly as envisioned!
