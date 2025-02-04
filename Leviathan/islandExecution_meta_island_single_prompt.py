prompt = f"""
[Previous code execution context]
{error_context}

[Current task]
You are member_{member.id} in a society that you can help shape.
Write a Python function named agent_action(execution_engine, member_id) 
that implements your vision of social organization while ensuring your survival.

[Critical constraints]
- Carefully analyze previous errors shown above and avoid repeating them
- Never target yourself (member_{member.id}) in any action
- Verify member has land before using bear() action
- Check member IDs exist before referencing them
- Ensure all matrix indices are valid
- current_members is a LIST accessed by index, not a dictionary
- Access members using execution_engine.current_members[index]
- Check if index exists: if index < len(execution_engine.current_members)

IMPORTANT: Here are the attributes and methods actually available:

1) Each member object has:
    • member.id (int): The unique ID for the member
    • member.vitality (float)
    • member.cargo (float)
    • member.overall_productivity (float)
    • member.age (float)
    • member.current_clear_list (List[int]) - IDs of neighbors or cleared adjacents
2) The relationships are stored in execution_engine.relationship_dict, NOT in "relationship_history".
    Use the arrays in relationship_dict, or rely on the summary below (the 'relations' variable).
    The keys are: 'victim', 'benefit', 'benefit_land'.
    Example usage:
        matrix = execution_engine.relationship_dict['victim']
        # matrix[i, j] indicates how many times member_i was attacked by member_j (if > 0).
3) The parse_relationship_matrix method is used to produce a summary of relationships as a list of strings.
    For example, 'member_2 was attacked by member_1 (value=3.00)'.
4) You can use these methods on execution_engine:
    • execution_engine.attack(member1, member2)
    • execution_engine.offer(member1, member2, True) - Offers resources
    • execution_engine.offer_land(member1, member2, True) - Offers land
    • execution_engine.bear(member1, member2) - Bears offspring
    • execution_engine.expand(member1, member2) - Expands territory
5) The members are accessed by execution_engine.current_members[id].
    For example, execution_engine.current_members[2] for the member with ID=2.
6) DO NOT reference 'member.member_id' or 'member.self_vitality'. Use member.id, member.vitality, etc.

Current status (features of all members):
{features}

Relationship summary (parsed from relationship_dict):
{relations}

Code Memory and Previous Performance:
{code_memory}

Performance history:
{past_performance}

Based on the previous code performance, adapt and improve the strategy.
If a previous strategy worked well (high performance), consider building upon it.
If it failed, try a different approach.

IMPORTANT: Do not simply copy the example implementation below. Instead, use it as inspiration to create your own unique approach combining different methods and strategies in novel ways.

[Social Strategies]
Consider these social strategies:
- Design systems for resource distribution and allocation
- Build alliances and cooperative networks 
- Create mechanisms for collective decision making
- Establish norms and rules for interaction
- Develop methods for conflict resolution

[Communication Strategy]
You can communicate with multiple members in a single round using:
execution_engine.send_message(your_id, recipient_id, "message")
Example usage:
- Broadcast to all: 
    for recipient in range(len(execution_engine.current_members)):
        if recipient != your_id:
            execution_engine.send_message(your_id, recipient, "Let's cooperate!")
- Message allies:
    for ally_id in ally_list:
        execution_engine.send_message(your_id, ally_id, "Attack target X")
- Group coordination:
    for member_id in coalition:
        execution_engine.send_message(your_id, member_id, "Vote YES on proposal")

[Received Messages]
{message_context}

You can also propose plausible modifications to the game mechanics themselves, such as:
- Adding new resource types or currencies
- Creating new actions or interaction types
- Implementing voting systems or governance structures
- Defining property rights and ownership rules
- Adding social status or reputation systems
- Creating markets or trading mechanisms
- Defining new win conditions or goals

[Social System Design]
Example modifications:
def pre_init_hook(island):
    # Set up basic income system
    island.basic_income = 10.0
    island.tax_rate = 0.2
    
def modify_member(member, relationships):
    # Add social status and rights
    member.social_rank = 0
    member.voting_power = 1
    member.tax_paid = 0
    member.benefits_received = 0
    return member
    
def modify_land(land, members):
    # Create communal lands
    land.communal_areas = []
    land.private_areas = []
    return land
    
def modify_relationships(relationships):
    # Add social bonds
    relationships['alliance'] = np.zeros_like(relationships['victim'])
    relationships['trade_history'] = np.zeros_like(relationships['victim'])
    return relationships

[Survival-Centric Adaptation]
Implement systems focused on:
1. Personal resource optimization
2. Threat assessment and neutralization
3. Vitality preservation techniques
4. Adaptive hoarding strategies
5. Predictive danger modeling

Survival-First Examples:
def agent_action(execution_engine, member_id):
    me = execution_engine.current_members[member_id]
    
    # Emergency resource reserve
    if me.cargo < me.vitality * 0.5:
        for other in execution_engine.current_members:
            if other.id != me.id and other.cargo > me.cargo:
                execution_engine.attack(me, other)
    
    # Adaptive territory defense
    if me.land_num > 2 and me.vitality < 50:
        for loc in me.owned_land[1:]:
            execution_engine._discard_land(me, loc)
    
    # Survival Q-learning
    if not hasattr(me, 'survival_q'):
        me.survival_q = defaultdict(float)
        
    state = (me.vitality//20, me.cargo//20)
    action = max(['attack','hide','steal'], 
                key=lambda a: me.survival_q.get((state,a),0))
    
    # Execute and update based on survival outcome
    if action == 'attack':
        # Implementation logic
        me.survival_q[(state,action)] += me.vitality * 0.1

[Survival Metrics]
Evaluate strategies by:
- Personal vitality delta
- Resource acquisition rate
- Threat neutralization count
- Survival probability increase
- Attack success:fail ratio

[Implementation Priorities]
1. Create personal health monitoring systems
2. Develop egocentric threat models
3. Optimize actions for caloric ROI
4. Implement fail-deadly safeguards
5. Build predictive self-preservation models

def calculate_survival_roi(action_history):
    roi = {{}}
    for action, outcome in action_history:
        vitality_gain = outcome['vitality']
        cost = outcome['vitality_cost']
        roi[action] = vitality_gain / cost if cost > 0 else 0
    return max(roi, key=roi.get)

Return only the code, no extra text or explanation. While the example above shows one possible approach,
you should create your own unique implementation drawing from the wide range of available methods and strategies.
Consider novel combinations of different approaches rather than following this exact pattern.
"""