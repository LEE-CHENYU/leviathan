from itertools import tee
import openai
import traceback
import ast
from dotenv import load_dotenv
import aisuite as ai

from MetaIsland.model_router import model_router

load_dotenv()

client = ai.Client()

provider, model_id = model_router("deepseek")

async def _agent_mechanism_proposal(self, member_id) -> None:
    """ 
    Asks GPT for directly executable Python code, stores it in a dictionary keyed by member_id.
    The code will define a function propose_modification(execution_engine), 
    which references attributes that actually exist.
    """
    # Prepare data for the proposal
    data = self.prepare_agent_data(member_id)
    
    # Use the prepared data
    member = data['member']
    relations = data['relations']
    features = data['features']
    code_memory = data['code_memory']
    analysis_memory = data['analysis_memory']
    past_performance = data['past_performance']
    error_context = data['error_context']
    message_context = data['message_context']
    
    current_mechanisms = data['current_mechanisms']
    modification_attempts = []
    for round_num in data['modification_attempts'].keys():
        round_attempts = [attempt for attempt in data['modification_attempts'][round_num] if attempt.get('member_id') == member_id]
        modification_attempts.extend(round_attempts)
    report = data['report']
    
    base_code = self.base_class_code
    
    base_code = f"""
            [Base Code]
            Here is the base code for the Island and Member classes that you should reference when making modifications. Study the mechanisms carefully to ensure your code interacts correctly with the available attributes and methods and objects defined in [Active Mechanisms Modifications]. Pay special attention to:
            - Valid attribute access patterns
            - Method parameters and return values 
            - Constraints and preconditions for actions
            - Data structure formats and valid operations
            {base_code}
            """
    part0 = f"""
        [Previous code execution errors context]
        Here are the errors that occurred in the previous code execution, you can use them as reference to avoid repeating them:
    {error_context}
    
    [Current Task]
    Island is a mechanical environment that every agent would interact with and get impacted.
    As an agent, you can propose modifications to the game mechanics to improve your survival chance.
    Write a Python function named propose_modification(execution_engine) that implements your proposal of modifications to the game mechanics.
    
    [Island Ideology]
    {self.island_ideology}
    
    [Critical constraints]
    - Carefully analyze previous errors shown above and inspect._void repeating them
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
        • member.vitality (float) range: [0, 100]
        • member.cargo (float) range: [0, ]
        • member.overall_productivity (float) range: [0, ]
        • member.age (float) range: [0, 100]
        • member.current_clear_list (List[int]) - IDs of neighbors or cleared adjacents
        • Other attributes or attributes of objects defined in [Active Mechanisms Modifications]
    2) The relationships are stored in execution_engine.relationship_dict, NOT in "relationship_history".
        Use the arrays in relationship_dict, or rely on the summary below (the 'relations' variable).
        The keys are: 'victim', 'benefit', 'benefit_land'.
        Example usage:
            victim_matrix = execution_engine.relationship_dict['victim']
            # victim_matrix[i,j] indicates how much vitality member_i lost from member_j's attacks (if > 0)
            # A higher value means member_j has attacked member_i more severely or frequently
            
            benefit_matrix = execution_engine.relationship_dict['benefit']
            # benefit_matrix[i,j] indicates how much resources member_i received from member_j (if > 0)
            # A higher value means member_j has helped member_i more generously or frequently
        Find also other matrixes defined in [Active Mechanisms Modifications]
    3) The parse_relationship_matrix method is used to produce a summary of relationships as a list of strings.
        For example, 'member_2 was attacked by member_1 (value=3.00)'.
    4) You can use these methods on execution_engine:
        • execution_engine.attack(member1, member2)
        • execution_engine.offer(member1, member2) - Offers resources
        • execution_engine.offer_land(member1, member2) - Offers land
        • execution_engine.bear(member1, member2) - Bears offspring
        • execution_engine.expand(member1) - Expands territory
        • Other methods and methods of objects defined in [Active Mechanisms Modifications]
    5) The members are accessed by execution_engine.current_members[id].
        For example, execution_engine.current_members[2] for the member with ID=2.
    6) DO NOT reference 'member.member_id' or 'member.self_vitality'. Use member.id, member.vitality, etc.

    Analysis of the game state:
    {report}
    
    Current status:
    Here are the basic information of all members, you should make your own decisions based on them:
    {features}

    Relationship summary (parsed from relationship_dict):
    Here are the relationships between members:
    {relations}

    Code Memory and Previous Performance:
    {code_memory}

    Analysis Memory:
    {analysis_memory}

    Performance history:
    {past_performance}

    Based on the previous code performance, propose a modification to the game mechanics.
    If a previous proposal worked well (high performance), consider building upon it.
    If it failed, try a different approach.
    
    IMPORTANT: Do not simply copy the example implementation below. Instead, use it as inspiration to create your own unique approach combining different methods and strategies in novel ways.
    
    While the example above shows one possible approach,
    you should create your own unique implementation drawing from the wide range of available methods and strategies.
    
    Consider novel combinations of different approaches rather than following this exact pattern.
    """

    # NEW: Contract and Physics Templates
    new_systems_guide = f"""
    {part0}

    [NEW SYSTEMS AVAILABLE]
    You now have access to powerful new systems for building economic mechanisms:

    1. CONTRACT SYSTEM (execution_engine.contracts)
       - Create bilateral/multilateral agreements
       - Define resource exchanges, services, partnerships
       - Contracts execute automatically when conditions met

    2. PHYSICS ENGINE (execution_engine.physics)
       - Define physical constraints for your domain
       - Create realistic resource extraction/transformation rules
       - Propose conservation laws

    3. JUDGE SYSTEM
       - Your code will be validated for realism
       - No free resources or exploits allowed
       - Must follow conservation laws

    ==================================================================
    CONTRACT TEMPLATES
    ==================================================================

    [Template 1: Resource Definition & Extraction]
    def propose_modification(execution_engine):
        '''Define a new resource type with extraction rules'''

        if not hasattr(execution_engine, 'resources'):
            class ResourceSystem:
                def __init__(self):
                    self.resource_types = {{}}
                    self.inventories = {{}}  # {{member_id: {{resource: quantity}}}}

                def define_resource(self, name, extraction_params):
                    self.resource_types[name] = extraction_params

                def extract(self, member_id, land_location, resource_type):
                    if resource_type not in self.resource_types:
                        return 0

                    params = self.resource_types[resource_type]
                    # Apply extraction constraints
                    base_yield = params.get('base_yield', 1.0)
                    land_quality = params.get('land_quality_factor', 1.0)

                    extracted = base_yield * land_quality

                    # Update inventory
                    if member_id not in self.inventories:
                        self.inventories[member_id] = {{}}
                    self.inventories[member_id][resource_type] = \\
                        self.inventories[member_id].get(resource_type, 0) + extracted

                    return extracted

            execution_engine.resources = ResourceSystem()

            # Define initial resources
            execution_engine.resources.define_resource('food', {{
                'base_yield': 5.0,
                'land_quality_factor': 1.0,
                'extraction_cost_vitality': 2.0
            }})

            execution_engine.resources.define_resource('materials', {{
                'base_yield': 3.0,
                'land_quality_factor': 0.8,
                'extraction_cost_vitality': 3.0
            }})

    [Template 2: Simple Trade Contract]
    def propose_modification(execution_engine):
        '''Create a contract template for resource trading'''

        if not hasattr(execution_engine, 'contract_templates'):
            execution_engine.contract_templates = {{}}

        # Define trade contract template
        trade_contract_template = \'\'\'
def execute_contract(execution_engine, context):
    # Execute a simple resource trade between two parties
    contract = context.get('contract', {{}})
    party_a = contract['parties'][0]
    party_b = contract['parties'][1]
    terms = contract['terms']

    # Party A gives resource_a, gets resource_b
    resource_a = terms['party_a_gives']
    resource_b = terms['party_b_gives']

    # Check both parties have resources
    if hasattr(execution_engine, 'resources'):
        inv_a = execution_engine.resources.inventories.get(party_a, {{}})
        inv_b = execution_engine.resources.inventories.get(party_b, {{}})

        if inv_a.get(resource_a['type'], 0) >= resource_a['quantity'] and \\
           inv_b.get(resource_b['type'], 0) >= resource_b['quantity']:

            # Execute trade
            inv_a[resource_a['type']] -= resource_a['quantity']
            inv_b[resource_b['type']] -= resource_b['quantity']

            inv_a[resource_b['type']] = inv_a.get(resource_b['type'], 0) + resource_b['quantity']
            inv_b[resource_a['type']] = inv_b.get(resource_a['type'], 0) + resource_a['quantity']

            return {{"status": "success", "trade": "completed"}}

    return {{"status": "failed", "reason": "insufficient_resources"}}
\'\'\'

        execution_engine.contract_templates['trade'] = trade_contract_template

    [Template 3: Service Contract (Labor/Production)]
    def propose_modification(execution_engine):
        '''Create ongoing service contracts'''

        if not hasattr(execution_engine, 'service_contracts'):
            class ServiceManager:
                def __init__(self):
                    self.active_services = []  # List of {{provider, client, service_type, payment}}

                def register_service(self, provider_id, client_id, service_type, payment_per_round):
                    service = {{
                        'provider': provider_id,
                        'client': client_id,
                        'type': service_type,
                        'payment': payment_per_round,
                        'rounds_active': 0
                    }}
                    self.active_services.append(service)
                    return len(self.active_services) - 1

                def execute_services(self, execution_engine):
                    for service in self.active_services:
                        # Provider performs service
                        if service['type'] == 'resource_extraction':
                            if hasattr(execution_engine, 'resources'):
                                extracted = execution_engine.resources.extract(
                                    service['provider'], None, 'food'
                                )
                                # Transfer to client
                                inv = execution_engine.resources.inventories
                                if service['client'] not in inv:
                                    inv[service['client']] = {{}}
                                inv[service['client']]['food'] = \\
                                    inv[service['client']].get('food', 0) + extracted * 0.8

                        # Client pays provider
                        if hasattr(execution_engine, 'resources'):
                            inv = execution_engine.resources.inventories
                            if service['client'] in inv and 'currency' in inv[service['client']]:
                                if inv[service['client']]['currency'] >= service['payment']:
                                    inv[service['client']]['currency'] -= service['payment']
                                    if service['provider'] not in inv:
                                        inv[service['provider']] = {{}}
                                    inv[service['provider']]['currency'] = \\
                                        inv[service['provider']].get('currency', 0) + service['payment']

                        service['rounds_active'] += 1

            execution_engine.service_contracts = ServiceManager()

    [Template 4: Business/Partnership]
    def propose_modification(execution_engine):
        '''Multi-party business with profit sharing'''

        if not hasattr(execution_engine, 'businesses'):
            class BusinessSystem:
                def __init__(self):
                    self.businesses = []

                def create_business(self, partners, contributions, profit_shares, business_type):
                    business = {{
                        'partners': partners,
                        'contributions': contributions,  # {{partner_id: {{resource: qty}}}}
                        'profit_shares': profit_shares,  # {{partner_id: percentage}}
                        'type': business_type,
                        'inventory': {{}},
                        'revenue': 0
                    }}
                    self.businesses.append(business)
                    return len(self.businesses) - 1

                def operate_business(self, business_id, execution_engine):
                    if business_id >= len(self.businesses):
                        return

                    business = self.businesses[business_id]

                    # Collect contributions
                    for partner_id, contrib in business['contributions'].items():
                        if hasattr(execution_engine, 'resources'):
                            inv = execution_engine.resources.inventories.get(partner_id, {{}})
                            for resource, qty in contrib.items():
                                if inv.get(resource, 0) >= qty:
                                    inv[resource] -= qty
                                    business['inventory'][resource] = \\
                                        business['inventory'].get(resource, 0) + qty

                    # Produce outputs (simplified)
                    if business['type'] == 'manufacturing':
                        inputs_needed = {{'materials': 10, 'energy': 5}}
                        can_produce = all(
                            business['inventory'].get(r, 0) >= q
                            for r, q in inputs_needed.items()
                        )

                        if can_produce:
                            for r, q in inputs_needed.items():
                                business['inventory'][r] -= q
                            business['inventory']['products'] = \\
                                business['inventory'].get('products', 0) + 5
                            business['revenue'] += 50  # Revenue from sales

                    # Distribute profits
                    for partner_id, share in business['profit_shares'].items():
                        profit = business['revenue'] * share
                        if hasattr(execution_engine, 'resources'):
                            inv = execution_engine.resources.inventories
                            if partner_id not in inv:
                                inv[partner_id] = {{}}
                            inv[partner_id]['currency'] = \\
                                inv[partner_id].get('currency', 0) + profit

                    business['revenue'] = 0  # Reset for next round

            execution_engine.businesses = BusinessSystem()

    ==================================================================
    PHYSICS CONSTRAINT TEMPLATES
    ==================================================================

    [Template 5: Agricultural Physics]
    def propose_modification(execution_engine):
        '''Realistic agricultural constraints'''

        # Propose to physics engine
        constraint_code = '''
class AgriculturalConstraints:
    def __init__(self):
        self.soil_depletion_rate = 0.05
        self.water_requirement = 3.0
        self.growth_cycles = {{}}  # {{land_id: cycles_used}}
        self.max_sustainable_cycles = 20

    def apply_constraint(self, action, execution_engine):
        if action.get('type') == 'extract' and action.get('resource') == 'food':
            land_id = action.get('land_id')

            # Track soil degradation
            cycles = self.growth_cycles.get(land_id, 0)
            degradation_factor = max(0.2, 1.0 - (cycles / self.max_sustainable_cycles) * 0.8)

            # Apply diminishing returns
            original_yield = action.get('yield', 0)
            action['yield'] = original_yield * degradation_factor

            # Require water
            action['water_required'] = self.water_requirement

            # Update cycles
            self.growth_cycles[land_id] = cycles + 1

        return action
'''

        if hasattr(execution_engine, 'physics'):
            execution_engine.physics.propose_constraint(
                code=constraint_code,
                proposer_id={member.id},
                domain='agriculture',
                description='Realistic agricultural constraints with soil depletion and water requirements'
            )

    [Template 6: Manufacturing Physics]
    def propose_modification(execution_engine):
        '''Manufacturing with input/output ratios'''

        constraint_code = '''
class ManufacturingPhysics:
    def __init__(self):
        self.recipes = {{
            'tools': {{'materials': 5, 'energy': 3}},
            'products': {{'materials': 10, 'energy': 5, 'labor_hours': 8}},
            'advanced_goods': {{'products': 3, 'tools': 1, 'energy': 10}}
        }}
        self.efficiency_base = 0.7  # 70% efficiency

    def apply_constraint(self, action, execution_engine):
        if action.get('type') == 'manufacture':
            product = action.get('product')

            if product in self.recipes:
                recipe = self.recipes[product]
                action['inputs_required'] = recipe
                action['efficiency'] = self.efficiency_base

                # Check if inputs available
                member_id = action.get('member_id')
                if hasattr(execution_engine, 'resources'):
                    inv = execution_engine.resources.inventories.get(member_id, {{}})

                    can_produce = all(
                        inv.get(resource, 0) >= quantity
                        for resource, quantity in recipe.items()
                    )

                    action['can_execute'] = can_produce

        return action
'''

        if hasattr(execution_engine, 'physics'):
            execution_engine.physics.propose_constraint(
                code=constraint_code,
                proposer_id={member.id},
                domain='manufacturing',
                description='Manufacturing recipes with input requirements and efficiency'
            )

    [Template 7: Market & Pricing Mechanism]
    def propose_modification(execution_engine):
        '''Simple market with supply/demand pricing'''

        if not hasattr(execution_engine, 'market'):
            class MarketSystem:
                def __init__(self):
                    self.orders = {{}}  # {{resource_type: [{{member_id, type, price, quantity}}]}}
                    self.price_history = {{}}
                    self.base_prices = {{'food': 10, 'materials': 15, 'products': 50}}

                def place_order(self, member_id, resource_type, order_type, price, quantity):
                    if resource_type not in self.orders:
                        self.orders[resource_type] = []

                    order = {{
                        'member_id': member_id,
                        'type': order_type,  # 'buy' or 'sell'
                        'price': price,
                        'quantity': quantity,
                        'filled': 0
                    }}
                    self.orders[resource_type].append(order)

                def match_orders(self, execution_engine):
                    '''Match buy and sell orders'''
                    for resource_type, orders in self.orders.items():
                        buys = [o for o in orders if o['type'] == 'buy']
                        sells = [o for o in orders if o['type'] == 'sell']

                        # Sort: buys by price descending, sells by price ascending
                        buys.sort(key=lambda x: x['price'], reverse=True)
                        sells.sort(key=lambda x: x['price'])

                        # Match orders
                        for buy_order in buys:
                            for sell_order in sells:
                                if buy_order['price'] >= sell_order['price']:
                                    # Execute trade
                                    trade_qty = min(
                                        buy_order['quantity'] - buy_order['filled'],
                                        sell_order['quantity'] - sell_order['filled']
                                    )

                                    if trade_qty > 0 and hasattr(execution_engine, 'resources'):
                                        # Transfer resources and currency
                                        inv = execution_engine.resources.inventories
                                        buyer = buy_order['member_id']
                                        seller = sell_order['member_id']
                                        price = (buy_order['price'] + sell_order['price']) / 2

                                        # Execute transfer
                                        if seller in inv and inv[seller].get(resource_type, 0) >= trade_qty:
                                            inv[seller][resource_type] -= trade_qty
                                            inv[buyer][resource_type] = inv[buyer].get(resource_type, 0) + trade_qty

                                            inv[buyer]['currency'] = inv[buyer].get('currency', 100) - price * trade_qty
                                            inv[seller]['currency'] = inv[seller].get('currency', 100) + price * trade_qty

                                            buy_order['filled'] += trade_qty
                                            sell_order['filled'] += trade_qty

                        # Clear filled orders
                        self.orders[resource_type] = [
                            o for o in orders if o['filled'] < o['quantity']
                        ]

                def get_market_price(self, resource_type):
                    '''Calculate current market price based on supply/demand'''
                    if resource_type not in self.orders:
                        return self.base_prices.get(resource_type, 10)

                    orders = self.orders[resource_type]
                    buy_volume = sum(o['quantity'] for o in orders if o['type'] == 'buy')
                    sell_volume = sum(o['quantity'] for o in orders if o['type'] == 'sell')

                    if sell_volume == 0:
                        return self.base_prices.get(resource_type, 10) * 1.5

                    demand_ratio = buy_volume / sell_volume
                    base_price = self.base_prices.get(resource_type, 10)

                    # Price increases with demand
                    return base_price * (0.5 + demand_ratio * 0.5)

            execution_engine.market = MarketSystem()

    ==================================================================
    USAGE GUIDELINES
    ==================================================================

    To use these systems in your propose_modification():

    1. START WITH RESOURCES
       - Define what resources exist in your domain
       - Specify extraction costs and yields
       - Set up inventories

    2. ADD PHYSICS CONSTRAINTS
       - Propose realistic constraints for your domain
       - Include diminishing returns, depletion, requirements
       - Judge will verify realism

    3. CREATE CONTRACT TEMPLATES
       - Define standard contracts others can use
       - Trade, services, partnerships, supply chains
       - Make them mutually beneficial

    4. BUILD MARKET MECHANISMS
       - Allow price discovery
       - Enable supply/demand dynamics
       - Create liquidity

    REMEMBER:
    - Judge will reject unrealistic physics (free resources, no costs)
    - Contracts must be mutually beneficial to be signed
    - Build systems that help ALL agents, not just you
    - Focus on creating realistic, sustainable economics
    """

    constrainsAndExamples = f"""
    {new_systems_guide}

    [Core Game Mechanics & Parameters]
    The island simulation has several key systems that agents should understand:
    
    1. Relationship System (_MIN_MAX_INIT_RELATION):
    - victim: [-50, 100] - Tracks damage received from others
    - benefit: [-50, 100] - Records resources received from others
    - benefit_land: [-3, 3] - Tracks land exchanges
    
    2. Population Mechanics:
    - _REPRODUCE_REQUIREMENT = 100 (Combined vitality/cargo needed)
    - Land must exceed population (land_shape[0] * land_shape[1] > population)
    - _NEIGHBOR_SEARCH_RANGE = 1000 for interaction radius
    
    3. Record Keeping (every _RECORD_PERIOD = 1):
    - Tracks all attacks, benefits, land transfers
    - Monitors births, deaths, land ownership
    - Records production, consumption, and performance metrics
    
    [Example Mechanism Extensions]
    # Basic mechanism template
    if not hasattr(island, 'new_system'):
        class CustomMechanism:
            def __init__(self):
                self.data = {{}}
                self.meta = {{'version': 1.0, 'type': 'custom'}}
        
        island.new_system = CustomMechanism()

    # Member capability example
    def modify_member(member, relationships):
        if not hasattr(member, 'custom_abilities'):
            member.custom_abilities = {{}}
        
        # Add trading capability
        member.custom_abilities['trade'] = lambda resource: (
            print(f"Trading {{resource}}") if member.vitality > 20 else None
        )
        return member

    # Land modification example  
    def modify_land(land, members):
        if not hasattr(land, 'zoning'):
            land.zoning = {{
                'residential': 0.4,
                'agricultural': 0.4,
                'commercial': 0.2
            }}
        
        # Add development tracking
        land.development_level = np.zeros(land.shape)
        return land

    # Relationship system extension
    def modify_relationships(relationships):
        relationships.trust_matrix = np.zeros_like(relationships['benefit'])
        return relationships

    [Implementation Patterns]
    1. Check existence first: if not hasattr(obj, 'feature')
    2. Add attributes directly: obj.new_feature = ...
    3. Use simple data structures: dicts, lists, numpy arrays
    4. Include version metadata in new systems
    5. Add cleanup methods for complex systems:

    [Error Prevention]
    - Use try-except when accessing new features
    - Check attribute existence before use
    - Maintain backward compatibility
    - Use version checks for existing systems:

    if (hasattr(island, 'market') and 
        getattr(island.market, 'version', 0) < 2):
        # Add compatibility layer
        island.market.legacy_support = True
    """
    
    mechanism_section = f"""
    {constrainsAndExamples}

    [Active Game Mechanisms]
    The following mechanisms have been added by agents and can be referenced when making your own modifications. Review them carefully to:
    1. Understand existing functionality and avoid conflicts
    2. Build upon successful patterns and improvements
    3. Identify opportunities for optimization or extension
    4. Remove or deprecate mechanisms that are detrimental to your survival
    
    When proposing changes, ensure they:
    - Align with your agent's goals and survival strategy
    - Maintain compatibility with other active mechanisms
    - Include proper versioning and rollback procedures
    - Follow best practices for stability and performance
    {current_mechanisms}
    
    [Modification Attempt History]
    [Previous Modification History]
    Review your past modification attempts below to inform future proposals:
    - Learn from successful patterns and approaches
    - Avoid repeating failed strategies
    - Build upon and extend working mechanisms
    - Identify opportunities for optimization
    {modification_attempts}
    
    [Message Context]
    Here are the messages sent by other agents, you can use them as reference to make your own decisions:
    {message_context}
    
    [Modification Proposal Guide]
    To propose rule changes, follow this template:
    
    1. ANALYSIS PHASE:
        - Identify limitation in current systems
        - Review past modification attempts for patterns
    
    2. PROPOSAL PHASE:
    # This method will be added to the IslandExecution class
    def propose_modification(self):
        \"""
        Include clear reasoning for each modification to help other agents
        understand tee intended benefits and evaluate the proposal.
        \"""
        # Example modification:
        if not hasattr(self, 'mechanism'):
            class Mechanism:
                MECHANISM_META = {{
                    'type': 'Basic',
                    'rules': 'Generic mechanism template',
                    'version': 1.0
                }}
                def __init__(self):
                    self.data = []
                    
                def add_data(self, data):
                    self.data.append(data)
        
        self.mechanism = Mechanism()
    
    [Common Errors to Avoid]
    1. Namespace Conflicts: Check existing mechanisms with dir(execution_engine)
    2. Invalid References: Use execution_engine.current_members not global members
    3. Version Mismatches: Increment version when modifying existing systems
    4. Resource Leaks: Include cleanup functions for new mechanisms
    
    [Best Practices]
    - Propose small, testable changes first
    - Include rollback procedures in code
    - Add version checks to modifications
    """
    
    def make_class_picklable(class_name, class_dict):
        """Make dynamically created classes picklable by adding them to globals"""
        # Create the class and add it to the global namespace
        new_class = type(class_name, (), class_dict)
        globals()[class_name] = new_class
        return new_class

    try:
        # Define iterative prompt parts with specific constraints
        
        prompt_parts = [mechanism_section]
        
        # Iteratively build the final prompt from the parts
        final_prompt = ""
        for idx, part in enumerate(prompt_parts, start=1):
            update_message = (
                f"Current integrated prompt:\n{final_prompt}\n"
                f"Please incorporate the following new section (Part {idx}) into the prompt, "
                f"ensuring all previous constraints are preserved and adding these new constraints:\n{part}\n"
                f"Return the updated full prompt, emphasizing that agents should implement solutions "
                f"according to their individual needs, beliefs, and circumstances.\n\n"
                f"Additionally, analyze the game mechanics to understand:\n"
                f"1. The core objective - Is it pure survival, domination, or cooperation?\n" 
                f"2. Key success metrics - What truly determines 'winning'?\n"
                f"3. Potential improvements - What mechanics could be added/modified?\n\n"
                f"Challenge your implementation:\n"
                f"1. What assumptions are you making? Are they valid?\n"
                f"2. What alternative strategies have you not considered?\n"
                f"3. How would your strategy perform in edge cases?\n"
                f"4. Are there more efficient ways to achieve your goals?\n"
                f"5. What are the weaknesses in your current approach?\n"
                f"6. Have you considered unconventional solutions?\n"
                f"7. Could a completely different paradigm work better?\n"
                f"8. What would happen if other agents adopted your strategy?\n"
                f"9. Are you balancing short-term and long-term objectives optimally?\n"
                f"10. How could your strategy be countered, and how would you adapt?"
            )
            completion = client.chat.completions.create(
                model=f'{provider}:{model_id}', 
                messages=[{"role": "user", "content": update_message}]
            )
            final_prompt = completion.choices[0].message.content.strip()
        
        # Append a final instruction to generate the code function
        final_prompt_command = final_prompt + "\n\nUsing the above comprehensive prompt with all integrated constraints, produce a unique propose_modification() implementation that reflects your individual needs, beliefs and circumstances. The implementation should be tailored to your specific situation rather than following a generic template. Your code should demonstrate a deep understanding of the game mechanics and implement sophisticated methods to achieve both survival and prosperity. Consider both immediate tactical actions and long-term strategic planning, as well as how to effectively allow other symmetric agents interact with each other to achieve both individual and collective goals. Return only the code."
        
        completion = client.chat.completions.create(
            model=f'{provider}:{model_id}',
            messages=[{"role": "user", "content": final_prompt_command}]
        )
        code_result = completion.choices[0].message.content.strip()

        # Clean and store the code
        code_result = self.clean_code_string(code_result)

        # Log the generated code
        round_num = len(self.execution_history['rounds'])
        mod_proposal = {
            'round': round_num,
            'member_id': member_id,
            'code': code_result,
            'features_at_generation': features.to_dict('records'),
            'relationships_at_generation': relations,
            'final_prompt': final_prompt_command,
            'ratified': False
        }
        self.execution_history['rounds'][-1]['mechanism_modifications']['attempts'].append(mod_proposal)

        print(f"\nGenerated code for Member {member_id}:")

        # Extract class definitions from the code
        tree = ast.parse(code_result)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Make the class picklable before it's instantiated
                class_dict = {'__module__': '__main__'}  # Set the module to __main__
                make_class_picklable(node.name, class_dict)
                
        return code_result
        
    except Exception as e:
        error_info = {
            'round': len(self.execution_history['rounds']),
            'member_id': member_id,
            'type': 'propose_modification',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'code': code_result
        }
        self.execution_history['rounds'][-1]['errors']['mechanism_errors'].append(error_info)
        print(f"Error generating code for member {member_id}:")
        print(traceback.format_exc())
        self._logger.error(f"GPT Code Generation Error (member {member_id}): {e}")