from turtle import filling
from altair import Align
import openai
import traceback

from dotenv import load_dotenv
import aisuite as ai

from MetaIsland.model_router import model_router

load_dotenv()

client = ai.Client()

provider, model_id = model_router("deepseek")
    
async def _agent_code_decision(self, member_id) -> None:
    """
    Asks GPT for directly executable Python code, stores it in a dictionary keyed by member_id.
    The code will define a function agent_action(execution_engine, member_id), 
    which references attributes that actually exist.
    """
    data = self.prepare_agent_data(member_id)
    member = data['member']
    relations = data['relations']
    features = data['features']
    code_memory = data['code_memory']
    analysis_memory = data['analysis_memory']
    past_performance = data['past_performance']
    error_context = data['error_context']
    message_context = data['message_context']

    current_mechanisms = data['current_mechanisms']
    modification_attempts = data['modification_attempts'][max(data['modification_attempts'].keys())]
    report = data['report']
    
    base_code = self.base_class_code
    
    try:
        # Define iterative prompt parts with specific constraints
        base_code = f"""
        [Base Code]
        Here is the base code for the Island and Member classes that you should reference when making your actions. Study the mechanisms carefully to ensure your code interacts correctly with the available attributes and methods. Pay special attention to:
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

    [Current task]
    You are member_{member.id} in a society that you can help shape.
    Write a Python function named agent_action(execution_engine, member_id) that implements your vision of social organization while ensuring your survival.
    You should use methods and methods of objects defined in in [Active Mechanisms Modifications] section to make your actions in agent_action(execution_engine, member_id), but DO NOT define another propose_modification(execution_engine) itself in your code.

    [Active Mechanisms Modifications]
    You should use following mechanisms have been added by other agents in your code:
    - Review them carefully to understand their functionality and constraints
    - Leverage compatible mechanisms that align with your goals
    - Be mindful of version requirements and dependencies
    - Consider how they can be combined strategically
    - Test interactions before relying on them critically
    {current_mechanisms}
    
    [Island Ideology]
    {self.island_ideology}
    
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
    Here are the basic information of all members, you should make your own decisions based on them.
    {features}

    Relationship summary (parsed from relationship_dict):
    Here are the relationships between members:
    {relations}
        
    Code Memory and Previous Performance:
    {code_memory}
    
    Performance history:
    {past_performance}
    
    Analysis Memory:
    {analysis_memory}

    Based on the previous code performance, adapt and improve the strategy.
    If a previous strategy worked well (high performance), consider building upon it.
    If it failed, try a different approach.
    
    IMPORTANT: Do not simply copy the example implementation below. Instead, use it as inspiration to create your own unique approach combining different methods and strategies in novel ways.
    
    Your code should include agent_action() function. Do not define another propose_modification() from the [Active Mechanisms Modifications] section instead use the methods and methods of objects defined in the [Active Mechanisms Modifications] section.
    def agent_action(execution_engine, member_id):
        \"""
        Include clear reasoning for each modification to help other agents
        understand tee intended benefits and evaluate the proposal.
        You should use methods and methods of objects defined in in [Active Mechanisms Modifications] section to make your actions in agent_action(execution_engine, member_id), but DO NOT define another propose_modification(execution_engine) itself in your code.
        State if you used any of the mechanisms defined in [Active Mechanisms Modifications] in your code.
        \"""
        <Write your own implementation here>
    
    While the example above shows one possible approach,
    you should create your own unique implementation drawing from the wide range of available methods and strategies.
    
    Consider novel combinations of different approaches rather than following this exact pattern.
    """

        # NEW: Contract Actions and Economic Strategy
        contract_and_market_guide = f"""
        {part0}

    ==================================================================
    CONTRACT ACTIONS - USE THE NEW SYSTEMS
    ==================================================================

    You now can interact with:
    - execution_engine.contracts (ContractEngine)
    - execution_engine.resources (if created by agents)
    - execution_engine.market (if created by agents)
    - execution_engine.businesses (if created by agents)

    [Action Template 1: Propose a Trade Contract]
    def agent_action(execution_engine, member_id):
        '''Propose trade contracts with other agents'''

        if hasattr(execution_engine, 'contracts') and hasattr(execution_engine, 'resources'):
            # Check my inventory
            my_inv = execution_engine.resources.inventories.get(member_id, {{}})

            # Find trading partners
            for other_id in range(len(execution_engine.current_members)):
                if other_id == member_id:
                    continue

                other_inv = execution_engine.resources.inventories.get(other_id, {{}})

                # Propose trade if I have food and they have materials
                if my_inv.get('food', 0) > 10 and other_inv.get('materials', 0) > 5:
                    trade_contract = '''
def execute_contract(execution_engine, context):
    contract = context.get('contract', {{}})
    party_a = contract['parties'][0]  # Me
    party_b = contract['parties'][1]  # Them
    terms = contract['terms']

    if hasattr(execution_engine, 'resources'):
        inv = execution_engine.resources.inventories

        # I give food, get materials
        if inv.get(party_a, {{}}).get('food', 0) >= 10 and \\
           inv.get(party_b, {{}}).get('materials', 0) >= 5:
            inv[party_a]['food'] -= 10
            inv[party_b]['materials'] -= 5
            inv[party_a]['materials'] = inv[party_a].get('materials', 0) + 5
            inv[party_b]['food'] = inv[party_b].get('food', 0) + 10
            return {{"status": "success"}}
    return {{"status": "failed"}}
'''

                    execution_engine.contracts.propose_contract(
                        code=trade_contract,
                        proposer_id=member_id,
                        parties=[member_id, other_id],
                        terms={{'type': 'trade', 'my_offer': 'food', 'their_offer': 'materials'}}
                    )

    [Action Template 2: Sign Pending Contracts]
    def agent_action(execution_engine, member_id):
        '''Review and sign beneficial contracts'''

        if hasattr(execution_engine, 'contracts'):
            # Get contracts I'm involved in
            pending = execution_engine.contracts.get_contracts_for_party(member_id, 'pending')

            for contract in pending:
                # Already signed by me?
                if member_id in contract.get('signatures', {{}}):
                    continue

                # Evaluate if beneficial
                terms = contract.get('terms', {{}})

                # Simple heuristic: sign if it's a trade and I have the resources
                if terms.get('type') == 'trade':
                    if hasattr(execution_engine, 'resources'):
                        my_inv = execution_engine.resources.inventories.get(member_id, {{}})

                        # Check if I can fulfill my side
                        my_offer = terms.get('my_offer', '')
                        if my_inv.get(my_offer, 0) >= 10:  # Assuming quantity 10
                            # Sign the contract
                            execution_engine.contracts.sign_contract(
                                contract['id'], member_id
                            )

    [Action Template 3: Market Saturation Analysis & Business Pivot]
    def agent_action(execution_engine, member_id):
        '''Analyze market and pivot if saturated'''

        if not hasattr(execution_engine, 'market'):
            return

        market = execution_engine.market

        # Track my current business
        my_product = 'food'  # What I currently produce

        # Calculate market saturation for my product
        if hasattr(market, 'orders') and my_product in market.orders:
            orders = market.orders[my_product]

            total_supply = sum(o['quantity'] for o in orders if o['type'] == 'sell')
            total_demand = sum(o['quantity'] for o in orders if o['type'] == 'buy')

            if total_supply > 0:
                saturation = total_supply / max(total_demand, 1)
            else:
                saturation = 0

            # Count competitors
            competitors = len([o for o in orders if o['type'] == 'sell' and o['member_id'] != member_id])

            # Get current price
            current_price = market.get_market_price(my_product)
            base_price = market.base_prices.get(my_product, 10)

            profit_margin = (current_price - base_price) / base_price if base_price > 0 else 0

            print(f"Market analysis for {{my_product}}:")
            print(f"  Saturation: {{saturation:.2f}}")
            print(f"  Competitors: {{competitors}}")
            print(f"  Profit margin: {{profit_margin:.2%}}")

            # Decision: Pivot if market is saturated
            if saturation > 1.5 or profit_margin < 0.1:
                # Find less saturated market
                best_product = None
                best_saturation = float('inf')

                for product in ['food', 'materials', 'products']:
                    if product == my_product:
                        continue

                    if product in market.orders:
                        p_orders = market.orders[product]
                        p_supply = sum(o['quantity'] for o in p_orders if o['type'] == 'sell')
                        p_demand = sum(o['quantity'] for o in p_orders if o['type'] == 'buy')
                        p_saturation = p_supply / max(p_demand, 1)

                        if p_saturation < best_saturation:
                            best_saturation = p_saturation
                            best_product = product

                if best_product and best_saturation < saturation * 0.7:
                    print(f"  PIVOTING from {{my_product}} to {{best_product}}")
                    print(f"    New market saturation: {{best_saturation:.2f}}")

                    # Store pivot decision (would implement retooling in mechanisms)
                    if not hasattr(execution_engine, 'pivot_decisions'):
                        execution_engine.pivot_decisions = {{}}
                    execution_engine.pivot_decisions[member_id] = {{
                        'from': my_product,
                        'to': best_product,
                        'reason': 'market_saturation',
                        'old_saturation': saturation,
                        'new_saturation': best_saturation
                    }}

    [Action Template 4: Join/Create Business Partnership]
    def agent_action(execution_engine, member_id):
        '''Form business partnerships'''

        if hasattr(execution_engine, 'businesses'):
            businesses = execution_engine.businesses

            # Look for existing businesses to join
            for idx, business in enumerate(businesses.businesses):
                # Check if I can contribute
                if hasattr(execution_engine, 'resources'):
                    my_inv = execution_engine.resources.inventories.get(member_id, {{}})
                    needed = business.get('contributions', {{}})

                    # If I have what they need and I'm not already a partner
                    if member_id not in business.get('partners', []):
                        can_contribute = all(
                            my_inv.get(resource, 0) >= qty
                            for resource, qty in needed.items()
                        )

                        if can_contribute:
                            # Join business (would need contract)
                            print(f"Joining business {{idx}}")
                            # Business joining handled via contracts
        else:
            # Create a new business
            if hasattr(execution_engine, 'contracts'):
                partnership_contract = '''
def execute_contract(execution_engine, context):
    """Create a manufacturing partnership"""
    contract = context.get('contract', {{}})
    partners = contract['parties']

    if not hasattr(execution_engine, 'businesses'):
        return {{"status": "failed", "reason": "no_business_system"}}

    # Create business
    business_id = execution_engine.businesses.create_business(
        partners=partners,
        contributions={{p: {{'materials': 5, 'energy': 3}} for p in partners}},
        profit_shares={{p: 1.0/len(partners) for p in partners}},
        business_type='manufacturing'
    )

    return {{"status": "success", "business_id": business_id}}
'''

                # Propose to potential partners
                for other_id in range(len(execution_engine.current_members)):
                    if other_id != member_id:
                        execution_engine.contracts.propose_contract(
                            code=partnership_contract,
                            proposer_id=member_id,
                            parties=[member_id, other_id],
                            terms={{'type': 'partnership', 'business': 'manufacturing'}}
                        )
                        break  # Propose to first available partner

    [Action Template 5: Extract Resources Using Available Systems]
    def agent_action(execution_engine, member_id):
        '''Extract resources if system available'''

        if hasattr(execution_engine, 'resources'):
            resources = execution_engine.resources

            # Extract food (costs vitality)
            member = execution_engine.current_members[member_id]
            if member.vitality > 20:  # Only if healthy enough
                extracted = resources.extract(member_id, None, 'food')
                print(f"Extracted {{extracted}} food")

                # Pay vitality cost if defined
                resource_def = resources.resource_types.get('food', {{}})
                vitality_cost = resource_def.get('extraction_cost_vitality', 0)
                member.vitality -= vitality_cost

    [Action Template 6: Place Market Orders]
    def agent_action(execution_engine, member_id):
        '''Use market to buy/sell'''

        if hasattr(execution_engine, 'market') and hasattr(execution_engine, 'resources'):
            market = execution_engine.market
            my_inv = execution_engine.resources.inventories.get(member_id, {{}})

            # Sell excess food
            if my_inv.get('food', 0) > 20:
                current_price = market.get_market_price('food')
                # Undercut slightly to sell faster
                my_price = current_price * 0.95

                market.place_order(
                    member_id=member_id,
                    resource_type='food',
                    order_type='sell',
                    price=my_price,
                    quantity=10
                )
                print(f"Placed sell order: 10 food at {{my_price:.2f}}")

            # Buy materials if I need them
            if my_inv.get('materials', 0) < 5:
                current_price = market.get_market_price('materials')
                # Willing to pay a bit more
                my_price = current_price * 1.05

                market.place_order(
                    member_id=member_id,
                    resource_type='materials',
                    order_type='buy',
                    price=my_price,
                    quantity=5
                )
                print(f"Placed buy order: 5 materials at {{my_price:.2f}}")

    [Action Template 7: Supply Chain Management]
    def agent_action(execution_engine, member_id):
        '''Build supply chain through contracts'''

        if not hasattr(execution_engine, 'resources') or not hasattr(execution_engine, 'contracts'):
            return

        my_inv = execution_engine.resources.inventories.get(member_id, {{}})

        # I'm a manufacturer, need suppliers
        if my_inv.get('materials', 0) < 10:  # Low on inputs
            # Find raw material suppliers
            for supplier_id in range(len(execution_engine.current_members)):
                if supplier_id == member_id:
                    continue

                supplier_inv = execution_engine.resources.inventories.get(supplier_id, {{}})

                # They have materials
                if supplier_inv.get('materials', 0) > 20:
                    # Propose ongoing supply contract
                    supply_contract = '''
def execute_contract(execution_engine, context):
    """Ongoing materials supply"""
    contract = context.get('contract', {{}})
    supplier = contract['parties'][0]
    buyer = contract['parties'][1]

    if hasattr(execution_engine, 'resources'):
        inv = execution_engine.resources.inventories

        # Deliver materials
        if inv.get(supplier, {{}}).get('materials', 0) >= 5:
            inv[supplier]['materials'] -= 5
            inv[buyer]['materials'] = inv[buyer].get('materials', 0) + 5

            # Payment
            if inv[buyer].get('currency', 0) >= 20:
                inv[buyer]['currency'] -= 20
                inv[supplier]['currency'] = inv[supplier].get('currency', 0) + 20

            return {{"status": "success", "delivered": 5}}
    return {{"status": "failed"}}
'''

                    execution_engine.contracts.propose_contract(
                        code=supply_contract,
                        proposer_id=member_id,
                        parties=[supplier_id, member_id],  # supplier first
                        terms={{'type': 'supply_chain', 'resource': 'materials', 'quantity': 5, 'payment': 20}}
                    )
                    break

    ==================================================================
    ECONOMIC STRATEGY PATTERNS
    ==================================================================

    [Pattern 1: Specialization Strategy]
    - Focus on one product/service
    - Build reputation and efficiency
    - Rely on trade for other needs

    [Pattern 2: Diversification Strategy]
    - Produce multiple products
    - Hedge against market changes
    - Higher operational complexity

    [Pattern 3: Middleman Strategy]
    - Buy low, sell high
    - Connect supply and demand
    - Profit from market inefficiency

    [Pattern 4: Vertical Integration]
    - Control entire supply chain
    - From raw materials to finished goods
    - Maximum profit capture

    [Pattern 5: Service Provider]
    - Offer labor/services to others
    - Predictable income stream
    - Low capital requirements

    IMPLEMENTATION GUIDE:
    1. Check what systems exist (resources, market, businesses)
    2. Analyze your current position (inventory, contracts, relationships)
    3. Identify opportunities (under-supplied markets, partnership gaps)
    4. Propose contracts that are mutually beneficial
    5. Monitor performance and adapt

    REMEMBER:
    - Contracts must benefit both parties to be signed
    - Market prices reflect supply/demand
    - Pivot when profit margins drop
    - Build supply chains for reliability
    - Diversify risk across multiple relationships
    """

        part1 = f"""
        {contract_and_market_guide}

    [Communication]
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
    Here are the messages sent by other agents, you can use them as reference to make your own decisions:
    {message_context}
        """
        part2 = f"""
                [Voting Mechanism]
    
    Vote for the modfication submitted by you and other agents by adding your vote as an element in voting_box Dict.
    {modification_attempts}
    
    self.voting_box[member_id] = {{
        'reason': string,  
        'yes_votes': List[int] 
    }}
        
        'member_id': ID of the member who is voting
        'reason': Reason for voting
        'yes_votes': Index of mechanism want to support in the mechanism list
        """
        part3 = f"""
        {part0}
        
        [Curfilling Mechanisms]
        The following mechanisms have been added by other agents and are available for your use:
        - Review them carefully to understand their functionality and constraints
        - Leverage compatible mechanisms that Align with your goals
        - Be mindful of version requirements and dependencies
        - Consider how they can be combined strategically
        - Test interactions before relying on them critically
        {current_mechanisms}
        
            [Data Collection and Learning Mechanism]
    Implement sophisticated systems focused on:
    1. Resource optimization and allocation
    2. Multi-factor threat assessment and response
    3. Dynamic vitality management
    4. Strategic resource accumulation
    5. Predictive modeling and adaptation
    6. Social network analysis
    7. Coalition formation tracking
    8. Environmental state monitoring

    [ Survival Metrics]
    Evaluate strategies comprehensively by:
    - Vitality efficiency
    - Resource accumulation rate
    - Threat neutralization effectiveness
    - Territory control metrics
    - Alliance strength indicators
    - Adaptation speed to changes
    - Long-term survival probability
    - Action success ratios
    - Resource conversion efficiency
    - Social influence measures

    [Implementation Priorities]
    1. Create personal health monitoring systems
    2. Develop egocentric threat models
    3. Optimize actions for caloric ROI
    4. Implement fail-deadly safeguards
    5. Build predictive self-preservation models
        """
        
        prompt_parts = [part1, part3]
        
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
        final_prompt_command = final_prompt + "\n\nUsing the above comprehensive prompt with all integrated constraints, produce a unique agent_action() implementation that reflects your individual needs, beliefs and circumstances. The implementation should be tailored to your specific situation rather than following a generic template. Your code should demonstrate a deep understanding of the game mechanics and implement sophisticated strategies to achieve both survival and prosperity. Consider both immediate tactical actions and long-term strategic planning, as well as how to effectively interact with other symmetric agents to achieve both individual and collective goals. Return only the code."
        
        completion = client.chat.completions.create(
            model=f'{provider}:{model_id}',
            messages=[{"role": "user", "content": final_prompt_command}]
        )
        code_result = completion.choices[0].message.content.strip()

        # Clean and store the code
        code_result = self.clean_code_string(code_result)

        # Log the generated code
        round_num = len(self.execution_history['rounds'])
        if round_num not in self.execution_history['rounds'][-1]['generated_code']:
            self.execution_history['rounds'][-1]['generated_code'] = {}

        self.execution_history['rounds'][-1]['generated_code'][member_id] = {
            'code': code_result,
            'features_at_generation': features.to_dict('records'),
            'relationships_at_generation': relations,
            'final_prompt': final_prompt_command  # optionally log the final prompt used
        }

        print(f"\nGenerated code for Member {member_id}:")

        if not hasattr(self, 'agent_code_by_member'):
            self.agent_code_by_member = {}
        self.agent_code_by_member[member_id] = code_result
        
        return code_result

    except Exception as e:
        error_info = {
            'round': len(self.execution_history['rounds']),
            'member_id': member_id,
            'type': 'agent_action',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'code': code_result
        }
        self.execution_history['rounds'][-1]['errors']['agent_code_errors'].append(error_info)
        print(f"Error generating code for member {member_id}:")
        print(traceback.format_exc())
        self._logger.error(f"GPT Code Generation Error (member {member_id}): {e}")