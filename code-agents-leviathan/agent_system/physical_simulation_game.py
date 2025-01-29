class PhysicalSimulationGame:
    def __init__(self, agents):
        """
        Initialize the physical simulation environment with a list of agents.
        Each agent will provide code that modifies an existing structure
        to withstand higher pressure.
        """
        self.agents = agents
        # Now store a list of versioned structures with 'design_state' and 'pressure_resistance'
        self.structures = []
        self.reward_pool = 1000
        self.agent_contributions = {}
        for agent in agents:
            self.agent_contributions[agent.id] = 0.0

    def approximate_mechanical_calculation(self, design_state):
        """
        A rudimentary formula to compute overall pressure resistance:
        Each item in 'existing_parts' adds +10 to a base value, for demonstration purposes.
        You can expand or customize this to reflect more realistic factors.
        """
        base_resistance = 50
        part_count = len(design_state.get('existing_parts', []))
        # For each part, add 10. You could add more logic if certain parts are stronger/weaker.
        total_resistance = base_resistance + (10 * part_count)
        return float(total_resistance)

    def get_simulation_state(self):
        """
        Return the current state of the simulation, including the latest structure
        and any relevant metrics.
        """
        # If no structures yet, return an empty "design_state"
        latest_design = self.structures[-1]["design_state"] if self.structures else {}
        return {
            "latest_design": latest_design,
            "pressure_environment": 100  # A baseline pressure value
        }

    def validate_structure_code(self, code_snippet):
        """
        Basic validation for structure-building code. Here, we ensure the code:
        1. Defines a function named 'build_structure(design_state)'
        2. The function returns a dictionary with keys 'design_state' and 'pressure_resistance'
        3. 'pressure_resistance' must be a float or integer
        """
        required_function_name = 'build_structure'
        try:
            namespace = {}
            exec(code_snippet, namespace)
            if required_function_name not in namespace:
                print("[Validation] Missing build_structure() function.")
                return False

            # Provide a dummy 'design_state' for testing
            dummy_state = {"existing_parts": ["test_part"]}
            result = namespace[required_function_name](dummy_state)
            if not isinstance(result, dict):
                print("[Validation] build_structure() must return a dictionary.")
                return False

            if 'design_state' not in result or 'pressure_resistance' not in result:
                print("[Validation] Missing required keys: 'design_state' or 'pressure_resistance'.")
                return False

            if not isinstance(result['pressure_resistance'], (float, int)):
                print("[Validation] pressure_resistance must be numeric.")
                return False

            return True
        except Exception as e:
            print(f"[Validation] Structure code failed to execute: {e}")
            return False

    def integrate_structure(self, agent_id, code_snippet):
        """
        If the code passes validation, execute it to obtain the updated structure details.
        The structure is now defined by code in 'design_state["structure_code"]'.
        We compile that code for validation, then override any numeric pressure value
        with our approximate mechanical calculation.
        """
        if not self.validate_structure_code(code_snippet):
            print(f"[Integration] Agent {agent_id} structure integration failed.")
            return False
        
        # Run the agent code, passing in the most recent design
        latest_design = self.structures[-1]['design_state'] if self.structures else {}

        namespace = {}
        exec(code_snippet, namespace)
        updated_structure = namespace['build_structure'](latest_design)

        # ---------------------------------------------------------------------
        # NEW LOGIC: If agent provides "structure_code" in design_state, compile it.
        structure_code = updated_structure['design_state'].get('structure_code')
        if structure_code:
            try:
                compile(structure_code, "<structure_code>", "exec")
                print("[Integration] structure_code compiled successfully.")
            except Exception as e:
                print(f"[Integration] structure_code failed to compile: {e}")
                return False
        # ---------------------------------------------------------------------

        # Override with rough mechanical calculation
        updated_structure['pressure_resistance'] = self.approximate_mechanical_calculation(
            updated_structure['design_state']
        )

        self.structures.append({
            'design_state': updated_structure['design_state'],
            'pressure_resistance': updated_structure['pressure_resistance'],
            'agent_id': agent_id
        })
        self.agent_contributions[agent_id] += 1.0
        print(
            f"[Integration] Agent {agent_id} integrated code-based structure. "
            f"Pressure resistance = {updated_structure['pressure_resistance']}"
        )
        return True

    def run_simulation_cycle(self):
        """
        Simulate the environment by allowing each agent to provide code that updates
        the current structure. We pass the existing design so that each agent can
        build upon prior contributions. Then, distribute rewards based on pressure resistance.
        """
        print("\n=== Starting Physical Simulation Cycle ===")
        sim_state = self.get_simulation_state()
        baseline_pressure = sim_state["pressure_environment"]
        latest_design = sim_state["latest_design"]

        # Let each agent propose structure-building code
        for agent in self.agents:
            print(f"\n[Agent {agent.id}] Generating update code...")
            build_prompt = f"""
            # You are updating an existing structure to withstand more pressure. 
            # The current design_state is: {latest_design}
            #
            # Requirements:
            # 1. Must define a function build_structure(design_state) -> dict.
            #    This function returns a new dictionary:
            #       {{
            #         'design_state': <updated design>,
            #         'pressure_resistance': <float or int representing pressure tolerance>
            #       }}
            # 2. Only basic Python is allowed.
            # 3. The environment's baseline pressure is {baseline_pressure}.
            # 
            # Example snippet:
            # def build_structure(design_state):
            #     # read from design_state and add new parts...
            #     design_state['existing_parts'].append('some_new_part')
            #     return {{
            #         'design_state': design_state,
            #         'pressure_resistance': 120.0
            #     }}
            """

            generated_code = agent.generate_code(build_prompt, context=None)
            print(f"[Agent {agent.id}] Generated code: {generated_code[:150]}...")
            self.integrate_structure(agent.id, generated_code)

        # Reward distribution
        self.distribute_rewards()

    def distribute_rewards(self):
        """
        Reward agents based on their structures' pressure resistance 
        relative to the total sum of all contributions across all versions.
        """
        if not self.structures:
            print("[Rewards] No structures to evaluate.")
            return

        total_resistance = sum(s['pressure_resistance'] for s in self.structures if s)
        if total_resistance == 0:
            print("[Rewards] No valid pressure resistance data in structures.")
            return

        # For each structure version, figure out that version's share of the total resistance
        # and reward the agent who created it proportionally.
        structure_rewards = {}
        for s in self.structures:
            agent_id = s['agent_id']
            pressure = s['pressure_resistance']
            share = (pressure / total_resistance) * self.reward_pool if total_resistance > 0 else 0
            structure_rewards[agent_id] = structure_rewards.get(agent_id, 0) + share

        # Apply rewards to each agent
        for agent in self.agents:
            agent_reward = structure_rewards.get(agent.id, 0)
            agent.reward += agent_reward
            print(f"[Rewards] Agent {agent.id} receives a reward of {agent_reward:.2f}.")

        # Decay contributions for the next cycle
        for agent_id in self.agent_contributions:
            self.agent_contributions[agent_id] *= 0.9 