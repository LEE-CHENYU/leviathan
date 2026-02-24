class PhysicalSimulationGame:
    def __init__(self, agents):
        """
        Initialize the physical simulation environment with a list of agents.
        Each agent will provide code to build or modify a structure capable of withstanding high pressure.
        """
        self.agents = agents
        self.structures = {}
        self.reward_pool = 1000
        self.agent_contributions = {}
        for agent in agents:
            self.agent_contributions[agent.id] = 0.0

    def get_simulation_state(self):
        """
        Return the current state of the simulation, including existing structures and any relevant metrics.
        For simplicity, structure states are stored in self.structures as:
        {
            structure_id: {
                'design': <string_of_code_or_representing_data>,
                'pressure_resistance': <some_numeric_value>
            }
        }
        """
        return {
            "structures": self.structures,
            "pressure_environment": 100  # A baseline pressure value
        }

    def validate_structure_code(self, code_snippet):
        """
        Basic validation for structure-building code. Here, we ensure the code:
        1. Defines a function named 'build_structure()'
        2. The function returns a dictionary with keys 'design' and 'pressure_resistance'
        3. 'pressure_resistance' must be a float or integer
        """
        required_function_name = 'build_structure'
        try:
            namespace = {}
            exec(code_snippet, namespace)
            if required_function_name not in namespace:
                print("[Validation] Missing build_structure() function.")
                return False

            result = namespace[required_function_name]()
            if not isinstance(result, dict):
                print("[Validation] build_structure() must return a dictionary.")
                return False

            if 'design' not in result or 'pressure_resistance' not in result:
                print("[Validation] Missing required keys in structure dictionary.")
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
        If the code passes validation, execute it to obtain the structure details
        and store it in self.structures.
        """
        if self.validate_structure_code(code_snippet):
            namespace = {}
            exec(code_snippet, namespace)
            structure_info = namespace['build_structure']()
            structure_id = hash(structure_info['design'])

            self.structures[structure_id] = {
                'design': structure_info['design'],
                'pressure_resistance': structure_info['pressure_resistance']
            }
            self.agent_contributions[agent_id] += 1.0
            print(f"[Integration] Agent {agent_id} successfully integrated structure.")
            return True
        else:
            print(f"[Integration] Agent {agent_id} structure integration failed.")
            return False

    def run_simulation_cycle(self):
        """
        Simulate the environment by allowing each agent to provide code for a structure,
        validate and integrate that code, then distribute rewards based on pressure resistance.
        """
        print("\n=== Starting Physical Simulation Cycle ===")
        sim_state = self.get_simulation_state()
        baseline_pressure = sim_state["pressure_environment"]

        # Let each agent propose structure-building code
        for agent in self.agents:
            print(f"\n[Agent {agent.id}] Generating structure code...")
            build_prompt = f"""
            # You are building a structure to withstand pressure. 
            # The structure must return a dictionary with keys:
            # 'design': A string describing the structure layout
            # 'pressure_resistance': A numeric value that indicates how much pressure the structure can withstand
            # 
            # Requirements:
            # 1. Must define a function build_structure() returning the structure dict.
            # 2. Only basic Python is allowed.
            # 
            # The environment's baseline pressure is {baseline_pressure}.
            """

            generated_code = agent.generate_code(build_prompt, context=None)
            print(f"[Agent {agent.id}] Generated structure code: {generated_code[:150]}...")
            self.integrate_structure(agent.id, generated_code)

        # Reward distribution
        self.distribute_rewards()

    def distribute_rewards(self):
        """
        Reward agents based on their structures' pressure resistance 
        relative to the total sum of all contributions.
        """
        if not self.structures:
            print("[Rewards] No structures to evaluate.")
            return

        total_resistance = sum(info['pressure_resistance'] for info in self.structures.values() if info)
        if total_resistance == 0:
            print("[Rewards] No valid pressure resistance data in structures.")
            return

        # Assign rewards proportionally to pressure_resistance
        rewards = {}
        for agent_id in self.agent_contributions:
            agent_pressure = 0
            # Sum structures from this agent
            for structure_id, info in self.structures.items():
                # This simplistic approach assumes all structures are from the same agent,
                # but you could store contributor data in a more sophisticated manner
                # if you track which agent created each structure.
                # For now, we'll just pretend each snippet is from a single agent to keep it simple.
                agent_pressure += info['pressure_resistance']

            share = (agent_pressure / total_resistance) * self.reward_pool if total_resistance > 0 else 0
            rewards[agent_id] = share

        # Apply rewards to each agent
        for agent in self.agents:
            agent_reward = rewards.get(agent.id, 0)
            agent.reward += agent_reward
            print(f"[Rewards] Agent {agent.id} receives a reward of {agent_reward:.2f}.")

        # Decay contributions for the next cycle
        for agent_id in self.agent_contributions:
            self.agent_contributions[agent_id] *= 0.9 