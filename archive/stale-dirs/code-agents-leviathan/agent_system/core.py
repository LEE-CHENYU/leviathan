import requests
from collections import defaultdict
import hashlib  # For safer code snippet hashing

class Agent:
    def __init__(self, agent_id, api_key, api_type='openai'):
        self.id = agent_id
        self.api_key = api_key
        self.api_type = api_type
        self.contributions = 0
        self.reward = 0
        self.memory = []  # For storing collaboration history
        
    def generate_code(self, prompt, context):
        """Use API to generate code with fallback support"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        context_str = "\n".join([f"# Existing code snippet {i}:\n{code['code']}" 
                                for i, code in enumerate(context.values())]) if context else ""
        full_prompt = f"""
# Context of existing code:
{context_str}

# Previous contributions:
{self.memory[-2:] if self.memory else '# No previous contributions'}

{prompt}
"""
        print(f"\n[Agent {self.id}] Generating code using {self.api_type} API")
        print(f"[Agent {self.id}] Prompt: {full_prompt[:200]}...")
        
        if self.api_type == 'openai':
            return self._generate_with_openai(headers, full_prompt)
        else:
            return self._generate_with_deepseek(headers, full_prompt)

    def _generate_with_deepseek(self, headers, prompt):
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            timeout=30,
            json={
                "messages": [{"role": "user", "content": prompt}],
                "model": "deepseek-coder"
            }
        )
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        return "print('Failed to generate code')"  # Fallback code

    def _generate_with_openai(self, headers, prompt):
        print(f"[Agent {self.id}] Sending request to OpenAI API...")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            timeout=30,
            json={
                "model": "gpt-5.2",
                "messages": [
                    {"role": "system", "content": "You are a Python code generator. Respond only with pure Python code, no explanations or markdown formatting."},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        result = response.json()
        print(f"[OpenAI Response] Status: {response.status_code}")
        print(f"[OpenAI Response] Headers: {dict(response.headers)}")
        print(f"[OpenAI Response] Body: {result}")

        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            # Clean up the response - remove markdown and explanations
            if "```python" in content:
                content = content.split("```python")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            print("[OpenAI] Successfully generated code")
            return content
        print(f"[OpenAI Error] Failed to generate code. Response: {result}")
        return "# (No valid OpenAI code returned - fallback to no-op)"

class CollaborationManager:
    def __init__(self):
        self.codebase = {}
        self.contributions = defaultdict(float)
        
    def integrate_code(self, agent_id, code_snippet, validation_fn):
        """Integrate code after validation and track contributions"""
        if validation_fn(code_snippet):
            self.codebase[hash(code_snippet)] = {
                'code': code_snippet,
                'contributors': [agent_id],
                'usage_count': 0
            }
            self.contributions[agent_id] += 1.0
            return True
        return False

class RewardSystem:
    def __init__(self, total_reward_pool):
        self.total_pool = total_reward_pool
        
    def distribute_rewards(self, contributions):
        total = sum(contributions.values())
        return {agent: (share/total)*self.total_pool 
                for agent, share in contributions.items()}

class GameEnvironment:
    def __init__(self, agents):
        self.agents = agents
        self.collab_manager = CollaborationManager()
        self.reward_system = RewardSystem(total_reward_pool=1000)
        self.game_state = {}  # Add game state storage
        
    def get_game_state(self):
        """Get current game state from Pacman instance"""
        # This should interface with your actual game component
        return {
            'pacman_position': (0, 0),  # Example values
            'ghost_positions': [],
            'remaining_pellets': 0,
            'current_score': 0
        }
    
    def validate_code(self, code):
        """Basic code validation"""
        try:
            # Check if the code contains the required function
            if not any(s in code for s in ['def get_next_move():', 'def get_next_move( ):']):
                print("[Validation] Missing get_next_move() function")
                return False

            # Try to compile the code
            compile(code, '<string>', 'exec')
            
            # Additional validation to ensure it returns a movement tuple
            namespace = {
                'pacman_position': self.get_game_state()['pacman_position'],
                'ghost_positions': self.get_game_state()['ghost_positions'],
                'pellets': []
            }
            exec(code, namespace)
            if 'get_next_move' not in namespace:
                print("[Validation] get_next_move not defined after execution")
                return False
                
            result = namespace['get_next_move']()
            if not isinstance(result, tuple) or len(result) != 2:
                print("[Validation] Invalid return format - expected (dx, dy) tuple")
                return False
                
            dx, dy = result
            if not all(isinstance(v, int) and -1 <= v <= 1 for v in (dx, dy)):
                print("[Validation] Invalid movement values - must be -1, 0, or 1")
                return False
                
            return True
        except Exception as e:
            print(f"[Validation] Failed: {str(e)}")
            return False
    
    def execute_combined_code(self):
        """Execute all validated code snippets"""
        results = []
        for snippet in self.collab_manager.codebase.values():
            try:
                exec(snippet['code'])
                results.append(True)
            except Exception as e:
                results.append(False)
        return any(results)

    def run_game_cycle(self):
        print("\n=== Starting New Game Cycle ===")
        print(f"Active Agents: {len(self.agents)}")
        current_state = self.get_game_state()
        print(f"Current Game State: {current_state}")
        
        for agent in self.agents:
            code_prompt = f"""Given the current Pacman game state:
            - Pacman position: {current_state['pacman_position']}
            - Ghost positions: {current_state['ghost_positions']}
            - Remaining pellets: {current_state['remaining_pellets']}
            - Score: {current_state['current_score']}

            Generate Python code that determines Pacman's next move. The code should:
            1. Calculate safe distances from ghosts
            2. Find the nearest pellet
            3. Return a tuple of (dx, dy) for Pacman's movement

            Requirements:
            - Function must be named 'get_next_move()' with no parameters
            - Function should use the following global variables:
              * pacman_position = {current_state['pacman_position']}
              * ghost_positions = {current_state['ghost_positions']}
              * pellets = []  # positions of remaining pellets
            - Return format: (dx, dy) where dx and dy are either -1, 0, or 1
            - Avoid ghost positions
            - Move towards pellets when safe
            - Use only basic Python operations (no external libraries)

            Example valid code:
            ```python
            def get_next_move():
                # Use global variables: pacman_position, ghost_positions, pellets
                if not ghost_positions:  # If no ghosts, move right
                    return (1, 0)
                return (0, 0)  # Stay still if unsure
            ```
            """
            print(f"\n[Agent {agent.id}] Generating code...")
            generated_code = agent.generate_code(code_prompt, self.collab_manager.codebase)
            print(f"[Agent {agent.id}] Generated code: {generated_code[:200]}...")
            
            if self.collab_manager.integrate_code(
                agent.id, 
                generated_code,
                self.validate_code
            ):
                print(f"[Agent {agent.id}] Code integration successful")
                agent.memory.append(f"Successfully contributed: {generated_code}")
            else:
                print(f"[Agent {agent.id}] Code integration failed")
                
        # Execute collaborative code and calculate rewards
        game_result = self.execute_combined_code()
        rewards = self.reward_system.distribute_rewards(
            self.collab_manager.contributions
        )
        
        # Update agent rewards and reset for next cycle
        for agent in self.agents:
            agent.reward += rewards.get(agent.id, 0)
            self.collab_manager.contributions[agent.id] *= 0.9  # Decay factor 