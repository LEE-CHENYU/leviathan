from agent_system.execution import execute_code_safely

class PacmanGame:
    def __init__(self):
        self.agents = []
        self.score = 0
        self.ghosts = []
        self.pellets = []
        self.pacman_position = (0, 0)  # Initial position
        
    def get_state(self):
        return {
            'pacman_position': self.pacman_position,
            'ghost_positions': [(x1, y1), ...],
            'remaining_pellets': count,
            'current_score': self.score
        }
        
    def apply_movement(self, codebase):
        """Execute combined code to determine movement"""
        print("\n=== Applying Movement ===")
        print(f"Current position: {self.pacman_position}")
        print(f"Number of code snippets to execute: {len(codebase)}")
        
        for snippet in codebase.values():
            print(f"\nExecuting snippet: {snippet['code'][:200]}...")
            if execute_code_safely(snippet['code']):
                # Example implementation - modify based on your actual game logic
                self.pacman_position = (
                    self.pacman_position[0] + 1,
                    self.pacman_position[1] + 1
                )
                self.score += 10
                print(f"Movement successful. New position: {self.pacman_position}")
                print(f"New score: {self.score}")
            else:
                print("Movement execution failed") 