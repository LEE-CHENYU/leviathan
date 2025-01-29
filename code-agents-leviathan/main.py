import os
from dotenv import load_dotenv
import requests
from agent_system.core import Agent, GameEnvironment
from game_components.pacman import PacmanGame
import traceback
from game_components.physical_simulation_game import PhysicalSimulationGame

def initialize_agents():
    """Create agent pool with fallback API support"""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    print(f"[DEBUG] Using OpenAI key: {openai_key[:10]}...{openai_key[-10:]}")  # Print first/last 10 chars for verification
    agents = []
    
    for i in range(int(os.getenv("AGENT_POOL_SIZE", 5))):
        agents.append(Agent(
            agent_id=f"agent_{i}",
            api_key=openai_key,
            api_type="openai"
        ))
    return agents

def run_simulation(cycles=10):
    """Main execution loop with error handling"""
    print("\n=== Starting Simulation ===")
    print(f"Initializing with {cycles} cycles")
    
    agents = initialize_agents()
    print(f"Created {len(agents)} agents")
    for agent in agents:
        print(f"Agent {agent.id}: Using {agent.api_type} API")
    
    game_env = GameEnvironment(agents)
    pacman = PacmanGame()
    
    for cycle in range(cycles):
        print(f"\n{'='*20} Game Cycle {cycle + 1} {'='*20}")
        
        try:
            game_env.run_game_cycle()
            pacman.apply_movement(game_env.collab_manager.codebase)
            
            print("\n=== Cycle Results ===")
            print(f"Total contributions: {len(game_env.collab_manager.codebase)}")
            print("\nAgent Rewards:")
            for agent in agents:
                print(f"{agent.id}: {agent.reward:.2f} points")
                
        except Exception as e:
            print("\n!!! Critical Error !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            break

if __name__ == "__main__":
    import argparse
    # Verify .env is loaded
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment variables!")
        exit(1)

    parser = argparse.ArgumentParser(description="Run collaborative AI game simulation")
    parser.add_argument("--cycles", type=int, default=10, help="Number of game cycles")
    parser.add_argument("--agents", type=int, default=5, help="Number of AI agents")
    args = parser.parse_args()
    
    os.environ["AGENT_POOL_SIZE"] = str(args.agents)
    
    # Let user choose game mode
    choice = input("Enter 'pacman' to run Pacman game or 'physical_sim' to run physical simulation: ")
    
    if choice == "pacman":
        run_simulation(args.cycles)
    elif choice == "physical_sim":
        agents = initialize_agents()
        print(f"\nStarting Physical Simulation with {len(agents)} agents")
        game = PhysicalSimulationGame(agents)
        game.run_simulation_cycle()
        
        # Print final results
        print("\n=== Final Results ===")
        for agent in agents:
            print(f"Agent {agent.id}: {agent.reward:.2f} points")
    else:
        print("Unknown game mode, exiting.") 