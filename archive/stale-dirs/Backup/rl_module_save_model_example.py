import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn

game = pyspiel.load_game("tic_tac_toe")
env = rl_environment.Environment(game)

# Initialize agent
agent = dqn.DQN(env.observation_spec(), env.action_spec(), hidden_layers_sizes=[128, 128])

# Train agent
for _ in range(1000):
    time_step = env.reset()
    while not time_step.last():
        agent_output = agent.step(time_step)
        time_step = env.step([agent_output.action])
    agent.step(time_step)

# Save the trained model
agent._q_network.save('dqn_tic_tac_toe_model')