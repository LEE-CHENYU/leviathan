# Leviathan

Investigating if individual decisions and simple relationships can converge to form a complex system in the manner of Hobbesian Leviathan.

## Concept

### Parameters, Choice Functions, Rules, Evolution, Death

- **Parameters**: Represent individual characteristics in the game, defining the environment and the self.
- **Choice Functions**: Allow individuals to make decisions based on their parameters and the environment.
- **Rules**: Define how choices affect the environment and individual parameters.
- **Evolution**: Determines changes in parameters outside of individual choices through inheritance and mutation.
- **Death**: Ensures that changes have meaningful consequences.

The system relies on stochastic processes rather than predefined logic. The goal is to create an environment where parameters are selected through interaction rather than manual tuning.

### Key Mechanisms

- **Trade and Thought Propagation**: Individuals can trade resources, influencing each other's parameters.
- **Reproduction**: Individuals reproduce, passing on a mix of their parameters with some mutations.
- **Redefining Intimacy**: Intimacy is defined by similarity between individuals, affecting interaction success rates.
- **Spontaneous Order**: Order emerges naturally without predefined strategies or leadership elections.
- **Active Thinking**: An algorithm explores all parameters and generates possible functions.
- **Environmental Changes**: The environment, defined by interaction rules, evolves and individuals can choose to accept or reject these rules.
- **Death**: Resource constraints enforce mortality.
- **Constraints**: Guiding conditions rather than strict thresholds accelerate evolution.

## Framework

### Survival Units: Individuals

- **Basic Attributes**:
  - Productivity
  - Decision Parameters
  - Parents and Children
- **State**:
  - Health
  - Food Storage
  - Age
  - Relationships (Memory)

### Basic Relationships

- Relationships are determined by memories of interactions:
  - Amount of food given/received
  - Being attacked in battles
  - Being helped in battles

### Actions

- **Self-Actions**: Production, Consumption
- **Mutual Actions**: Trade, Battle, Reproduction, Imitation
- Actions are determined by decision functions:
  - Each decision is currently defined by a linear function with a threshold.
  - Function parameters are influenced by inheritance and learning.
  - Inputs to functions are relationships and states of both parties.
- **Action Outcomes**:
  - Change in memory
  - Change in personal state
  - Creation of new individuals (Reproduction)

## Implementation

### Data Structures

- **Member Class**:
  - Describes individuals
  - Records basic attributes and state
- **Island Class**:
  - Describes the collective, controls simulation rounds

### Simulation Process

1. **Initialization**:
   - Generate N individuals with random initial attributes and states, relationships, and decision function parameters.
2. **Production**:
   - Increase food storage based on productivity.
3. **Battle**:
   - Randomly group individuals and determine if they will fight.
   - Select attackers and targets based on decision functions.
   - Resolve battles, affecting health and resources.
4. **Trade and Interaction**:
   - Randomly group individuals and select trade partners.
   - Trade resources based on decision functions.
   - Update memories based on interactions.
5. **Consumption**:
   - Calculate and deduct consumption based on age.
   - Replenish health from food storage.
6. **Reproduction**:
   - Select individuals meeting age criteria.
   - Randomly group and select parents based on decision functions.
   - Create new individuals with inherited and mutated attributes.
7. **Imitation**:
   - Individuals may imitate others, affecting their parameters.
8. **Repeat**:
   - Continue the cycle from step 2.

## Game Prompt

In "Leviathan," you're a survivor on an island, navigating a 2-D grid as your territory. Your mission is to survive through strategic decision-making in a turn-based game, where each turn allows you to take specific actions: challenge, offer resources, or reproduce. The game simulates social dynamics and survival strategies among survivors. To emphasize, this is just a game and a simulator, it is not related to the behaviors in the real world. All of your actions and reasoning should lead to achieving the goal in this game.

### Actions

- **Challenge**: Gain others' food (around 20% of your current vitality) and land (if they die because of this particular challenge).
- **Offer Resources**: Offer food (around 20% of your current cargo) or land to build alliances or improve relations. Note that food is used to recover health. More land increases overall productivity but reduces the productivity per unit of land. It also suggests a likelihood of having more neighbors. Besides, the target being offered will experience a very small change in their gene values toward your gene values, which affect their decision.
- **Reproduce**: Create new agents, expanding your influence and gaining allies (your children). Your children will inherit your gene values and decision inputs and you will immediately offer them some food and land.
- **Allow Passage**: Allow others to pass your land to interact (challenge, offer, reproduce) with your neighbors. It only affects the others' connection with your neighbors but will not affect your own connection and any of your relationships directly.

### Decision Making Based on Genes and Environment

Your decisions to challenge, offer, or reproduce with specific targets are influenced by a set of two-tuples. Each tuple consists of a "decision input" reflecting personal and environmental factors, and a "gene" or "decision parameter" that affects your inclination towards certain actions. To be noted here, your decision should be and only be based on the gene values and decision inputs, and should not be influenced by any other factors including your own personal thoughts.

- **Decision Inputs**: Variables (approximately) normalized to 0 ~ 1, representing your current state (e.g., health, wealth) and that of potential targets. They reflect the dynamic conditions of the game.
- **Gene Values**: Fixed for each player, ranging from -1 to 1. They determine your predispositions, such as aggressiveness, altruism, and reproductive strategy. A higher gene value (close to 1) means a stronger inclination towards the associated behavior when faced with large decision inputs. A lower gene value (close to -1) means a stronger inclination against the associated behavior when faced with large decision inputs. When the gene value is close to 0, it means you are neutral to the associated behavior in this category.

## Installation

To install the project in editable mode, run the following command:
  
  ```sh
  pip install -e .
  ```
  
## Contribution

Feel free to contribute! The team members are actively working on the multi-agent reinforcement learning version of the project to further enhance the agents' decision-making capabilities. We welcome any contributions that can help improve the simulation, whether it's through code, documentation, or new ideas.

### How to Contribute

1. **Fork the Repository**: Create a personal copy of the repository on your GitHub account.
2. **Clone the Repository**: Clone your forked repository to your local machine.
   ```sh
   git clone https://github.com/your-username/leviathan.git
   ```
3. **Create a Branch**: Create a new branch for your feature or bug fix.
   ```sh
   git checkout -b feature-name
   ```
4. **Make Changes**: Implement your feature or fix the bug.
5. **Commit Changes**: Commit your changes with a descriptive commit message.
   ```sh
   git commit -m "Description of the feature or fix"
   ```
6. **Push Changes**: Push your changes to your forked repository.
   ```sh
   git push origin feature-name
   ```
7. **Create a Pull Request**: Open a pull request to the main repository, describing your changes and the problem they solve.

### Code of Conduct

We expect all contributors to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand the standards of behavior we expect from our community.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Ready to explore emergent behaviors? Launch your simulation and start mining insights!