# README: WEATHER-ADAPTIVE U.A.V SIMULATION FOR OIL SPILL DETECTION THROUGH DEEP Q LEARNING

## Project Overview
This project simulates a multi-drone system for detecting and managing oil spills using reinforcement learning. The simulation integrates dynamic environmental conditions, weather systems, and collaborative drone actions to optimize spill detection. A Deep Q-Learning (DQN) approach is implemented to train drones for efficient exploration and detection, guided by a Sequential Decision Framework.

---

## Code Files and Structure

### `analysis.py`
This module handles data analysis and visualization of simulation results. Key features include:
- **Reward Analysis**: Plots cumulative rewards over episodes for evaluating DQN performance.
- **Drone Trajectories**: Visualizes movement paths for all drones during simulation.
- **Detection Metrics**: Generates heatmaps and action distributions to assess detection efficiency and strategy.

### `config.py`
Defines configuration parameters for the simulation, such as:
- **Grid Size**: Defines the spatial dimensions of the simulation environment.
- **Drone Count**: Number of drones deployed in the simulation.
- **Weather Parameters**: Probability distributions for weather conditions like rain, fog, and visibility.
- **Reward Function Weights**: Parameters tuning the reward system to balance detection accuracy and energy efficiency.

### `dqn_agent.py`
Implements the Deep Q-Learning agent, responsible for learning optimal policies. Key components include:
- **Neural Network Architecture**: Maps states to Q-values for each action.
- **Epsilon-Greedy Strategy**: Balances exploration and exploitation during training.
- **Replay Buffer**: Stores experiences for mini-batch training to stabilize learning.

### `drone.py`
Defines the `Drone` class, encapsulating drone-specific attributes and behaviors:
- **Sensors**: Infrared, LiDAR, and environmental sensors for detecting oil spills.
- **Battery Management**: Tracks battery usage and triggers recharge behavior.
- **Movement Logic**: Executes actions based on policy decisions.

### `environment.py`
Models the simulation environment, including:
- **Grid Representation**: A 1200x800 grid representing land and water regions.
- **Dynamic Weather**: Updates weather conditions influencing visibility and spill spread.
- **Oil Spill Generation**: Randomly initializes spills and simulates their spread based on wind and currents.

### `oilspillage.py`
Handles the simulation of oil spill dynamics:
- **Spill Initialization**: Randomly assigns starting positions for spills.
- **Spread Modeling**: Uses probabilistic models influenced by wind and water currents.

### `policy.py`
This file defines the **policy function** in the Sequential Decision Framework, incorporating all five modeling stages (state, decision, exogenous information, transition function, and objective function):

1. **State Variables (`S_t`)**:
   - Includes drone positions, battery levels, oil spill locations, and environmental conditions, dynamically updated per simulation step.

2. **Decision Variables (`X_t`)**:
   - Represents drone actions such as movement (up, down, left, right, stay) or sensor activation.
   - Actions are selected using the epsilon-greedy strategy based on Q-values.

3. **Exogenous Information (`W_t`)**:
   - Captures external factors like weather changes, visibility, and random oil spill events.
   - These stochastic elements influence state transitions.

4. **Transition Function**:
   - Defines state evolution based on actions and environmental dynamics, producing the next state (`S_{t+1}`).

5. **Objective Function**:
   - Optimizes the cumulative reward by maximizing the Q-value through the Bellman equation.
   - Rewards encourage efficient oil spill detection and penalize unnecessary actions or high energy use.

This module integrates seamlessly with `train_dqn.py`, ensuring policy-driven learning during Q-learning. It provides the abstraction needed to test and refine alternative decision-making strategies while remaining mathematically aligned with the theoretical framework.

### `train_dqn.py`
Trains the DQN agent using the environment and policy:
- **Episode Management**: Handles multiple episodes for iterative learning.
- **Reward Calculation**: Updates Q-values based on rewards from actions.
- **Model Saving**: Exports the trained DQN model for deployment.

### `weather.py`
Simulates weather conditions impacting drone operations:
- **Condition Generation**: Randomly determines rain, fog, and storm probabilities.
- **Visibility Impact**: Adjusts drone sensor range based on weather.
- **Wind Influence**: Modifies oil spill spread dynamics.

---

## How to Run the Project

1. **Install Dependencies**: Ensure Python 3.x and required libraries (e.g., NumPy, Matplotlib, TensorFlow) are installed.
2. **Configure Settings**: Modify parameters in `config.py` as needed.
3. **Run Simulation**: Execute `train_dqn.py` to start training the drones.
4. **Analyze Results**: Use `analysis.py` to visualize outcomes.

---

## Key Features
- **Reinforcement Learning**: Utilizes DQN for decision-making under uncertainty.
- **Dynamic Environment**: Simulates realistic weather and oil spill scenarios.
- **Comprehensive Analysis**: Provides detailed metrics and visualizations for evaluation.

---

### Training Script (train_dqn.py)
The `train_dqn.py` script orchestrates the training process for the DQN model. Key features include:
- Dynamic epsilon-greedy exploration.
- Logging of rewards and policy evaluations.
- Saving and loading of trained models for reproducibility.

### Environment Module (environment.py)
The `environment.py` file defines the simulation environment, including:
- Grid-based representation of the geographical area.
- Initialization of oil spills and their probabilistic spread.
- Interaction dynamics between drones and the environment.

### Analysis Tools (analysis.py)
This module provides tools for:
- Generating heatmaps of drone positions.
- Visualizing trajectories and action distributions.
- Evaluating detection delays and overall performance.

## Contact

For questions, feedback, or collaboration opportunities, please contact the team:

- **Member 1**: Adegorite Michael Adefola — 24801131@stu.mmu.ac.uk
- **Member 2**: Eluwande Olamiposi Samuel — 24776309@stu.mmu.ac.uk
- **Member 3**: Ashamu Isaac Inioluwa — email@example.com

