import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        
        # Define state space
        self.state_space = {
            "drone_position": (0, 0, 0),  # x, y, z coordinates
            "battery_level": 100,         # percentage
            "weather_conditions": {"visibility": 1.0, "wind_speed": 0.5},
            "spill_probabilities": np.zeros((100, 100)),  # 100x100 grid
        }

        # Observation space: normalized values for states
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )

        # Action space: discrete actions (e.g., move, adjust sensors)
        self.action_space = spaces.Discrete(5)

        # Internal variables
        self.state = None
        self.reward_history = []

    def reset(self):
        """Reset environment to the initial state."""
        self.state = {
            "drone_position": (0, 0, 0),
            "battery_level": 100,
            "weather_conditions": {"visibility": 1.0, "wind_speed": 0.5},
            "spill_probabilities": np.zeros((100, 100)),
        }
        return np.array([
            *self.state["drone_position"],
            self.state["battery_level"] / 100,
            self.state["weather_conditions"]["visibility"],
        ])

    def step(self, action):
        """Apply an action to the environment."""
        self.state = self.transition_model(self.state, action)
        reward = self.reward_function(self.state, action)
        self.reward_history.append(reward)
        done = self.state["battery_level"] <= 0  # Episode ends when battery is depleted
        return (
            np.array([
                *self.state["drone_position"],
                self.state["battery_level"] / 100,
                self.state["weather_conditions"]["visibility"],
            ]),
            reward,
            done,
            {},
        )

    def transition_model(self, state, action):
        """Simulate the state transition based on action."""
        if action == 0:  # Move forward
            state["drone_position"] = (
                state["drone_position"][0] + 1,
                state["drone_position"][1],
                state["drone_position"][2],
            )
        elif action == 1:  # Increase altitude
            state["drone_position"] = (
                state["drone_position"][0],
                state["drone_position"][1],
                state["drone_position"][2] + 1,
            )
        elif action == 2:  # Decrease altitude
            state["drone_position"] = (
                state["drone_position"][0],
                state["drone_position"][1],
                max(0, state["drone_position"][2] - 1),
            )
        # Simulate battery usage
        state["battery_level"] -= 5

        # Simulate environmental changes
        state["weather_conditions"]["visibility"] = max(
            0.1, state["weather_conditions"]["visibility"] - 0.01
        )

        return state

    def reward_function(self, state, action):
        """Calculate rewards based on state and action."""
        if state["weather_conditions"]["visibility"] < 0.3:
            return -10  # Penalty for poor visibility
        if state["battery_level"] < 20:
            return -50  # Heavy penalty for low battery
        if action in [0, 1]:  # Useful actions
            return 10
        return -1  # Small penalty for other actions

# Simulate environment without RL
env = DroneEnv()
state = env.reset()
steps = 50

for _ in range(steps):
    action = np.random.choice(env.action_space.n)  # Random action
    state, reward, done, _ = env.step(action)
    print(f"Step: {_}, Action: {action}, State: {state}, Reward: {reward}")
    if done:
        print("Episode finished due to battery depletion.")
        break


# Visualization
plt.plot(range(len(env.reward_history)), env.reward_history)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Reward History")
plt.show()

# Save for analysis
np.save("reward_history_no_rl.npy", env.reward_history)