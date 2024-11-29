import gym
from gym import spaces
import numpy as np

class DroneEnvironment(gym.Env):
    def __init__(self, num_drones=3):
        super(DroneEnvironment, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # 0: Idle, 1: Up, 2: Down, 3: Left, 4: Right
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(num_drones, 4), dtype=np.float32
        )  # Each drone: [x, y, fuel, payload]

        self.num_drones = num_drones
        self.state = None
        self.reset()

    def reset(self):
        # Initialize drones and oil spills
        self.state = np.random.rand(self.num_drones, 4) * 100  # Random positions
        return self.state

    def step(self, actions):
        rewards = []
        done = False
        for i, action in enumerate(actions):
            # Apply actions (simplistic update for example)
            if action == 1: self.state[i][1] += 1  # Move up
            elif action == 2: self.state[i][1] -= 1  # Move down
            elif action == 3: self.state[i][0] -= 1  # Move left
            elif action == 4: self.state[i][0] += 1  # Move right
            
            # Simplified reward: distance to a target oil spill
            rewards.append(-np.linalg.norm(self.state[i][:2] - [50, 50]))
        
        return np.array(self.state), rewards, done, {}

    def render(self, mode="human"):
        # Render drones' positions
        print(f"Drones' states: {self.state}")
