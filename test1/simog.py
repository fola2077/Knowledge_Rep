import pybullet as p
import pybullet_data
import numpy as np
import time
from test_qlearn import DroneAgent

class DroneEnvironment:
    def __init__(self, num_drones=3):
        self.num_drones = num_drones
        self.drones = []
        self.oil_spills = []
        self.client = p.connect(p.GUI)  # Start PyBullet GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load URDF assets
        p.setGravity(0, 0, -9.8)  # Set gravity for the environment
        self.ground = p.loadURDF("plane.urdf")  # Load a flat ground plane
        self.initialize_oil_spills()
        self.initialize_drones()

    def initialize_drones(self):
        """Initialize drones as small cubes at random positions."""
        for _ in range(self.num_drones):
            x, y = np.random.uniform(0, 5), np.random.uniform(0, 5)
            drone = p.loadURDF("cube_small.urdf", [x, y, 1])  # Place 1m above ground
            self.drones.append(drone)

    def initialize_oil_spills(self):
        """Initialize oil spills as red spheres at random positions."""
        for _ in range(5):  # Assume 5 oil spills
            x, y = np.random.uniform(0, 5), np.random.uniform(0, 5)
            oil_spill = p.loadURDF("sphere2.urdf", [x, y, 0.1], globalScaling=0.5)  # Small sphere
            self.oil_spills.append(oil_spill)

    def reset(self):
        """Reset drones to random positions."""
        for drone in self.drones:
            x, y = np.random.uniform(0, 5), np.random.uniform(0, 5)
            p.resetBasePositionAndOrientation(drone, [x, y, 1], [0, 0, 0, 1])

    def step(self, actions):
        """Update drone positions based on actions."""
        rewards = []
        for i, action in enumerate(actions):
            pos, _ = p.getBasePositionAndOrientation(self.drones[i])
            x, y, z = pos

            # Move based on action
            if action == 1: y += 0.1  # Move Up
            elif action == 2: y -= 0.1  # Move Down
            elif action == 3: x -= 0.1  # Move Left
            elif action == 4: x += 0.1  # Move Right

            # Update drone position
            p.resetBasePositionAndOrientation(self.drones[i], [x, y, z], [0, 0, 0, 1])

            # Calculate reward (distance to average oil spill position)
            oil_spill_positions = np.array([p.getBasePositionAndOrientation(spill)[0][:2] for spill in self.oil_spills])
            avg_spill_position = oil_spill_positions.mean(axis=0)
            rewards.append(-np.linalg.norm(np.array([x, y]) - avg_spill_position))

        return self.get_state(), rewards, False, {}

    def get_state(self):
        """Return current positions of all drones."""
        return [p.getBasePositionAndOrientation(drone)[0][:2] for drone in self.drones]

    def render(self):
        """Rendering is handled by PyBullet's GUI; no need for additional code here."""
        pass

    def close(self):
        """Disconnect PyBullet."""
        p.disconnect()

# Initialize environment and agent
env = DroneEnvironment(num_drones=3)
agent = DroneAgent(state_dim=2, action_dim=5)  # State dim = 2 (x, y), 5 actions

num_episodes = 500
rewards_history = []

try:
    for episode in range(num_episodes):
        env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            state = env.get_state()
            actions = [agent.select_action(s) for s in state]  # Actions for each drone
            next_state, rewards, done, _ = env.step(actions)

            # Update agent
            for i, reward in enumerate(rewards):
                agent.store_transition(state[i], actions[i], reward, next_state[i], done)
            agent.train_step()

            total_reward += sum(rewards)
            step += 1

            # Step simulation for visualization
            p.stepSimulation()
            time.sleep(0.03)  # Small delay for real-time visualization

        rewards_history.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

finally:
    # Ensure resources are released
    env.close()

