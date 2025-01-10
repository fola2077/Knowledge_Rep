# evaluate.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from policy import DroneEnv
from dqn_agent import DQNAgent

def load_agents(checkpoint_path, num_drones, env):
    agents = []
    checkpoint = torch.load(checkpoint_path)
    for i in range(num_drones):
        agent = DQNAgent(
            drone_id=i,
            state_dim=env.observation_space,
            action_dim=env.action_space,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            buffer_capacity=50000,
            batch_size=128,
            epsilon_start=0.0,  # Set epsilon to 0 for evaluation
            epsilon_end=0.0,
            epsilon_decay=1
        )
        agent.policy_net.load_state_dict(checkpoint[f'drone_{i}_policy_net'])
        agent.policy_net.eval()
        agents.append(agent)
    return agents

def evaluate_agents(agents, env, num_episodes=10):
    num_drones = len(agents)
    env.num_drones = num_drones
    rewards_per_episode = []
    all_positions = [[] for _ in range(num_drones)]
    action_counts = np.zeros((num_drones, env.action_space), dtype=int)
    detection_counts = np.zeros(num_drones, dtype=int)
    detection_delays = []
    weather_counts = {}
    oil_detection_grid = np.zeros((env.environment.grid_width, env.environment.grid_height))
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = [0 for _ in range(num_drones)]
        done = False
        env.oil_detection_grid = np.zeros((env.environment.grid_width, env.environment.grid_height))
        
        while not done:
            state_tensor = [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in state]
            actions = []
            for i, agent in enumerate(agents):
                with torch.no_grad():
                    q_values = agent.policy_net(state_tensor[i])
                    action = q_values.argmax(dim=1).item()
                actions.append(action)
                action_counts[i, action] += 1

            next_state, rewards, done, infos = env.step(actions)

            # Collect rewards
            for i in range(num_drones):
                total_reward[i] += rewards[i]

                # Collect positions
                position = env.drones[i].position
                all_positions[i].append([position.x, position.y])

                # Update detection counts and delays
                if infos[i].get('detected', False):
                    detection_counts[i] += 1
                    if infos[i].get('detection_delay') is not None:
                        detection_delays.append(infos[i]['detection_delay'])

            # Collect weather frequencies
            current_weather = env.weather_system.current_state.name
            weather_counts[current_weather] = weather_counts.get(current_weather, 0) + 1

            state = next_state

        # Update oil detection grid
        oil_detection_grid += env.oil_detection_grid
        rewards_per_episode.append(sum(total_reward))
        print(f'Episode {episode + 1} completed.')
    
    return {
        'rewards_per_episode': rewards_per_episode,
        'all_positions': all_positions,
        'action_counts': action_counts,
        'detection_counts': detection_counts,
        'detection_delays': detection_delays,
        'weather_counts': weather_counts,
        'oil_detection_grid': oil_detection_grid
    }

def plot_results(data, num_drones):
    # Rewards per Episode
    plt.figure()
    plt.plot(data['rewards_per_episode'])
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    
    # Oil Spill Detection Frequency Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data['oil_detection_grid'].T, cmap='inferno')
    plt.title('Oil Spill Detection Frequency Heatmap')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.show()
    
    # Action Frequency
    actions = ['Up', 'Down', 'Left', 'Right', 'Scan']
    for i in range(num_drones):
        plt.figure()
        plt.bar(actions, data['action_counts'][i])
        plt.title(f'Action Frequency for Drone {i+1}')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.show()
    
    # Weather Frequency
    plt.figure()
    plt.bar(data['weather_counts'].keys(), data['weather_counts'].values())
    plt.title('Weather Condition Frequency During Evaluation')
    plt.xlabel('Weather Condition')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
    
    # Drone Detection Frequency
    plt.figure()
    plt.bar(range(1, num_drones + 1), data['detection_counts'])
    plt.title('Oil Spill Detections per Drone')
    plt.xlabel('Drone ID')
    plt.ylabel('Number of Detections')
    plt.xticks(range(1, num_drones + 1))
    plt.show()
    
    # Time Detection Frequency (Detection Delays)
    if len(data['detection_delays']) > 0:
        plt.figure()
        plt.hist(data['detection_delays'], bins=20, edgecolor='black')
        plt.title('Distribution of Detection Delays')
        plt.xlabel('Detection Delay (Minutes)')
        plt.ylabel('Frequency')
        plt.show()
    else:
        print("No detection delays to plot.")
    
    # Drone Trajectories
    for i in range(num_drones):
        positions = np.array(data['all_positions'][i])
        plt.figure(figsize=(8, 6))
        plt.plot(positions[:, 0], positions[:, 1], label=f'Drone {i+1} Trajectory')
        plt.title(f'Drone {i+1} Movement Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.show()

def main():
    checkpoint_path = 'checkpoint3.pth.tar'
    num_drones = 3  # Set the number of drones as per your training
    num_episodes = 100  # Number of evaluation episodes
    
    env = DroneEnv(num_drones=num_drones)
    agents = load_agents(checkpoint_path, num_drones, env)
    data = evaluate_agents(agents, env, num_episodes)
    plot_results(data, num_drones)

if __name__ == "__main__":
    main()