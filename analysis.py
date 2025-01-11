# ANALYSIS.PY

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
    rewards_per_episode_per_drone = [[] for _ in range(num_drones)]
    detection_delays_per_drone = [[] for _ in range(num_drones)]
    # For drone coverage map
    drone_coverage_grid = np.zeros((env.environment.grid_width, env.environment.grid_height))

    # **Initialize oil concentration sum grid**
    oil_concentration_sum_grid = np.zeros((env.environment.grid_width, env.environment.grid_height))
    num_time_steps = 0
    oil_spill_origins = set()

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = [0 for _ in range(num_drones)]
        done = False
        env.oil_detection_grid = np.zeros((env.environment.grid_width, env.environment.grid_height))
        # Reset per-episode data if needed

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
                rewards_per_episode_per_drone[i].append(rewards[i])

                # Collect positions
                position = env.drones[i].position
                all_positions[i].append([position.x, position.y])

                # Convert positions to grid indices
                x = int(round(position.x))
                y = int(round(position.y))

                # Ensure indices are within grid bounds
                x = max(0, min(x, env.environment.grid_width - 1))
                y = max(0, min(y, env.environment.grid_height - 1))

                # Update drone coverage grid
                drone_coverage_grid[x, y] += 1

                # Update detection counts and delays
                if infos[i].get('detected', False):
                    detection_counts[i] += 1
                    if infos[i].get('detection_delay') is not None:
                        detection_delays.append(infos[i]['detection_delay'])
                        detection_delays_per_drone[i].append(infos[i]['detection_delay'])

            # Collect weather frequencies
            current_weather = env.weather_system.current_state.name
            weather_counts[current_weather] = weather_counts.get(current_weather, 0) + 1

            # Collect oil concentration grid and sum up
            oil_concentration_grid = env.oil_spillage.get_oil_concentration_grid()
            oil_concentration_sum_grid += oil_concentration_grid
            num_time_steps += 1

            # Collect oil spill origins
            spill_origins = env.oil_spillage.get_current_oil_spill_origins()
            oil_spill_origins.update(spill_origins)

            state = next_state

        # Update oil detection grid
        oil_detection_grid += env.oil_detection_grid
        # Debug statements to check oil_detection_grid values
        print(f"After episode {episode + 1}, oil_detection_grid stats: max={np.max(env.oil_detection_grid)}, min={np.min(env.oil_detection_grid)}, mean={np.mean(env.oil_detection_grid)}, std={np.std(env.oil_detection_grid)}")
        rewards_per_episode.append(sum(total_reward))
        print(f'Episode {episode + 1} completed.')

    # Calculate average oil concentration grid
    average_oil_concentration_grid = oil_concentration_sum_grid / num_time_steps

    return {
        'rewards_per_episode': rewards_per_episode,
        'all_positions': all_positions,
        'action_counts': action_counts,
        'detection_counts': detection_counts,
        'detection_delays': detection_delays,
        'weather_counts': weather_counts,
        'oil_detection_grid': oil_detection_grid,
        'rewards_per_episode_per_drone': rewards_per_episode_per_drone,
        'detection_delays_per_drone': detection_delays_per_drone,
        'drone_coverage_grid': drone_coverage_grid,
        'average_oil_concentration_grid': average_oil_concentration_grid,
        'oil_spill_origins': list(oil_spill_origins)
    }

def plot_results(data, num_drones):
    import matplotlib.ticker as ticker
    # Rewards per Episode
    plt.figure()
    plt.plot(data['rewards_per_episode'])
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # Average Reward per Episode per Drone
    plt.figure()
    for i in range(num_drones):
        avg_reward = np.cumsum(data['rewards_per_episode_per_drone'][i]) / (np.arange(len(data['rewards_per_episode_per_drone'][i])) + 1)
        plt.plot(avg_reward, label=f'Drone {i+1}')
    plt.title('Average Reward per Episode per Drone')
    plt.xlabel('Time Step')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()

    # Oil Spill Concentration Heatmap
    plt.figure(figsize=(10, 8))
    oil_conc_grid = data['average_oil_concentration_grid'].T
    mask = oil_conc_grid == 0
    sns.heatmap(oil_conc_grid, cmap='Blues', mask=mask, cbar_kws={'label': 'Average Oil Concentration'})
    plt.title('Average Oil Spill Concentration')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')

    # Plot oil spill origins
    oil_spill_origins = data.get('oil_spill_origins', [])
    if oil_spill_origins:
        x_coords = [origin[0] for origin in oil_spill_origins]
        y_coords = [origin[1] for origin in oil_spill_origins]
        plt.scatter(x_coords, y_coords, c='red', marker='x', label='Oil Spill Origins')
        plt.legend()

    plt.show()

    # Drone Coverage Map (Heatmap of Drone Positions)
    plt.figure(figsize=(10, 8))
    drone_grid = data['drone_coverage_grid'].T
    mask = drone_grid == 0
    sns.heatmap(drone_grid, cmap='viridis', mask=mask, cbar_kws={'label': 'Visit Frequency'})
    plt.title('Drone Coverage Map')
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

    # Average Detection Delay per Drone
    for i in range(num_drones):
        if len(data['detection_delays_per_drone'][i]) > 0:
            plt.figure()
            plt.hist(data['detection_delays_per_drone'][i], bins=20, edgecolor='black')
            plt.title(f'Detection Delays for Drone {i+1}')
            plt.xlabel('Detection Delay (Minutes)')
            plt.ylabel('Frequency')
            plt.show()
        else:
            print(f"No detection delays to plot for Drone {i+1}.")

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
    checkpoint_path = 'checkpoint5.pth.tar'
    num_drones = 3 # Set the number of drones as per your training
    num_episodes = 50  # Number of evaluation episodes

    env = DroneEnv(num_drones=num_drones)
    agents = load_agents(checkpoint_path, num_drones, env)
    data = evaluate_agents(agents, env, num_episodes)
    plot_results(data, num_drones)

if __name__ == "__main__":
    main()