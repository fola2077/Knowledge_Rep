# train_dqn.py 
import sys
import numpy as np
import seaborn as sns
import torch
import random
from policy import DroneEnv
from drone import Drone
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm  # Import tqdm for the progress bar
import os

def train(num_episodes=1000, num_drones=1, target_update=10, checkpoint_interval=10, checkpoint_path='checkpoint3.pth.tar'):
    env = DroneEnv(num_drones=num_drones)
    state_dim = env.observation_space
    action_dim = env.action_space
    
    agents = [drone.agent for drone in env.drones]
    
    rewards_per_episode = []
    moving_average_rewards = []
    window_size = 10
    reward_window = deque(maxlen=window_size)
    all_positions = [[] for _ in range(num_drones)]
    
    # Check if a checkpoint exists and load it
    start_episode = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        loaded_drones = len([key for key in checkpoint.keys() if key.startswith('drone_') and key.endswith('_policy_net')])
        if loaded_drones != num_drones:
            print(f"Checkpoint has {loaded_drones} drones, but training is set to {num_drones} drones.")
            print("Starting training from scratch.")
        else:
            start_episode = checkpoint['episode'] + 1
            for i, agent in enumerate(agents):
                agent.policy_net.load_state_dict(checkpoint[f'drone_{i}_policy_net'])
                agent.target_net.load_state_dict(checkpoint[f'drone_{i}_target_net'])
                agent.optimizer.load_state_dict(checkpoint[f'drone_{i}_optimizer'])
            print(f"Resuming training from episode {start_episode}")

    remaining_episodes = num_episodes - start_episode
    if remaining_episodes <= 0:
        print("Training already completed based onn the checkpoint.")
        return
    
    try:
        # Initialize the progress bar
        with tqdm(total=num_episodes, desc="Training Progress", initial=start_episode) as pbar:

            for episode in range(start_episode, num_episodes):
                state = env.reset()
                total_reward = [0 for _ in range(num_drones)]
                done = False
                episode_positions = [[] for _ in range(num_drones)]
                
                while not done:
                    state_tensor = [torch.tensor(s, dtype=torch.float32) for s in state]
                    actions = [agent.select_action(s) for agent, s in zip(agents, state_tensor)]
                    next_state, rewards, done, infos = env.step(actions)

                    # # Convert next_states to tensors
                    # next_state_tensor = [torch.tensor(s, dtype=torch.float32) for s in next_state]
                    
                    # Record positions
                    for i, drone in enumerate(env.drones):
                        position = drone.position
                        all_positions[i].append([position.x, position.y])
                        episode_positions[i].append([position.x, position.y])
                        # detection_delay = infos[i].get('detection_delay', 0.0) 
                        # if detection_delay is None:
                        #     detection_delay = 0.0

                    for i, agent in enumerate(agents):
                        drone = env.drones[i]
                        agent.push_transition(state[i], actions[i], rewards[i], next_state[i], done)
                        agent.update()
                        total_reward[i] += rewards[i]

                    state = next_state
                    
                    # Update target networks periodically
                    if episode % target_update == 0:
                        for agent in agents:
                            agent.update_target_network()
                

                # At the end of the episode
                episode_total_reward = sum(total_reward)
                rewards_per_episode.append(sum(total_reward))
                reward_window.append(episode_total_reward)                

                # Compute moving average
                moving_avg = np.mean(reward_window)
                moving_average_rewards.append(moving_avg)

                pbar.update(1)  # Update the progress bar by one episode
                pbar.set_postfix({'Episode': episode + 1, 'Average Reward': moving_avg})

                # Every 10 episodes, calculate and display the average reward
                if (episode + 1) % checkpoint_interval == 0:
                    avg_reward = np.mean(rewards_per_episode[-checkpoint_interval:])
                    pbar.set_postfix({'Average Reward': avg_reward})
                    tqdm.write(f"Episode {episode+1}/{num_episodes}, Average Reward: {avg_reward}")
                    
                    # Save a checkpoint
                    save_checkpoint(agents, episode, num_drones, checkpoint_path)
    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted by user. Saving current state...")
        save_checkpoint(agents, episode, num_drones, checkpoint_path)
        tqdm.write("Checkpoint saved. Exiting.")
        sys.exit(0)

    # Plotting after training
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode, label='Total Reward per Episode', alpha=0.4)
    plt.plot(moving_average_rewards, label=f'{window_size}-Episode Moving Average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Progress')
    plt.legend()
    plt.show()

    # Heat Map
    for i in range(num_drones):
        positions = np.array(all_positions[i])
        if positions.size == 0:
            print(f"No position data for drone {i+1}.")
            continue
        x = positions[:, 0]
        y = positions[:, 1]
        sns.kdeplot(x=x, y=y, fill=True, cmap="Reds")
        plt.title(f'Drone {i+1} Position Heat Map')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show()

    # Plot trajectories after training
    plot_trajectories(all_positions, num_drones)

def save_checkpoint(agents, episode, num_drones, checkpoint_path):
    """
    Saves the state of the agents to a checkpoint file.
    """
    checkpoint = {'episode': episode}
    for i, agent in enumerate(agents):
        checkpoint[f'drone_{i}_policy_net'] = agent.policy_net.state_dict()
        checkpoint[f'drone_{i}_target_net'] = agent.target_net.state_dict()
        checkpoint[f'drone_{i}_optimizer'] = agent.optimizer.state_dict()
    torch.save(checkpoint, checkpoint_path)
    tqdm.write(f"Checkpoint saved at episode {episode + 1}")

def plot_trajectories(all_positions, num_drones, num_episodes_to_plot=5):
    for i in range(num_drones):
        plt.figure(figsize=(8, 6))
        positions = np.array(all_positions[i])
        plt.plot(positions[:, 0], positions[:, 1], label=f'Drone {i+1} Trajectory')
        plt.title(f'Drone {i+1} Movement Trajectories')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    train(num_episodes=110, num_drones=3, target_update=10)
