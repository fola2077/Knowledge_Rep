from test_qlearn import DroneAgent
from test import DroneEnvironment

env = DroneEnvironment(num_drones=3)
agent = DroneAgent(state_dim=4, action_dim=5)

num_episodes = 500
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        actions = [agent.select_action(s) for s in state]  # Actions for each drone
        next_state, rewards, done, _ = env.step(actions)
        
        for i, reward in enumerate(rewards):
            agent.store_transition(state[i], actions[i], reward, next_state[i], done)
        
        agent.train_step()
        total_reward += sum(rewards)
        state = next_state

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
