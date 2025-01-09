# dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from collections import namedtuple
import logging


# Define the Transition namedtuple
Transition = namedtuple('Transition',
                       ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.array(state, copy=False)
        next_state = np.array(next_state, copy=False)
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # batch = random.sample(self.buffer, batch_size)
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))
        # return state, action, reward, next_state, done
        transitions = random.sample(self.buffer, batch_size)
        return transitions  # Return the transitions directly
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        print(f"Initializing QNetwork with state_dim={state_dim}, action_dim={action_dim}") 
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(
        self,
        drone_id,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        buffer_capacity=50000,
        batch_size=128,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=10000
    ):
        self.drone_id = drone_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5,
            patience=100,
            verbose=True
        )

        # Set up a logger for the agent
        self.logger = logging.getLogger(f'DQNAgent_Drone_{self.drone_id}')
        handler = logging.FileHandler(f'drone_{self.drone_id}.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        self.steps_done += 1
        epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        if random.random() < epsilon:
            action = random.randrange(self.action_dim)
            self.logger.debug(f"Drone {self.drone_id} selected random action {action} with epsilon {epsilon}")
            return action
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
            self.logger.debug(f"Drone {self.drone_id} selected action {action} with epsilon {epsilon}")
            return action
    
    def push_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            self.logger.debug(f"Drone {self.drone_id} has insufficient memory for replay: {len(self.memory)}")
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to PyTorch tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_Q = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next state values using target net
        with torch.no_grad():
            next_Q = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_Q = reward_batch + self.gamma * next_Q * (1 - done_batch)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_Q, target_Q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Add gradient clipping
        self.optimizer.step()   
    
    def update_target_network(self):
        """Update the target network to match the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
