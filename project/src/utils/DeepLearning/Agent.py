import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

from .CNN import CNN
from .Critic import Critic


class Agent:
    def __init__(self, num_actions, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, device='cpu'):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.cnn = CNN(num_actions).to(device)
        self.critic = Critic().to(device)
        self.optimizer_cnn = optim.Adam(self.cnn.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.001)
        self.memory = deque(maxlen=10_000)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.cnn(state)
            return torch.argmax(q_values, dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in minibatch:
            state, action, reward, next_state, done = self.memory[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        q_values = self.cnn(states)
        target = q_values.clone().detach()
        target_action = actions.squeeze(1)
        target = target.squeeze(1)
        target = target.unsqueeze(1)
        target = target.scatter_(1, target_action.unsqueeze(1), target)
        target = rewards + (1 - dones) * self.gamma * target.max(1, keepdim=True)[0]
        loss = F.mse_loss(q_values, target)
        self.optimizer_cnn.zero_grad()
        loss.backward()
        self.optimizer_cnn.step()

    def decrease_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
