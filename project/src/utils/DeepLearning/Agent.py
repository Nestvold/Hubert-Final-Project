from .Actor import Actor
from .Critic import Critic

import torch
from torch.optim import Adam


import os
import datetime
from math import exp
from numpy import array, float32
from random import random, randint
from gym.wrappers import FrameStack


class Agent:
    def __init__(self, n_stacks, environments, name: str = 'Hubert', cuda_nr: int = 0, learning_rate: float = 0.001,
                 gamma: float = 0.999, epsilon: float = 0.8, epsilon_min: float = 0.001):
        self.name = name
        self.environments = environments
        self.device = torch.device(f'cuda:{cuda_nr}' if torch.cuda.is_available() else 'cpu')

        self.actor = Actor(input_channels=n_stacks, n_actions=3, device=self.device)
        self.critic = Critic(n_stacks=n_stacks, device=self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.learning_rate)

        self.max_steps_per_episode = 500
        self.n_episodes = 1_000 if self.device == 'cpu' else 100_000

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = self.n_episodes

    def epsilon_by_episode(self, episode):
        return self.epsilon_final + (self.epsilon - self.epsilon) * exp(-1. * episode / self.epsilon_decay)

    # Function to preprocess observation
    def observation_to_tensor(self, obs):
        obs = array(obs).reshape(1, self.n_stacks, 9, 9).astype(float32)
        return torch.tensor(obs, device=self.device)

    def train(self):

        print(f'[TRAINING STARTED]')
        print(f'Number of episodes: {self.n_episodes:,}'.replace(',', ' '))
        print()

        try:
            # Training loop
            for episode in range(self.n_episodes):
                # Instantiating environment
                env = self.environments[episode % len(self.environments)]
                env = FrameStack(env=env, num_stack=self.n_stacks)

                obs, _ = env.reset()
                done = False
                total_reward = 0
                total_energy = 0
                epsilon = self.epsilon_by_episode(episode)

                for step in range(self.max_steps_per_episode):
                    # print(env.y, env.x)
                    obs_tensor = self.observation_to_tensor(obs)
                    action_prob = self.actor(obs_tensor)
                    # Exploration - exploitation tradeoff
                    if random() < epsilon:
                        action = randint(0, 2)
                    else:
                        action = torch.multinomial(action_prob, 1).item()

                    next_obs, reward, done, _, data = env.step(action)
                    next_obs_tensor = self.observation_to_tensor(next_obs)

                    # Calculate the target value using the critic network
                    target_value = reward + self.gamma * self.critic(next_obs_tensor).item() * (1 - int(done))

                    # Update the critic network
                    value = self.critic(obs_tensor)
                    critic_loss = (value - target_value) ** 2
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                    self.critic_optimizer.step()

                    # Update the actor network
                    advantage = target_value - value.item()
                    actor_loss = -torch.log(action_prob[:, action]) * advantage
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                    self.actor_optimizer.step()

                    if total_energy >= 150:
                        break

                    total_energy += data['energy']
                    total_reward += reward
                    obs = next_obs

                print(
                    f'Episode: {episode + 1:<8} Environment: {env.name:<18} Best height: {env.best_y:<5} Energy: {total_energy:<10} Total Reward: {round(total_reward, 2):<7}')

        except KeyboardInterrupt:
            print("\nTraining was interrupted. Saving models...")

        # Save models
        now = datetime.now()
        formatted_time = now.strftime('%m-%d-%H-%M')
        if not os.path.exists(f'src/models/{formatted_time}'):
            os.makedirs(f'src/models/{formatted_time}')
        torch.save(self.actor.state_dict(), f'src/models/{formatted_time}/actor_model.pth')
        torch.save(self.critic.state_dict(), f'src/models/{formatted_time}/critic_model.pth')
        print("Models saved.")

