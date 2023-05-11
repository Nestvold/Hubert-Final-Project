from .Actor import Actor
from .Critic import Critic

import torch
from torch.optim import Adam
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import os
from math import exp
from datetime import datetime
from numpy import array, float32, zeros
from random import random, randint
from gym.wrappers import FrameStack


class Agent:
    def __init__(self, n_stacks, environments, name: str = 'Hubert', cuda_nr: int = 0, learning_rate: float = 0.001,
                 gamma: float = 0.999, epsilon: float = 0.8, epsilon_min: float = 0.001):
        self.name = name
        self.n_stacks = n_stacks
        self.environments = environments
        self.device = torch.device(f'cuda:{cuda_nr}' if torch.cuda.is_available() else 'cpu')

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.actor = Actor(input_channels=self.n_stacks, n_actions=3, device=self.device)
        self.critic = Critic(n_stacks=self.n_stacks, device=self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.learning_rate)

        self.max_steps_per_episode = 500
        self.n_episodes = 1_000 if self.device == 'cpu' else 100_000

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = self.n_episodes

    def epsilon_by_episode(self, episode):
        return self.epsilon_min + (self.epsilon - self.epsilon) * exp(-1. * episode / self.epsilon_decay)

    # Function to preprocess observation
    def observation_to_tensor(self, obs):
        obs = array(obs).reshape(1, self.n_stacks, 9, 9).astype(float32)
        return torch.tensor(obs, device=self.device)

    def train(self):

        # Logging
        writer = SummaryWriter(log_dir=f'logs/{self.learning_rate}')
        episode_rewards = []
        critic_losses = []
        actor_losses = []

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

                total_reward = 0
                total_energy = 0

                actions = zeros(shape=3)

                for step in range(self.max_steps_per_episode):

                    obs_tensor = self.observation_to_tensor(obs)
                    action_prob = self.actor(obs_tensor)
                    # print(action_prob)
                    # Exploration - exploitation tradeoff
                    dist = Categorical(action_prob)
                    action = dist.sample()
                    probability_of_action = dist.log_prob(action)

                    actions[action.item()] += 1

                    next_obs, reward, done, _, data = env.step(action.item())
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
                    actor_loss = -probability_of_action * advantage
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                    self.actor_optimizer.step()

                    # Logging
                    episode_rewards.append(reward)
                    critic_losses.append(critic_loss.item())
                    actor_losses.append(actor_loss.item())

                    avg_reward = sum(episode_rewards) / len(episode_rewards) if len(episode_rewards) > 0 else 0.0
                    avg_critic_loss = sum(critic_losses) / len(critic_losses) if len(critic_losses) > 0 else 0.0
                    avg_actor_loss = sum(actor_losses) / len(actor_losses) if len(actor_losses) > 0 else 0.0

                    writer.add_scalar('Reward/Average Reward', avg_reward, episode)
                    writer.add_scalar('Loss/Critic Loss', avg_critic_loss, episode)
                    writer.add_scalar('Loss/Actor Loss', avg_actor_loss, episode)

                    # If he reaches the max-cap on energy consumption
                    if total_energy >= 150:
                        break

                    total_energy += data['energy']
                    total_reward += reward
                    obs = next_obs

                print(f'Episode: {episode + 1:<8} Environment: {env.name:<18} Best height: {(-env.best_y + env.grid.shape[0])-2:>2}/{env.grid.shape[0]-2:<5} Energy: {total_energy:<10} Total Reward: {round(total_reward, 2):<7} Actions: {actions}')

            print('\nTraining completed.', end=' ')

        except KeyboardInterrupt:
            print('\nTraining was interrupted.', end=' ')

        writer.close()

        print("Saving models...")
        now = datetime.now()
        formatted_time = now.strftime('%m-%d-%H-%M')

        if not os.path.exists(f'src/models/{formatted_time}'):
            os.makedirs(f'src/models/{formatted_time}')

        torch.save(self.actor.state_dict(), f'src/models/{formatted_time}/actor_model.pth')
        torch.save(self.critic.state_dict(), f'src/models/{formatted_time}/critic_model.pth')

        print("Models saved.")
