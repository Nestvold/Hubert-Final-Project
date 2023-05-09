# Utils module
from project.src.utils import Actor, Critic, ENVIRONMENTS

# Other modules
import os
import torch
import numpy as np
from math import exp
from torch.optim import Adam
from datetime import datetime
from random import random, randint
from gym.wrappers import FrameStack

envs = ENVIRONMENTS(folder_path='resources/training_maps/').list_of_envs

n_stacks = 5
cuda_nr = 2

# Setting device
device = torch.device(f'cuda:{cuda_nr}' if torch.cuda.is_available() else 'cpu')

# Actor and critic
actor = Actor(input_channels=n_stacks, n_actions=3, device=device)
critic = Critic(n_stacks=n_stacks, device=device)

# Hyperparameters
max_steps_per_episode = 500
n_episodes = 100 if device == 'cpu' else 100_000
learning_rate = 0.001
gamma = 0.999

# Set up the optimizers
actor_optimizer = Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = Adam(critic.parameters(), lr=learning_rate)

# E-greedy
epsilon_start = 0.8
epsilon_final = 0.001
epsilon_decay = n_episodes


def epsilon_by_episode(episode):
    return epsilon_final + (epsilon_start - epsilon_final) * exp(-1. * episode / epsilon_decay)


# Function to preprocess observation
def observation_to_tensor(obs):
    obs = np.array(obs).reshape(1, n_stacks, 9, 9).astype(np.float32)
    return torch.tensor(obs, device=device)


def train():
    print(f'[TRAINING STARTED]')
    print(f'Number of episodes: {n_episodes:,}'.replace(',', ' '))
    print()

    try:
        # Training loop
        for episode in range(n_episodes):
            # Instantiating environment
            env = envs[episode % len(envs)]
            env = FrameStack(env=env, num_stack=n_stacks)

            obs, _ = env.reset()
            done = False
            total_reward = 0
            total_energy = 0
            epsilon = epsilon_by_episode(episode)

            for step in range(max_steps_per_episode):
                # print(env.y, env.x)
                obs_tensor = observation_to_tensor(obs)
                action_prob = actor(obs_tensor)
                # Exploration - exploitation tradeoff
                if random() < epsilon:
                    action = randint(0, 2)
                else:
                    action = torch.multinomial(action_prob, 1).item()

                next_obs, reward, done, _, data = env.step(action)
                next_obs_tensor = observation_to_tensor(next_obs)

                # Calculate the target value using the critic network
                target_value = reward + gamma * critic(next_obs_tensor).item() * (1 - int(done))

                # Update the critic network
                value = critic(obs_tensor)
                critic_loss = (value - target_value) ** 2
                critic_optimizer.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                critic_optimizer.step()

                # Update the actor network
                advantage = target_value - value.item()
                actor_loss = -torch.log(action_prob[:, action]) * advantage
                actor_optimizer.zero_grad()
                actor_loss.backward()
                # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor_optimizer.step()

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
    if not os.path.exists(f'Levels/models/{formatted_time}'):
        os.makedirs(f'Levels/models/{formatted_time}')
    torch.save(actor.state_dict(), f'src/models/{formatted_time}/actor_model.pth')
    torch.save(critic.state_dict(), f'src/models/{formatted_time}/critic_model.pth')
    print("Models saved.")


train()
