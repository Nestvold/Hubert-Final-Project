import torch
import torch.nn as nn
import torch.optim as optim


# Define the CNN network
class CNN(nn.Module):
    def __init__(self, num_actions, num_stacked_frames, device):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_stacked_frames, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, num_actions)
        self.to(device=device)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the Critic network
class Critic(nn.Module):
    def __init__(self, device):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(32 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 1)
        self.to(device=device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the Actor class
class Actor:
    def __init__(self, num_actions, num_stacked_frames, device, learning_rate: float = 0.001):
        self.network = CNN(num_actions, num_stacked_frames, device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.critic = Critic(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def get_action(self, state):
        print(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.network(state)
        action = torch.argmax(action_probs, dim=1).item()
        return action

    def train(self, state, action, reward, next_state):
        self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Compute the action probabilities and values
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        print(state)
        print(next_state)
        action_probs = self.network(state)
        value = self.critic(state)
        next_value = self.critic(next_state)

        # Compute the advantages and update the critic
        target_value = reward + next_value
        td_error = target_value - value
        value_loss = td_error.pow(2).mean()
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network using the advantages
        log_prob = torch.log(action_probs.squeeze(0)[action])
        actor_loss = -(log_prob * td_error.detach()).mean()
        actor_loss.backward()
        self.optimizer.step()