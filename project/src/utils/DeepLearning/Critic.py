# Torch Neural Network module
from torch.nn import Module, Linear, ReLU


class Critic(Module):
    def __init__(self, n_stacks, device):
        super(Critic, self).__init__()

        self.n_stacks = n_stacks

        self.fc1 = Linear(9 * 9 * self.n_stacks, 512)
        self.fc2 = Linear(512, 1)
        self.relu = ReLU()

        self.device = device
        self.to(self.device)
        print(f'[CRITIC] - {self.device}')

    def forward(self, x):
        x = x.view(-1, 9 * 9 * self.n_stacks)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
