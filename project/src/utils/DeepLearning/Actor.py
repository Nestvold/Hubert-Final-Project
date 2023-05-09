# Torch Neural Network module
from torch.nn import Module, Conv2d, Linear, ReLU, Softmax


class Actor(Module):
    def __init__(self, input_channels, n_actions, device):
        super(Actor, self).__init__()

        self.conv1 = Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = Linear(in_features=1600, out_features=512)
        self.fc2 = Linear(in_features=512, out_features=n_actions)
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)

        self.device = device
        self.to(self.device)
        print(f'[ACTOR ] - {self.device}')

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 1600)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x + 1e-8
