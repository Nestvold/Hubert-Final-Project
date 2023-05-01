from torch.nn import Module, Linear, ReLU


class Critic(Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = Linear(9 * 9 * 3, 512)
        self.fc2 = Linear(512, 1)
        self.relu = ReLU()

    def forward(self, x):
        x = x.view(-1, 9 * 9 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
