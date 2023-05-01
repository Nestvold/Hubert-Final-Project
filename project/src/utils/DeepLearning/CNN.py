from torch.nn import Module, Conv2d, Linear, ReLU


class CNN(Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = Linear(in_features=64 * 7 * 7, out_features=512)
        self.fc2 = Linear(in_features=512, out_features=3)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
