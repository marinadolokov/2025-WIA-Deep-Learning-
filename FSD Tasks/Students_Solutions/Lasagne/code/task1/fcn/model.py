import torch.nn as nn

class FullyConnectedModel(nn.Module):
    def __init__(self):
        super(FullyConnectedModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6, 128)  # first hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)       # second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 12)        # output layer

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x