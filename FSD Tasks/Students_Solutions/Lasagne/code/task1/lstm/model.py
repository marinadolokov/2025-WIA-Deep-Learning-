import torch.nn as nn

# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, num_classes=12, num_layers=2, dropout=0.):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # change time major if necessary
        if x.shape[1] == 6:
            x = x.moveaxis(-1,-2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])