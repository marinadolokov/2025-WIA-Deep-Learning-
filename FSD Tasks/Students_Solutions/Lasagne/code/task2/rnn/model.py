import torch.nn as nn
import torch

# SEQUENCE GENERATOR
class SequenceGenerator(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden


# SEQUENCE GENERATOR WITH EMBEDDING
class SequenceGeneratorEmbedded(nn.Module):
    def __init__(self, sensor_dim=6, label_embed_dim=10, hidden_dim=128, num_layers=2, num_labels=12):
        super().__init__()
        self.label_embed = nn.Embedding(num_labels, label_embed_dim) # embedding labels into vector
        self.rnn = nn.LSTM(sensor_dim + label_embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, sensor_dim)

    def forward(self, x, label, hidden=None):
        label_vec = self.label_embed(label)             # (batch, embed_dim)
        label_vec = label_vec.unsqueeze(1)              # (batch, 1, embed_dim)
        x = torch.cat([x, label_vec], dim=2)            # feed the label embedded vector as additional input: (batch, 1, sensor_dim + embed_dim)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)                              # (batch, 1, sensor_dim)
        return out, hidden
