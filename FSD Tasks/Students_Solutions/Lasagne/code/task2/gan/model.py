import torch
import torch.nn as nn

LABELS = 12
LENGTH = 128
FEATURES = 6

#### GENERATORS

class LSTMGenerator(nn.Module):
    def __init__(self, noise_dim=5, label_dim=12, embed_dim=3, hidden_dim=32, linear_dim=64, 
                 num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.label_dim = label_dim
        self.noise_dim = noise_dim
        self.linear_dim = linear_dim
        # Label embedding or one-hot encoding size
        self.label_embedding = nn.Embedding(num_embeddings=label_dim, embedding_dim=embed_dim)

        # Project noise + label to LSTM input size
        # self.fc = nn.Linear(noise_dim + embed_dim, hidden_dim)

        # LSTM for sequence generation
        self.lstm = nn.LSTM(input_size=noise_dim + embed_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)

        # Project LSTM output to feature dimension
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim*LENGTH, FEATURES*LENGTH)
        )

    def forward(self, noise, labels):
        """
        noise: Tensor of shape (batch_size, noise_dim)
        labels: Tensor of shape (batch_size,) with integer class labels
        """
        # One-hot or embedded labels
        label_embed = self.label_embedding(labels).unsqueeze(1).repeat(1, LENGTH, 1)   # (batch_size, LENGTH, label_dim)

        # Concatenate noise and labels
        lstm_input = torch.cat((noise, label_embed), dim=-1)  # (batch_size, LENGTH, noise_dim + label_dim)

        # Initial LSTM input (broadcasted to entire sequence)
        # initial_input = self.fc(conditioned_input)  # (batch_size, hidden_dim)
        # initial_input = conditioned_input  # (batch_size, hidden_dim)
        # lstm_input = initial_input.unsqueeze(1).repeat(1, LENGTH, 1)  # (batch_size, seq_len, hidden_dim)

        # Generate sequence
        lstm_out, _ = self.lstm(lstm_input)
        output_seq = self.output_layer(lstm_out)  # (batch_size, seq_len, feature_dim)
        return  output_seq.reshape(-1, LENGTH, FEATURES)

    def sample(self, batch_size, device=None):
        return torch.randn(batch_size, LENGTH, self.noise_dim, device=device)


class CNNGenerator(nn.Module):
    def __init__(self, latent_dim=100, embedding_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        # Embedding for class label
        self.label_emb = nn.Embedding(LABELS, embedding_dim)

        # Project noise + label to a small feature map
        self.project = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 256 * 8),  # 8 is starting temporal resolution
            # nn.BatchNorm1d(256 * 8),
            nn.ReLU(True)
        )

        # CNN decoder (upsample from 8 to 128)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # 16 -> 32
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),    # 32 -> 64
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, FEATURES, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.Identity()  # or Identity if data not normalized
        )

    def sample(self, batch_size, device=None):
        return 0.2*torch.randn(batch_size, self.latent_dim, device=device)

    def forward(self, z, labels):
        # Embed and concatenate label
        label_embed = self.label_emb(labels)
        x = torch.cat([z, label_embed], dim=1)

        # Project and reshape to [B, C, T]
        x = self.project(x)  # [B, 256*8]
        x = x.view(x.size(0), 256, 8)  # [B, 256, 8]

        return self.net(x).moveaxis(-1,-2)  # [B, 6, 128]

#### DISCRIMINATORS

class LSTMDiscriminator(nn.Module):
    def __init__(self, label_dim=12, hidden_dim=32, embed_dim=3,
                 num_layers=1):
        super().__init__()
        self.label_embedding = nn.Embedding(num_embeddings=label_dim, embedding_dim=embed_dim)

        self.lstm = nn.LSTM(input_size=6 + embed_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, sequence, labels, prob=True):
        """
        sequence: Tensor of shape (batch_size, seq_len, input_dim)
        labels: Tensor of shape (batch_size,) with integer class labels
        """
        batch_size, seq_len, _ = sequence.shape

        # Embed labels and repeat across sequence length
        label_embed = self.label_embedding(labels)  # (batch_size, label_dim)
        label_embed = label_embed.unsqueeze(1).repeat(1, LENGTH, 1)  # (batch_size, seq_len, label_dim)

        # Concatenate label embedding with input sequence at each time step
        conditioned_input = torch.cat((sequence, label_embed), dim=2)  # (batch_size, seq_len, input_dim + label_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(conditioned_input)

        # Use the final hidden state (last timestep) for classification
        final_hidden = lstm_out[..., -1, :]  # (batch_size, hidden_dim)
        logits = self.output_layer(final_hidden)  # (batch_size, 1)
        if prob:
            return torch.sigmoid(logits) # probability of being real
        else:
            return logits


class CNNDiscriminator(nn.Module):
    def __init__(self, embedding_dim=6, seq_len=128):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        # Class embedding: same spatial shape as input [B, 6, 128]
        self.label_emb = nn.Embedding(LABELS, embedding_dim)

        # CNN encoder
        self.net = nn.Sequential(
            nn.Conv1d(FEATURES + embedding_dim, 32, kernel_size=4, stride=2, padding=1),  # 128 → 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 → 32
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),  # 32 → 16
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),  # 16 → 8
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 1, kernel_size=8),  # 8 → 1 (global)
        )

    def forward(self, x, labels, prob=True):
        """
        x: [B, 6, 128]
        labels: [B]
        """
        x = x.squeeze()
        if x.shape[1] != FEATURES:
            x = x.moveaxis(-1,-2)
        
        label_embed = self.label_emb(labels)  # [B, 6]
        label_embed = label_embed.unsqueeze(2).expand(-1, -1, self.seq_len)  # [B, 6, 128]

        # Concatenate along channel dimension
        x = torch.cat([x, label_embed], dim=1)  # [B, 12, 128]

        out = self.net(x)  # [B, 1, 1]
        out = out.squeeze(2).squeeze(1)  # [B]
        if prob:
            return torch.sigmoid(out) 
        else:
            return out