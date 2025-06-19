import torch
import torch.nn as nn

class CNNGenerator(nn.Module):
    def __init__(self, label_dim, noise_dim, output_length, output_dim, hidden_dim = 512):
        super().__init__()
        self.init_length = output_length // 16
        self.hidden_dim = hidden_dim
        input_dim = noise_dim + label_dim

        self.project = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim * self.init_length),
            nn.ReLU(True)
        )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU()
            )

        self.deconv = nn.Sequential(
            up_block(self.hidden_dim, self.hidden_dim // 2),
            up_block(self.hidden_dim // 2, self.hidden_dim // 4),
            up_block(self.hidden_dim // 4, self.hidden_dim // 8),
            nn.ConvTranspose1d(self.hidden_dim // 8, output_dim, 4, 2, 1),
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)
        x = self.project(x).view(-1, self.hidden_dim, self.init_length)
        x = self.deconv(x)
        return x.permute(0, 2, 1)  # (batch, seq_len, dim)

class CNNDiscriminator(nn.Module):
    def __init__(self, label_dim, input_length, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.embed = nn.Linear(label_dim, input_length)

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(in_c, out_c, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.conv = nn.Sequential(
            conv_block(input_dim, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 1024)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear((input_length // 16) * 1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # x: (B, L, D), labels: (B, label_dim)
        label_proj = self.embed(labels).unsqueeze(1)  # (B, 1, L)
        x = x.permute(0, 2, 1)  # (B, D, L)
        x = x + label_proj  # add projection for conditional discrimination
        out = self.conv(x)
        out = self.flatten(out)
        return self.fc(out)
    


class MLPGenerator(nn.Module):
    def __init__(self, label_dim, noise_dim, output_length, output_dim):
        super().__init__()
        self.output_length = output_length
        self.output_dim = output_dim
        input_dim = noise_dim + label_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_length * output_dim),
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)  # (batch, noise+label)
        x = self.model(x)                       # (batch, output_length*output_dim)
        x = x.view(-1, self.output_length, self.output_dim)  # (batch, seq_len, dim)
        return x


class MLPDiscriminator(nn.Module):
    def __init__(self, label_dim, input_length, input_dim):
        super().__init__()
        input_size = input_length * input_dim + label_dim

        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # x: (B, L, D), labels: (B, label_dim)
        x = x.view(x.size(0), -1)           # flatten (B, L*D)
        x = torch.cat((x, labels), dim=1)   # (B, L*D + label_dim)
        out = self.model(x)
        return out
