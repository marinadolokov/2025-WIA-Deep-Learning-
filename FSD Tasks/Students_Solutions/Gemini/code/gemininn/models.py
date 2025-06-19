import torch
import torch.nn as nn
import torch.nn.functional as F




class ConfigurableTwoBranchCNN(nn.Module):
    """
    Two-branch CNN with fully dynamic flatten sizing.

    Args:
        sensor_shape (tuple): (height, width) of your sensor input, e.g. (128,6).
        num_aux_feats (int): size of the auxiliary feature vector.
        num_labels (int): number of output classes/labels.
        conv_channels (list of int): output channels for each Conv2d layer.
        kernel_size (int or tuple): conv kernel size.
        padding (int or tuple): conv padding.
        pool_sizes (list of tuple): pooling sizes after each conv.
        in_channels (int): # of input channels (default 1).
        mlp_hidden (int): hidden size of the aux MLP.
        head_hidden (int): hidden size of the combined head.
        dropout (float): dropout in head.
    """
    def __init__(
        self,
        sensor_shape,
        num_aux_feats,
        num_labels,
        conv_channels,
        kernel_size=3,
        padding=1,
        pool_sizes=None,
        in_channels=1,
        mlp_hidden=32,
        head_hidden=128,
        dropout=0.5
    ):
        super().__init__()
        H, W = sensor_shape

        # default pooling if none provided
        if pool_sizes is None:
            pool_sizes = [(2,2)] * len(conv_channels)
        assert len(pool_sizes) == len(conv_channels), "pool_sizes must match conv layers"

        # build CNN branch
        layers = []
        prev = in_channels
        for out_ch, pool in zip(conv_channels, pool_sizes):
            layers += [
                nn.Conv2d(prev, out_ch, kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool)
            ]
            prev = out_ch
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        # compute flattened dim
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            flat_dim = self.cnn(dummy).shape[1]

        # aux MLP
        self.mlp_aux = nn.Sequential(
            nn.Linear(num_aux_feats, mlp_hidden),
            nn.ReLU(inplace=True)
        )

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim + mlp_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_labels)
        )

    def forward(self, sens, aux):
        """
        sens: Tensor of shape (B, in_channels, H, W)
        aux:  Tensor of shape (B, num_aux_feats)
        """
        h1 = self.cnn(sens)    # → (B, flat_dim)
        h2 = self.mlp_aux(aux) # → (B, mlp_hidden)
        return self.classifier(torch.cat([h1, h2], dim=1))





class ResNetBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=(1,1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        
        self.downsample = None
        if stride != (1,1) or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return F.relu(out)

class FlexibleResNet(nn.Module):
    """Builds a ResNet with a variable number of layers.

    Args:
        layers (list[int]): number of BasicBlocks in each stage.
        channels (list[int]): output channels for each stage (must match len(layers)).
        num_aux_feats (int): size of auxiliary feature vector.
        num_labels (int): number of target labels.
    """
    def __init__(self, layers, channels, num_aux_feats, num_labels, dropout=0.3, in_chanels=1):
        super().__init__()
        assert len(layers) == len(channels), "layers and channels length mismatch"

        self.stem = nn.Sequential(
            nn.Conv2d(in_chanels, channels[0], kernel_size=3, stride=(2,1), padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )

        in_ch = channels[0]
        stages = []
        for n_blocks, out_ch in zip(layers, channels):
            stage_layers = []
            # first block possibly downsamples height only
            stage_layers.append(ResNetBasicBlock(in_ch, out_ch, stride=(2,1)))
            in_ch = out_ch
            for _ in range(1, n_blocks):
                stage_layers.append(ResNetBasicBlock(in_ch, out_ch))
            stages.append(nn.Sequential(*stage_layers))
        self.stages = nn.Sequential(*stages)

        # global average pool over height+width
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # aux branch
        self.mlp_aux = nn.Sequential(
            nn.Linear(num_aux_feats, 64),
            nn.ReLU()
        )

        # figure latent dim
        with torch.no_grad():
            dummy = torch.zeros(1, in_chanels, 128, 6)
            feat = self.forward_backbone(dummy)
            self.backbone_dim = feat.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )

    # backbone forward for dim check
    def forward_backbone(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x)
        return torch.flatten(x, 1)

    def forward(self, sens, aux):
        h = self.forward_backbone(sens)
        a = self.mlp_aux(aux)
        out = torch.cat([h, a], dim=1)
        return self.classifier(out)
    

    
class DBVAE(nn.Module):
    def __init__(self, latent_dim=32, num_classes=2):
        super(DBVAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
        )
        self.fc_mu     = nn.Linear(128*4*4, latent_dim)
        self.fc_logvar = nn.Linear(128*4*4, latent_dim)
        # Classifier head
        self.fc_class = nn.Linear(latent_dim, num_classes)
        # Decoder
        self.dec_fc   = nn.Linear(latent_dim, 128*4*4)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32,  1, 3, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 128, 4, 4)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon      = self.decode(z)
        logits     = self.fc_class(z)
        return recon, mu, logvar, logits
    
