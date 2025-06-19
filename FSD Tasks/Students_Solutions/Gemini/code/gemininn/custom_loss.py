import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalBCE(nn.Module):
    def __init__(self, pos_weight, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)         # [B,C]
        probs = torch.sigmoid(logits)
        pt = probs*targets + (1-probs)*(1-targets)
        focal = (1 - pt).pow(self.gamma)
        return (focal * bce).mean()


class DBVAELoss(nn.Module):  
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, recon, x, mu, logvar, logits, y):
        if recon.shape[2:] != x.shape[2:]:
            recon = F.interpolate(
                recon,
                size=x.shape[2:],            # (H, W) of the ground truth
                mode='bilinear', 
                align_corners=False
            )
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        cls_ = F.cross_entropy(logits, y)
        return recon_loss + self.beta * kl + cls_