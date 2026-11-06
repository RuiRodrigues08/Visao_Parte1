import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        gamma: focusing factor. The higher it is, the more the network focuses on hard examples (2 is the usual default).
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Standard cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Probability of the true class (pt)
        pt = torch.exp(-ce_loss)
        
        # Focal Loss formula: (1 - pt)^gamma * CE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss