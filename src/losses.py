import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss para classificação multiclasse.
        gamma: O fator de foco. Quanto maior, mais a rede foca nos exemplos difíceis. (2 é o padrão).
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calcula a Cross Entropy normal
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Extrai as probabilidades (pt)
        pt = torch.exp(-ce_loss)
        
        # Aplica a fórmula da Focal Loss: (1 - pt)^gamma * CE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss