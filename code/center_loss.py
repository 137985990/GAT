import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """Center Loss for feature clustering by class.
    Maintains one center per class and penalizes distance of features to their class center.
    """
    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.5, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_buffer('centers', torch.zeros(num_classes, feat_dim, device=device))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.bool, device=device))

    @torch.no_grad()
    def _init_centers(self, features, labels):
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.any():
                self.centers[c] = features[mask].mean(dim=0)
        self.initialized.fill_(True)

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        # features: [B, D], labels: [B]
        if not self.initialized.item():
            self._init_centers(features, labels)
        batch_centers = self.centers[labels]
        diff = features - batch_centers
        loss = (diff.pow(2).sum(dim=1)).mean()
        # update centers (moving average)
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.any():
                    c_feat = features[mask].mean(dim=0)
                    self.centers[c] = (1 - self.alpha) * self.centers[c] + self.alpha * c_feat
        return loss