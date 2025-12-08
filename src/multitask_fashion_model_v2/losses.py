import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.05, weight=None, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction
        self.ignore_index = ignore_index
    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, target, weight=self.weight, reduction='none', ignore_index=self.ignore_index)
        smooth = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean() if self.reduction == "mean" else (loss.sum() if self.reduction == "sum" else loss)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = "mean", ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction
        self.ignore_index = ignore_index
    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        ce = F.nll_loss(log_probs, target, weight=self.weight, reduction='none', ignore_index=self.ignore_index)
        pt = torch.gather(probs, 1, target.unsqueeze(1)).squeeze(1)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction == "mean" else (loss.sum() if self.reduction == "sum" else loss)
