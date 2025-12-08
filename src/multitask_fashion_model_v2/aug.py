import random
from typing import Tuple

import torch
from torch import Tensor


def _mixup_data(x: Tensor, y: Tensor, alpha: float = 0.2) -> Tuple[Tensor, Tuple[Tensor, Tensor, float]]:
    """
    MixUp augmentation.
    Returns mixed images and a tuple (y_a, y_b, lam) for loss computation.
    """
    if alpha <= 0:
        return x, (y, y, 1.0)

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, (y, y[idx], lam)


def _mixup_crit(criterion, pred: Tensor, y_tuple: Tuple[Tensor, Tensor, float]) -> Tensor:
    """
    Criterion wrapper for MixUp targets.
    criterion should accept (pred, target) and return a scalar loss.
    """
    y_a, y_b, lam = y_tuple
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def _cutmix_data(x: Tensor, y: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tuple[Tensor, Tensor, float]]:
    """
    CutMix augmentation.
    Returns patched images and a tuple (y_a, y_b, lam_adj) for loss computation.
    """
    if alpha <= 0:
        return x, (y, y, 1.0)

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    b, _, h, w = x.size()
    idx = torch.randperm(b, device=x.device)

    cx, cy = random.randint(0, w), random.randint(0, h)
    cut_w = int(w * (1.0 - lam) ** 0.5)
    cut_h = int(h * (1.0 - lam) ** 0.5)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)

    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adj = 1.0 - ((x2 - x1) * (y2 - y1)) / float(w * h)
    return x, (y, y[idx], lam_adj)


def mixup_or_cutmix(
    x: Tensor,
    y: Tensor,
    p_mixup: float = 0.4,
    p_cutmix: float = 0.2,
) -> Tuple[Tuple[Tensor, Tuple[Tensor, Tensor, float]], str]:
    """
    Randomly apply MixUp or CutMix based on probabilities.

    Returns:
        ((x_out, (y_a, y_b, lam)), mode)
        where mode ∈ {"mixup", "cutmix", "none"}.
    """
    r = random.random()
    if r < p_mixup:
        return _mixup_data(x, y, alpha=0.2), "mixup"
    if r < p_mixup + p_cutmix:
        return _cutmix_data(x, y, alpha=1.0), "cutmix"
    return (x, (y, y, 1.0)), "none"
