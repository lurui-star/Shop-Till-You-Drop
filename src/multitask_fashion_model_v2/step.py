from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from torch import Tensor

from .losses import LabelSmoothingCE, FocalLoss
from .aug import mixup_or_cutmix, _mixup_crit
from .patch_pool import MaterialSidecar
from .balance import MovingAverageLossScaler

__all__ = ["train_one_step"]

def train_one_step(
    model,
    batch: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    tokenizer,
    device: torch.device,
    *,
    sidecar: Optional[MaterialSidecar] = None,
    material_loss_kind: str = "labelsmooth",  # "ce" | "labelsmooth" | "focal"
    mixup_p: float = 0.0,
    cutmix_p: float = 0.0,
    w_cat: float = 1.0,
    w_gen: float = 0.5,
    w_mat: float = 1.0,
    w_cap: float = 1.0,
    class_weights_mat: Optional[Tensor] = None,
    balancer: Optional[MovingAverageLossScaler] = None,
    decode_mode: Optional[str] = None,  # <-- NEW (optional, ignored here)
    **kwargs,                              # <-- NEW (swallow future args safely)
) -> Dict[str, float]:
    """One training step for the multi-head model (category/gender/material + caption)."""
    model.train()

    images   = batch["images"].to(device)
    cap_in   = batch["cap_in"].to(device)   if batch["cap_in"] is not None else None
    cap_tgt  = batch["cap_tgt"].to(device)  if batch["cap_tgt"] is not None else None
    y_cat    = batch["y_cat"].to(device)    if batch["y_cat"] is not None else None
    y_gender = batch["y_gender"].to(device) if batch["y_gender"] is not None else None
    y_mat    = batch["y_material"].to(device) if batch["y_material"] is not None else None

    out = model(images, cap_in=cap_in)
    logs: Dict[str, float] = {}
    total = 0.0

    if y_cat is not None and "logits_category" in out:
        Lc = F.cross_entropy(out["logits_category"], y_cat)
        logs["category_ce"] = float(Lc.item())
        total = total + w_cat * (balancer.scale("category", Lc) if balancer else Lc)

    if y_gender is not None and "logits_gender" in out:
        Lg = F.cross_entropy(out["logits_gender"], y_gender)
        logs["gender_ce"] = float(Lg.item())
        total = total + w_gen * (balancer.scale("gender", Lg) if balancer else Lg)

    if cap_tgt is not None and "logits_caption" in out:
        B, T, V = out["logits_caption"].shape
        Lcap = F.cross_entropy(
            out["logits_caption"].reshape(B * T, V),
            cap_tgt.reshape(B * T),
            ignore_index=tokenizer.pad_id,
        )
        logs["caption_nll"] = float(Lcap.item())
        total = total + w_cap * (balancer.scale("caption", Lcap) if balancer else Lcap)

    # Material head (+ optional MixUp/CutMix & sidecar)
    if y_mat is not None and "logits_material" in out:
        if material_loss_kind == "labelsmooth":
            mat_crit = LabelSmoothingCE(smoothing=0.05, weight=class_weights_mat)
        elif material_loss_kind == "focal":
            mat_crit = FocalLoss(gamma=2.0, weight=class_weights_mat)
        else:
            mat_crit = None

        logits_mat = out["logits_material"]
        y_tuple = None
        mix_mode = "none"

        if mixup_p > 0 or cutmix_p > 0:
            (images_aug, y_tuple), mix_mode = mixup_or_cutmix(
                images.clone(), y_mat, p_mixup=mixup_p, p_cutmix=cutmix_p
            )
            if sidecar is not None:
                z_side = sidecar(images_aug)
                logits_mat = model.material_head(z_side)
            else:
                logits_mat = model(images_aug, cap_in=None)["logits_material"]

        if y_tuple is None:
            Lm = mat_crit(logits_mat, y_mat) if mat_crit else F.cross_entropy(logits_mat, y_mat)
        else:
            Lm = _mixup_crit(mat_crit or F.cross_entropy, logits_mat, y_tuple)

        logs["material_ce"] = float(Lm.item())
        total = total + w_mat * (balancer.scale("material", Lm) if balancer else Lm)
        logs["mix_mode"] = mix_mode

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    logs["total"] = float(total.item())
    return logs
