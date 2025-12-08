from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class PatchAttentionPool(nn.Module):
    """Self-attention over patch tokens → pooled descriptor (fine texture)."""
    def __init__(self, dim: int, n_heads: int = 4, out_dim: Optional[int] = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim or dim)

    def forward(self, patches: Tensor) -> Tensor:
        """
        Args:
            patches: [B, P, D] patch tokens (no CLS). If P==1 it's fine; we still pool.
        Returns:
            [B, D_out]
        """
        x, _ = self.attn(patches, patches, patches)  # [B, P, D]
        x = self.ln(x)
        pooled = x.mean(dim=1)                       # [B, D]
        return self.proj(pooled)                     # [B, D_out]


class MaterialSidecar(nn.Module):
    """
    Auxiliary encoder specialized for material/texture, without touching the v1 backbone.

    It uses a small ViT from timm, pulls patch tokens, applies PatchAttentionPool,
    then projects to the model's common embedding dimension.
    """
    def __init__(
        self,
        out_dim: int = 512,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
    ):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError(
                "timm is required for MaterialSidecar. Install with `pip install timm`."
            ) from e

        self.side = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        d = getattr(self.side, "num_features", None)
        if d is None:
            # Fallback: infer feature dim by a tiny forward
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feats = self.side(dummy)
                d = feats.shape[-1] if feats.ndim >= 2 else int(feats.shape[1])
        self.pool = PatchAttentionPool(dim=d, n_heads=4, out_dim=out_dim)
        self.proj = nn.Linear(out_dim, out_dim)

    def _extract_patches(self, tokens: Tensor | dict | tuple) -> Tensor:
        """
        Normalize various timm outputs to patch tokens [B, P, D] (no CLS).
        """
        # Some timm models return dicts like {"x": tokens}
        if isinstance(tokens, dict):
            tokens = tokens.get("x", next(iter(tokens.values())))
        # If it's a tuple/list, take the first element as features
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]

        # tokens could be:
        #  - [B, 1+P, D] (CLS + patches) → drop CLS
        #  - [B, P, D] (patches only)
        #  - [B, D] (pooled) → synthesize P=1
        if tokens.ndim == 3:
            if tokens.size(1) > 1:
                # Assume first token is CLS
                return tokens[:, 1:, :]
            return tokens  # already [B, 1, D]
        elif tokens.ndim == 2:
            return tokens.unsqueeze(1)  # [B, 1, D]
        else:
            raise ValueError(f"Unsupported token shape: {tokens.shape}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: images [B, 3, H, W]
        Returns:
            [B, out_dim] material-focused descriptor
        """
        tokens = self.side.forward_features(x)
        patches = self._extract_patches(tokens)   # [B, P, D]
        z = self.pool(patches)                    # [B, out_dim]
        return self.proj(z)                       # [B, out_dim]
