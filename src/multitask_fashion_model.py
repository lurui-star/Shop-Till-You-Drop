import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


# =========================
# Tokenizer (word-level)
# =========================
SPECIAL_TOKENS = {
    "pad": "<pad>",
    "bos": "<bos>",
    "eos": "<eos>",
    "unk": "<unk>",
}

class TextTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.stoi = vocab
        self.itos = {i: s for s, i in vocab.items()}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @staticmethod
    def basic_tokenize(s: str) -> List[str]:
        return str(s).strip().lower().split()

    @classmethod
    def build_from_captions(
        cls,
        captions: List[str],
        min_freq: int = 2,
        max_size: Optional[int] = None,
    ) -> "TextTokenizer":
        from collections import Counter
        cnt = Counter()
        for c in captions:
            cnt.update(cls.basic_tokenize(c))

        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        words = [w for w, n in cnt.items() if n >= min_freq]
        words.sort(key=lambda w: (-cnt[w], w))
        if max_size is not None:
            words = words[: max(0, max_size - len(vocab))]
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
        return cls(vocab)
    

    def encode(self, s: str, add_bos_eos: bool = True) -> List[int]:
        toks = [self.stoi.get(t, self.unk_id) for t in self.basic_tokenize(s)]
        if add_bos_eos:
            return [self.bos_id] + toks + [self.eos_id]
        return toks

    def decode(self, ids: List[int]) -> str:
        # stop at eos if present
        words = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.bos_id, self.pad_id):
                continue
            words.append(self.itos.get(i, SPECIAL_TOKENS["unk"]))
        return " ".join(words)

    
# =========================
# Collate for captioning
# =========================
def collate_batch(samples: List[Dict], tokenizer: TextTokenizer, expect_category: bool = True, expect_gender: bool = True):
    """
    Converts a list of dataset items into a batch:
      images: Bx3xHxW
      y_cat: B (optional)
      y_gender: B (optional)
      cap_in: BxT (teacher forcing input, starts with <bos>)
      cap_tgt: BxT (target, ends with <eos>)
      lengths: B (original lengths with bos/eos)
    """
    images = torch.stack([s["image"] for s in samples], dim=0)

    # category (optional)
    y_cat = None
    if expect_category and ("label" in samples[0]) and (samples[0]["label"] is not None):
        # ensure tensor long
        y_cat = torch.tensor([int(s["label"]) for s in samples], dtype=torch.long)

    # gender (optional)
    y_gender = None
    if expect_gender and ("gender" in samples[0]) and (samples[0]["gender"] is not None):
        vals = [s["gender"] for s in samples]

        # Case 1: already numeric
        if all(isinstance(v, (int, np.integer)) for v in vals):
            y_gender = torch.tensor([int(v) for v in vals], dtype=torch.long)

        # Case 2: strings but no map → skip gender for this batch
        else:
            y_gender = None  # just don't compute gender loss

    # captions
    caps = [s.get("caption", "") for s in samples]
    seqs = [tokenizer.encode(c, add_bos_eos=True) for c in caps]
    lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
    T = int(lengths.max().item())
    cap_in = torch.full((len(samples), T - 1), tokenizer.pad_id, dtype=torch.long)
    cap_tgt = torch.full((len(samples), T - 1), tokenizer.pad_id, dtype=torch.long)
    for i, ids in enumerate(seqs):
        # e.g., [bos, ..., eos] -> input=[bos..second_last], target=[next..eos]
        inp = ids[:-1]
        tgt = ids[1:]
        cap_in[i, :len(inp)] = torch.tensor(inp, dtype=torch.long)
        cap_tgt[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return {
        "images": images,
        "y_cat": y_cat,
        "y_gender": y_gender,
        "cap_in": cap_in,
        "cap_tgt": cap_tgt,
        "lengths": lengths,
    }

# =========================
# Caption Decoder (GRU)
# =========================
class GRUCaptionDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, emb_dim: int = 256, num_layers: int = 1, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.img2hid = nn.Linear(d_model, emb_dim)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, num_layers=num_layers, batch_first=True, dropout=0.0 if num_layers == 1 else dropout)
        self.proj = nn.Linear(emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feat: torch.Tensor, cap_in: torch.Tensor) -> torch.Tensor:
        """
        img_feat: [B, d_model] image embedding
        cap_in:   [B, T] teacher-forced input (token ids)
        returns:  [B, T, V] logits over vocab
        """
        B, T = cap_in.shape
        x = self.embed(cap_in)                  # [B,T,E]
        h0 = torch.tanh(self.img2hid(img_feat)) # [B,E]
        h0 = h0.unsqueeze(0).contiguous()       # [1,B,E]
        x = self.dropout(x)
        out, _ = self.gru(x, h0)                # [B,T,E]
        out = self.proj(self.dropout(out))      # [B,T,V]
        return out

    @torch.no_grad()
    def generate(self, img_feat: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 30) -> torch.Tensor:
        """
        Greedy decoding. Returns [B, L] token ids (incl. BOS, until EOS or max_len).
        """
        B = img_feat.size(0)
        h = torch.tanh(self.img2hid(img_feat)).unsqueeze(0)  # [1,B,E]
        y = torch.full((B, 1), bos_id, dtype=torch.long, device=img_feat.device)

        outputs = [y]
        for _ in range(max_len - 1):
            x = self.embed(y[:, -1:])          # [B,1,E]
            out, h = self.gru(x, h)            # [B,1,E]
            logits = self.proj(out[:, -1, :])  # [B,V]
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)  # [B,1]
            outputs.append(next_tok)
            y = torch.cat(outputs, dim=1)
            # early stop if everyone hit EOS
            if torch.all(next_tok.squeeze(1) == eos_id):
                break
        return y


# =========================
# Backbone (ViT/ResNet)
# =========================
class ImageBackbone(nn.Module):
    def __init__(self, name: str = "vit_base_patch16_224", out_dim: int = 768, proj_dim: int = 512, pretrained: bool = True):
        super().__init__()
        if _HAS_TIMM:
            self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)  # returns pooled feature
            feat_dim = self.backbone.num_features
        else:
            from torchvision.models import resnet50, ResNet50_Weights
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            m.fc = nn.Identity()
            self.backbone = m
            feat_dim = 2048
        self.proj = nn.Linear(feat_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)      # [B, feat_dim]
        z = self.norm(self.proj(f))  # [B, proj_dim]
        return z


# =========================
# Multi-head model
# =========================
@dataclass
class MultiTaskConfig:
    backbone: str = "vit_base_patch16_224"
    proj_dim: int = 512
    num_categories: Optional[int] = None   # set int to enable category head
    num_genders: Optional[int] = None      # set int to enable gender head
    vocab_size: int = 30000
    pad_id: int = 0
    caption_emb_dim: int = 256
    caption_layers: int = 1
    dropout: float = 0.1
    loss_w_category: float = 1.0
    loss_w_gender: float = 0.5
    loss_w_caption: float = 1.0

class FashionMultiTaskModel(nn.Module):
    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = ImageBackbone(cfg.backbone, proj_dim=cfg.proj_dim)

        # heads
        self.category_head = nn.Linear(cfg.proj_dim, cfg.num_categories) if cfg.num_categories else None
        self.gender_head   = nn.Linear(cfg.proj_dim, cfg.num_genders) if cfg.num_genders else None
        self.caption_head  = GRUCaptionDecoder(
            vocab_size=cfg.vocab_size,
            d_model=cfg.proj_dim,
            emb_dim=cfg.caption_emb_dim,
            num_layers=cfg.caption_layers,
            dropout=cfg.dropout,
            pad_id=cfg.pad_id,
        )

    def forward(self, images: torch.Tensor, cap_in: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        images: [B,3,H,W]
        cap_in: [B,T] (optional for training captioning)
        Returns a dict with logits for available heads.
        """
        z = self.backbone(images)  # [B,D]
        out = {"img_feat": z}

        if self.category_head is not None:
            out["logits_category"] = self.category_head(z)  # [B,C]
        if self.gender_head is not None:
            out["logits_gender"]   = self.gender_head(z)    # [B,G]
        if cap_in is not None:
            out["logits_caption"]  = self.caption_head(z, cap_in)  # [B,T,V]
        return out

    @torch.no_grad()
    def generate(self, images: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 30) -> torch.Tensor:
        z = self.backbone(images)
        y = self.caption_head.generate(z, bos_id=bos_id, eos_id=eos_id, max_len=max_len)
        return y


# =========================
# Loss helper
# =========================
def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    y_cat: Optional[torch.Tensor],
    y_gender: Optional[torch.Tensor],
    cap_tgt: Optional[torch.Tensor],
    pad_id: int,
    w_category: float,
    w_gender: float,
    w_caption: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes a weighted sum of losses that are present in outputs.
    """
    losses = {}
    total = 0.0

    if ("logits_category" in outputs) and (y_cat is not None):
        loss_c = F.cross_entropy(outputs["logits_category"], y_cat)
        losses["category_ce"] = float(loss_c.item())
        total = total + w_category * loss_c

    if ("logits_gender" in outputs) and (y_gender is not None):
        loss_g = F.cross_entropy(outputs["logits_gender"], y_gender)
        losses["gender_ce"] = float(loss_g.item())
        total = total + w_gender * loss_g

    if ("logits_caption" in outputs) and (cap_tgt is not None):
        # flatten caption logits/targets
        B, T, V = outputs["logits_caption"].shape
        logits = outputs["logits_caption"].reshape(B * T, V)
        tgt = cap_tgt.reshape(B * T)
        loss_cap = F.cross_entropy(logits, tgt, ignore_index=pad_id)
        losses["caption_nll"] = float(loss_cap.item())
        total = total + w_caption * loss_cap

    return total, losses


# =========================
# Minimal training loop stub
# =========================
def one_training_step(
    model: FashionMultiTaskModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    cfg: MultiTaskConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    images = batch["images"].to(device)
    cap_in = batch["cap_in"].to(device) if batch["cap_in"] is not None else None

    y_cat = batch["y_cat"].to(device) if batch["y_cat"] is not None else None
    y_gender = batch["y_gender"].to(device) if batch["y_gender"] is not None else None
    cap_tgt = batch["cap_tgt"].to(device) if batch["cap_tgt"] is not None else None

    out = model(images, cap_in=cap_in)
    loss, parts = multitask_loss(
        out, y_cat, y_gender, cap_tgt,
        pad_id=cfg.pad_id,
        w_category=cfg.loss_w_category,
        w_gender=cfg.loss_w_gender,
        w_caption=cfg.loss_w_caption,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    parts["total"] = float(loss.item())
    return parts



# =========================
# Worker Init
# =========================
def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
