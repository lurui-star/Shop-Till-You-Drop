"""
Beam / nucleus decoding WITHOUT editing the v1 GRUCaptionDecoder.
We build a one-step function by capturing the decoder's modules and hidden
state in a tiny helper (`Stepper`).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


class Stepper:
    """Runs a single decoding step, preserving GRU hidden state across calls."""

    def __init__(self, decoder, img_feat: Tensor):
        self.dec = decoder
        # img_feat: [B, D] → initial hidden state [1, B, E]
        self.h = torch.tanh(self.dec.img2hid(img_feat)).unsqueeze(0)
        self.device = img_feat.device

    @torch.no_grad()
    def step(self, last_token_ids: Tensor) -> Tensor:
        """Advance one token. Args: last_token_ids: [B, 1]. Returns logits [B, V]."""
        x = self.dec.embed(last_token_ids.to(self.device))  # [B,1,E]
        out, self.h = self.dec.gru(x, self.h)               # [B,1,E]
        return self.dec.proj(out[:, -1, :])                 # [B,V]


@torch.no_grad()
def greedy(decoder, img_feat: Tensor, bos_id: int, eos_id: int, max_len: int = 16) -> Tensor:
    """Greedy decoding. Returns token ids [B, L]."""
    bsz = img_feat.size(0)
    stepper = Stepper(decoder, img_feat)
    y = torch.full((bsz, 1), bos_id, dtype=torch.long, device=img_feat.device)
    for _ in range(max_len - 1):
        logits = stepper.step(y[:, -1:])
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        y = torch.cat([y, nxt], dim=1)
        if torch.all(nxt.squeeze(1) == eos_id):
            break
    return y


@torch.no_grad()
def beam_search(
    decoder,
    img_feat: Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int = 16,
    beam_size: int = 5,
    length_penalty: float = 0.7,
) -> Tensor:
    """
    Simple per-example beam search. Returns padded sequences [B, L].
    length_penalty > 0 favors longer sequences a bit less.
    """
    device = img_feat.device
    B = img_feat.size(0)
    outs: List[Tensor] = []

    for b in range(B):
        stepper = Stepper(decoder, img_feat[b : b + 1])  # single example
        beams = [(0.0, [bos_id])]
        finished = []

        for _ in range(max_len - 1):
            new_beams = []
            for logp, seq in beams:
                if seq[-1] == eos_id:
                    finished.append((logp, seq))
                    continue
                logits = stepper.step(torch.tensor([[seq[-1]]], device=device))  # [1,V]
                log_probs = F.log_softmax(logits[0], dim=-1)                     # [V]
                topk_logp, topk_idx = torch.topk(log_probs, k=beam_size)
                for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                    new_beams.append((logp + lp, seq + [int(idx)]))
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

        finished.extend(beams)
        # length normalization
        finished = [
            (lp / ((5 + len(s)) ** length_penalty / (6 ** length_penalty)), s) for lp, s in finished
        ]
        best_seq = max(finished, key=lambda x: x[0])[1]
        outs.append(torch.tensor(best_seq, device=device, dtype=torch.long))

    return torch.nn.utils.rnn.pad_sequence(outs, batch_first=True, padding_value=bos_id)


@torch.no_grad()
def nucleus(
    decoder,
    img_feat: Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int = 16,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> Tensor:
    """Top-p (nucleus) sampling. Returns token ids [B, L]."""
    B = img_feat.size(0)
    stepper = Stepper(decoder, img_feat)
    y = torch.full((B, 1), bos_id, dtype=torch.long, device=img_feat.device)

    for _ in range(max_len - 1):
        logits = stepper.step(y[:, -1:]) / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)  # [B,V]
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        K = (cum <= top_p).sum(dim=-1).clamp(min=1)  # [B]

        nxt_ids = []
        for i in range(B):
            keep = sorted_idx[i, : K[i]]
            p = probs[i, keep] / probs[i, keep].sum()
            nxt_ids.append(keep[torch.multinomial(p, 1)])
        nxt = torch.stack(nxt_ids).unsqueeze(1)  # [B,1]

        y = torch.cat([y, nxt], dim=1)
        if torch.all(nxt.squeeze(1) == eos_id):
            break

    return y
