from __future__ import annotations
from typing import Dict, Mapping, Tuple
import torch
from torch import Tensor

class MovingAverageLossScaler:
    """
    EMA-based per-head loss scaler to keep multi-task losses on comparable scales.

    - Bias-corrected EMA
    - Optional warmup steps before scaling
    - Optional min/max clamp on the divisor
    - Helpers to freeze/unfreeze specific heads and scale loss dicts
    """

    def __init__(
        self,
        beta: float = 0.98,
        eps: float = 1e-6,
        *,
        warmup_steps: int = 0,
        min_scale: float = 1e-3,
        max_scale: float = 1e3,
    ) -> None:
        self.beta: float = float(beta)
        self.eps: float = float(eps)
        self.warmup_steps: int = int(warmup_steps)
        self.min_scale: float = float(min_scale)
        self.max_scale: float = float(max_scale)

        self._ema: Dict[str, float] = {}
        self._t: Dict[str, int] = {}
        self._frozen: Dict[str, bool] = {}  # name -> True means bypass scaling

    # ---------- lifecycle ----------
    def reset(self) -> None:
        self._ema.clear()
        self._t.clear()
        self._frozen.clear()

    def state_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "ema": dict(self._ema),
            "t": dict(self._t),
            "beta": self.beta,
            "eps": self.eps,
            "warmup_steps": self.warmup_steps,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "frozen": dict(self._frozen),
        }

    def load_state_dict(self, state: Dict[str, Dict[str, float]]) -> None:
        self._ema = {k: float(v) for k, v in state.get("ema", {}).items()}
        self._t = {k: int(v) for k, v in state.get("t", {}).items()}
        self.beta = float(state.get("beta", self.beta))
        self.eps = float(state.get("eps", self.eps))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.min_scale = float(state.get("min_scale", self.min_scale))
        self.max_scale = float(state.get("max_scale", self.max_scale))
        self._frozen = {k: bool(v) for k, v in state.get("frozen", {}).items()}

    # ---------- controls ----------
    def freeze(self, name: str) -> None:
        self._frozen[name] = True

    def unfreeze(self, name: str) -> None:
        self._frozen[name] = False

    def is_frozen(self, name: str) -> bool:
        return self._frozen.get(name, False)

    # ---------- core ----------
    def scale(self, name: str, loss: Tensor, bias_correct: bool = True) -> Tensor:
        """
        Scale `loss` by its EMA divisor. If frozen or within warmup, returns `loss` unchanged.
        """
        if self.is_frozen(name):
            return loss

        # robust scalar for EMA update
        val = torch.nan_to_num(loss.detach(), nan=0.0, posinf=1e6, neginf=-1e6).item()

        # step
        t = self._t.get(name, 0) + 1
        self._t[name] = t

        # ema update
        ema_prev = self._ema.get(name, val)
        ema = self.beta * ema_prev + (1.0 - self.beta) * val
        self._ema[name] = ema

        if t <= self.warmup_steps:
            return loss  # no scaling during warmup

        denom = ema
        if bias_correct:
            denom = ema / (1.0 - self.beta**t + 1e-12)

        # clamp the scale to avoid extreme divisions
        denom = float(torch.clamp(torch.tensor(denom), min=self.eps))
        denom = max(denom, self.eps)
        inv = 1.0 / denom
        inv = max(min(inv, self.max_scale), self.min_scale)

        return loss * inv

    # ---------- convenience ----------
    def scale_dict(
        self,
        losses: Mapping[str, Tensor],
        bias_correct: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Scale a dict of named losses and return (sum_of_scaled, per_head_scaled).
        """
        scaled: Dict[str, Tensor] = {}
        total = torch.zeros((), dtype=next(iter(losses.values())).dtype, device=next(iter(losses.values())).device)
        for k, v in losses.items():
            s = self.scale(k, v, bias_correct=bias_correct)
            scaled[k] = s
            total = total + s
        return total, scaled
