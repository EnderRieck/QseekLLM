from __future__ import annotations

import math

from torch.optim import Optimizer

from llmtrain.training.config import SchedulerConfig


class TokenCosineScheduler:
    def __init__(self, optimizer: Optimizer, cfg: SchedulerConfig, *, max_tokens: int) -> None:
        self.optimizer = optimizer
        self.cfg = cfg
        self.max_tokens = max_tokens
        self.decay_tokens = cfg.decay_tokens or max_tokens
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.consumed_tokens = 0
        self._apply_lr()

    def step(self, consumed_tokens: int) -> None:
        self.consumed_tokens = int(consumed_tokens)
        self._apply_lr()

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> dict:
        return {"consumed_tokens": self.consumed_tokens, "base_lrs": list(self.base_lrs)}

    def load_state_dict(self, state: dict) -> None:
        self.consumed_tokens = int(state.get("consumed_tokens", 0))
        self.base_lrs = [float(x) for x in state.get("base_lrs", self.base_lrs)]
        self._apply_lr()

    def _apply_lr(self) -> None:
        scale = self._scale()
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def _scale(self) -> float:
        if self.cfg.warmup_tokens > 0 and self.consumed_tokens < self.cfg.warmup_tokens:
            return max(1.0e-8, self.consumed_tokens / self.cfg.warmup_tokens)
        denom = max(1, self.decay_tokens - self.cfg.warmup_tokens)
        progress = min(1.0, max(0.0, (self.consumed_tokens - self.cfg.warmup_tokens) / denom))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cfg.min_lr_ratio + (1.0 - self.cfg.min_lr_ratio) * cosine


class TokenWSDScheduler:
    """Warmup-Stable-Decay scheduler.

    Three phases driven by consumed_tokens (peak LR = base_lr from optimizer):
      1. Warmup [0, warmup_tokens):
         linear ramp 0 -> peak_lr
      2. Stable [warmup_tokens, stable_end):
         hold peak_lr
      3. Decay [stable_end, max_tokens]:
         cosine decay peak_lr -> peak_lr * min_lr_ratio

    `decay_tokens` controls the *length* of the decay phase, so
    stable_end = max_tokens - decay_tokens. If `stable_tokens` is also
    given it is treated as the length of the stable phase and used as a
    consistency check; otherwise it is derived.
    """

    def __init__(self, optimizer: Optimizer, cfg: SchedulerConfig, *, max_tokens: int) -> None:
        self.optimizer = optimizer
        self.cfg = cfg
        self.max_tokens = int(max_tokens)
        self.warmup_tokens = int(cfg.warmup_tokens)
        decay_len = int(cfg.decay_tokens) if cfg.decay_tokens is not None else 0
        if decay_len <= 0:
            raise ValueError("WSD scheduler requires positive decay_tokens (length of decay phase)")
        if self.warmup_tokens + decay_len > self.max_tokens:
            raise ValueError(
                f"warmup_tokens ({self.warmup_tokens}) + decay_tokens ({decay_len}) "
                f"exceeds max_tokens ({self.max_tokens})"
            )
        self.decay_len = decay_len
        self.stable_end = self.max_tokens - decay_len
        if cfg.stable_tokens is not None:
            expected = self.stable_end - self.warmup_tokens
            if int(cfg.stable_tokens) != expected:
                raise ValueError(
                    f"stable_tokens ({cfg.stable_tokens}) inconsistent with derived "
                    f"length ({expected}) = max_tokens - decay_tokens - warmup_tokens"
                )
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.consumed_tokens = 0
        self._apply_lr()

    def step(self, consumed_tokens: int) -> None:
        self.consumed_tokens = int(consumed_tokens)
        self._apply_lr()

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> dict:
        return {"consumed_tokens": self.consumed_tokens, "base_lrs": list(self.base_lrs)}

    def load_state_dict(self, state: dict) -> None:
        self.consumed_tokens = int(state.get("consumed_tokens", 0))
        self.base_lrs = [float(x) for x in state.get("base_lrs", self.base_lrs)]
        self._apply_lr()

    def _apply_lr(self) -> None:
        scale = self._scale()
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def _scale(self) -> float:
        t = self.consumed_tokens
        if self.warmup_tokens > 0 and t < self.warmup_tokens:
            return max(1.0e-8, t / self.warmup_tokens)
        if t < self.stable_end:
            return 1.0
        progress = min(1.0, max(0.0, (t - self.stable_end) / max(1, self.decay_len)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cfg.min_lr_ratio + (1.0 - self.cfg.min_lr_ratio) * cosine


def build_scheduler(optimizer: Optimizer, cfg: SchedulerConfig, *, max_tokens: int):
    if cfg.type == "cosine":
        return TokenCosineScheduler(optimizer, cfg, max_tokens=max_tokens)
    if cfg.type == "wsd":
        return TokenWSDScheduler(optimizer, cfg, max_tokens=max_tokens)
    raise ValueError(f"Unknown scheduler type: {cfg.type}")
