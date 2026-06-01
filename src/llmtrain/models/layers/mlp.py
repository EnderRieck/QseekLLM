from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from llmtrain.models.config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def build_swiglu(cfg: ModelConfig) -> nn.Module:
    if not cfg.liger_swiglu:
        return SwiGLU(cfg.hidden_size, cfg.intermediate_size)
    try:
        from liger_kernel.transformers import LigerSwiGLUMLP
    except Exception as exc:
        raise RuntimeError("model.liger_swiglu=true requires liger_kernel") from exc
    return LigerSwiGLUMLP(cfg)
