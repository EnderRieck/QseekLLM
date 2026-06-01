from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y.to(dtype) * self.weight).to(dtype)


def build_rms_norm(hidden_size: int, eps: float, *, use_liger: bool = False) -> nn.Module:
    if not use_liger:
        return RMSNorm(hidden_size, eps)
    try:
        from liger_kernel.transformers import LigerRMSNorm
    except Exception as exc:
        raise RuntimeError("model.liger_rms_norm=true requires liger_kernel") from exc
    return LigerRMSNorm(hidden_size, eps=eps, casting_mode="llama", in_place=True)
