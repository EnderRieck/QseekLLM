from __future__ import annotations

import torch
from torch import nn

from llmtrain.training.config import OptimizerConfig


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    if cfg.type != "adamw":
        raise ValueError(f"Unsupported optimizer: {cfg.type}")
    if cfg.foreach and cfg.fused:
        raise ValueError("AdamW foreach and fused modes are mutually exclusive")
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
        foreach=cfg.foreach,
        fused=cfg.fused,
    )
