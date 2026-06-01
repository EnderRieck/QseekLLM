from __future__ import annotations

import math

from torch import nn

from llmtrain.models.config import ModelConfig


def init_weights(module: nn.Module, cfg: ModelConfig) -> None:
    std = 0.02 / math.sqrt(max(1, cfg.num_hidden_layers))
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
