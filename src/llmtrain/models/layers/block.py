from __future__ import annotations

import torch
from torch import nn

from llmtrain.models.config import ModelConfig
from llmtrain.models.layers.attention import KVCache, SelfAttention
from llmtrain.models.layers.mlp import build_swiglu
from llmtrain.models.layers.rmsnorm import build_rms_norm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = build_rms_norm(cfg.hidden_size, cfg.rms_norm_eps, use_liger=cfg.liger_rms_norm)
        self.self_attn = SelfAttention(cfg)
        self.post_attention_layernorm = build_rms_norm(cfg.hidden_size, cfg.rms_norm_eps, use_liger=cfg.liger_rms_norm)
        self.mlp = build_swiglu(cfg)

    def forward(
        self,
        x: torch.Tensor,
        document_ids: torch.Tensor | None = None,
        *,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        attn_out = self.self_attn(
            self.input_layernorm(x),
            document_ids=document_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        present = None
        if use_cache:
            attn_out, present = attn_out
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        if use_cache:
            assert present is not None
            return x, present
        return x
