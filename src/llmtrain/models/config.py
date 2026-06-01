from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class ModelConfig(BaseModel):
    model_type: Literal["qwen_like"] = "qwen_like"
    vocab_size: PositiveInt = 150_000
    hidden_size: PositiveInt = 2048
    intermediate_size: PositiveInt = 11008
    num_hidden_layers: PositiveInt = 24
    num_attention_heads: PositiveInt = 16
    num_key_value_heads: PositiveInt = 8
    max_position_embeddings: PositiveInt = 4096
    rope_theta: float = Field(1_000_000.0, gt=0)
    rms_norm_eps: float = Field(1.0e-6, gt=0)
    hidden_act: Literal["silu", "swish"] = "silu"
    attention_backend: Literal["auto", "sdpa", "flash_attn"] = "auto"
    rotary_backend: Literal["auto", "torch", "triton"] = "auto"
    tie_word_embeddings: bool = False
    fused_linear_cross_entropy: bool = False
    liger_rms_norm: bool = False
    liger_swiglu: bool = False

    model_config = {"extra": "forbid"}
