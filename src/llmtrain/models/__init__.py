from __future__ import annotations

from llmtrain.models.config import ModelConfig
from llmtrain.models.decoder import TransformerLM


def build_model(cfg: ModelConfig) -> TransformerLM:
    if cfg.model_type != "qwen_like":
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")
    return TransformerLM(cfg)


__all__ = ["TransformerLM", "build_model"]
