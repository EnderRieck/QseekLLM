from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from llmtrain.models.config import ModelConfig
from llmtrain.models.init_weights import init_weights
from llmtrain.models.layers.attention import KVCache
from llmtrain.models.layers.block import TransformerBlock
from llmtrain.models.layers.rmsnorm import build_rms_norm


@dataclass(frozen=True)
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[KVCache] | None = None


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = build_rms_norm(cfg.hidden_size, cfg.rms_norm_eps, use_liger=cfg.liger_rms_norm)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.fused_linear_cross_entropy = _build_fused_linear_cross_entropy() if cfg.fused_linear_cross_entropy else None
        self.use_activation_checkpointing = False
        self.activation_checkpointing_interval = 1
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(lambda module: init_weights(module, cfg))

    def set_activation_checkpointing(self, enabled: bool = True, *, interval: int = 1) -> None:
        self.use_activation_checkpointing = enabled
        self.activation_checkpointing_interval = max(1, int(interval))

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        document_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pad_token_id: int | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        if use_cache and document_ids is not None and past_key_values is not None:
            raise ValueError("document_ids with KV cache is not supported")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} KV cache entries, got {len(past_key_values)}")
        x = self.embed_tokens(input_ids)
        next_cache: list[KVCache] | None = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[idx]
            should_checkpoint = (
                self.use_activation_checkpointing
                and self.training
                and not use_cache
                and idx % self.activation_checkpointing_interval == 0
            )
            if should_checkpoint:
                x = checkpoint(layer, x, document_ids, use_reentrant=False)
            else:
                out = layer(x, document_ids=document_ids, past_key_value=past, use_cache=use_cache)
                if use_cache:
                    x, present = out
                    assert next_cache is not None
                    next_cache.append(present)
                else:
                    x = out
        targets = input_ids if labels is None else labels
        hidden = self.norm(x)
        can_use_fused_loss = (
            self.training
            and self.fused_linear_cross_entropy is not None
            and not use_cache
            and hidden.shape[1] > 1
        )
        loss = None
        if can_use_fused_loss:
            loss = self.fused_loss(hidden, targets, document_ids=document_ids, pad_token_id=pad_token_id)
            logits = hidden.new_empty((hidden.shape[0], hidden.shape[1], 0))
        else:
            logits = self.lm_head(hidden)
            loss = (
                self.loss(logits, targets, document_ids=document_ids, pad_token_id=pad_token_id)
                if not use_cache or logits.shape[1] > 1
                else None
            )
        return CausalLMOutput(logits=logits, loss=loss, past_key_values=next_cache)

    def fused_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        document_ids: torch.Tensor | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[1] < 2:
            return hidden_states.sum() * 0.0
        shift_hidden = hidden_states[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        if document_ids is not None:
            same_doc = document_ids[:, 1:] == document_ids[:, :-1]
            shift_labels = shift_labels.masked_fill(~same_doc, -100)
        if pad_token_id is not None:
            shift_labels = shift_labels.masked_fill(shift_labels == int(pad_token_id), -100)
        if not torch.any(shift_labels != -100):
            return hidden_states.sum() * 0.0
        if self.fused_linear_cross_entropy is None:
            raise RuntimeError("fused_linear_cross_entropy is not enabled")
        return self.fused_linear_cross_entropy(
            self.lm_head.weight,
            shift_hidden.view(-1, shift_hidden.size(-1)),
            shift_labels.view(-1),
        )

    @staticmethod
    def loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        document_ids: torch.Tensor | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        if logits.shape[1] < 2:
            return logits.sum() * 0.0
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        if document_ids is not None:
            same_doc = document_ids[:, 1:] == document_ids[:, :-1]
            shift_labels = shift_labels.masked_fill(~same_doc, -100)
        if pad_token_id is not None:
            shift_labels = shift_labels.masked_fill(shift_labels == int(pad_token_id), -100)
        if not torch.any(shift_labels != -100):
            return logits.sum() * 0.0
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )


def _build_fused_linear_cross_entropy() -> nn.Module | None:
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except Exception as exc:
        raise RuntimeError("model.fused_linear_cross_entropy=true requires liger_kernel") from exc
    return LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
