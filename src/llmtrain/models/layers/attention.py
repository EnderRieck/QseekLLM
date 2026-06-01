from __future__ import annotations

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from llmtrain.models.config import ModelConfig
from llmtrain.models.layers.rotary import RotaryEmbedding

KVCache = tuple[torch.Tensor, torch.Tensor]


class SelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.hidden_size % cfg.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if cfg.num_attention_heads % cfg.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attention_backend = cfg.attention_backend
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=cfg.max_position_embeddings,
            theta=cfg.rope_theta,
            backend=cfg.rotary_backend,
        )

    def forward(
        self,
        x: torch.Tensor,
        document_ids: torch.Tensor | None = None,
        *,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        past_len = 0 if past_key_value is None else int(past_key_value[0].shape[1])
        q, k = self.rotary(q, k, position_offset=past_len)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=1)
            v = torch.cat((past_v, v), dim=1)
        present = (k, v)

        y = self._flash_attention(q, k, v, document_ids=document_ids, past_len=past_len)
        if y is not None:
            out = self.o_proj(y)
            if use_cache:
                return out, present
            return out

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)
        attn_mask = _attention_mask(
            document_ids,
            query_len=seq_len,
            key_len=k.shape[1],
            past_len=past_len,
            device=q.device,
        )
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=attn_mask is None and past_len == 0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(y)
        if use_cache:
            return out, present
        return out

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        document_ids: torch.Tensor | None,
        past_len: int,
    ) -> torch.Tensor | None:
        if self.attention_backend == "sdpa":
            return None
        if q.device.type != "cuda":
            if self.attention_backend == "flash_attn":
                raise RuntimeError("model.attention_backend=flash_attn requires CUDA tensors")
            return None
        funcs = _flash_attn_functions()
        if funcs is None:
            if self.attention_backend == "flash_attn":
                raise RuntimeError("model.attention_backend=flash_attn requires flash-attn")
            return None
        flash_attn_func, flash_attn_varlen_func = funcs
        if document_ids is not None:
            if past_len != 0:
                raise ValueError("document_ids with KV cache is not supported")
            return _flash_packed_attention(q, k, v, document_ids, flash_attn_varlen_func)
        if past_len != 0:
            return None
        return flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(), dropout_p=0.0, causal=True).reshape(
            q.shape[0],
            q.shape[1],
            self.num_heads * self.head_dim,
        )


def _attention_mask(
    document_ids: torch.Tensor | None,
    *,
    query_len: int,
    key_len: int,
    past_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if document_ids is not None:
        if past_len != 0:
            raise ValueError("document_ids with KV cache is not supported")
        return _document_causal_mask(document_ids)
    if past_len == 0:
        return None
    query_positions = past_len + torch.arange(query_len, device=device)
    key_positions = torch.arange(key_len, device=device)
    causal = key_positions[None, :] <= query_positions[:, None]
    return causal[None, None, :, :]


def _document_causal_mask(document_ids: torch.Tensor) -> torch.Tensor:
    same_doc = document_ids[:, None, :, None] == document_ids[:, None, None, :]
    causal = torch.ones(
        (document_ids.shape[1], document_ids.shape[1]),
        dtype=torch.bool,
        device=document_ids.device,
    ).tril()
    return same_doc & causal[None, None, :, :]


@lru_cache(maxsize=1)
def _flash_attn_functions():
    try:
        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    except Exception:
        return None
    return flash_attn_func, flash_attn_varlen_func


def _flash_packed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, document_ids: torch.Tensor, flash_attn_varlen_func) -> torch.Tensor:
    batch, seq_len, _, _ = q.shape
    q_fa = q.contiguous().view(batch * seq_len, q.shape[2], q.shape[-1])
    k_fa = k.contiguous().view(batch * seq_len, k.shape[2], k.shape[-1])
    v_fa = v.contiguous().view(batch * seq_len, v.shape[2], v.shape[-1])
    cu_seqlens = _document_cu_seqlens(document_ids)
    y = flash_attn_varlen_func(
        q_fa,
        k_fa,
        v_fa,
        cu_seqlens,
        cu_seqlens,
        seq_len,
        seq_len,
        dropout_p=0.0,
        causal=True,
    )
    return y.view(batch, seq_len, q.shape[2], q.shape[-1]).reshape(batch, seq_len, q.shape[2] * q.shape[-1])


def _document_cu_seqlens(document_ids: torch.Tensor) -> torch.Tensor:
    batch, seq_len = document_ids.shape
    flat = document_ids.reshape(-1)
    starts = torch.empty(flat.shape, dtype=torch.bool, device=flat.device)
    starts[0] = True
    starts[1:] = flat[1:] != flat[:-1]
    starts[::seq_len] = True
    segment_starts = torch.nonzero(starts, as_tuple=False).flatten().to(torch.int32)
    total_tokens = torch.tensor([batch * seq_len], dtype=torch.int32, device=flat.device)
    ends = torch.cat((segment_starts[1:], total_tokens))
    lengths = ends - segment_starts
    cu_seqlens = torch.empty(lengths.numel() + 1, dtype=torch.int32, device=flat.device)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    return cu_seqlens
