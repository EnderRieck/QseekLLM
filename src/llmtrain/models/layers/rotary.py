from __future__ import annotations

from functools import lru_cache

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, max_position_embeddings: int, theta: float, backend: str = "auto") -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings
        self.backend = backend

    def _cos_sin_half(self, seq_len: int, device: torch.device, *, position_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(position_offset, position_offset + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        return freqs.cos(), freqs.sin()

    def _cos_sin(self, seq_len: int, device: torch.device, *, position_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        cos_half, sin_half = self._cos_sin_half(seq_len, device, position_offset=position_offset)
        emb = torch.cat((cos_half, cos_half), dim=-1)
        sin = torch.cat((sin_half, sin_half), dim=-1)
        return emb[None, :, None, :], sin[None, :, None, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor, *, position_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        triton_out = self._triton_rotary(q, k, position_offset=position_offset)
        if triton_out is not None:
            return triton_out
        cos, sin = self._cos_sin(q.shape[1], q.device, position_offset=position_offset)
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

    def _triton_rotary(self, q: torch.Tensor, k: torch.Tensor, *, position_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.backend == "torch":
            return None
        if q.device.type != "cuda":
            if self.backend == "triton":
                raise RuntimeError("model.rotary_backend=triton requires CUDA tensors")
            return None
        apply_rotary_emb_func = _flash_rotary_function()
        if apply_rotary_emb_func is None:
            if self.backend == "triton":
                raise RuntimeError("model.rotary_backend=triton requires flash-attn rotary kernels")
            return None
        cos, sin = self._cos_sin_half(q.shape[1], q.device, position_offset=position_offset)
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        return (
            apply_rotary_emb_func(q, cos, sin, interleaved=False, inplace=False),
            apply_rotary_emb_func(k, cos, sin, interleaved=False, inplace=False),
        )


@lru_cache(maxsize=1)
def _flash_rotary_function():
    try:
        from flash_attn.layers.rotary import apply_rotary_emb_func
    except Exception:
        return None
    return apply_rotary_emb_func
