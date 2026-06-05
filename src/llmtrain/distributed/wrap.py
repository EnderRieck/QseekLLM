from __future__ import annotations

from functools import partial

import torch
from torch import nn

from llmtrain.distributed.config import DistributedConfig
from llmtrain.distributed.env import DistributedContext
from llmtrain.models.layers.block import TransformerBlock


def configure_model_for_training(model: nn.Module, cfg: DistributedConfig) -> nn.Module:
    if hasattr(model, "set_activation_checkpointing"):
        model.set_activation_checkpointing(
            cfg.activation_checkpointing,
            interval=cfg.activation_checkpointing_interval,
        )
    return model


def wrap_model(model: nn.Module, cfg: DistributedConfig, ctx: DistributedContext) -> nn.Module:
    model = configure_model_for_training(model, cfg)
    if not ctx.enabled or cfg.backend == "none":
        return _maybe_compile(model, cfg)
    if cfg.backend == "ddp":
        return _wrap_ddp(_maybe_compile(model, cfg), cfg, ctx)
    if cfg.backend == "fsdp":
        _compile_transformer_blocks(model, cfg)
        return _wrap_fsdp(model, cfg)
    if cfg.backend == "zero":
        raise NotImplementedError("ZeRO is reserved for a later Phase 3 backend; use fsdp or ddp for now.")
    raise ValueError(f"Unsupported distributed backend: {cfg.backend}")


def unwrap_model(model: nn.Module) -> nn.Module:
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        if hasattr(model, "module"):
            model = model.module
            continue
        return model


def _wrap_ddp(model: nn.Module, cfg: DistributedConfig, ctx: DistributedContext) -> nn.Module:
    device_ids = [ctx.local_rank] if ctx.device.type == "cuda" else None
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=ctx.local_rank if ctx.device.type == "cuda" else None,
        find_unused_parameters=cfg.ddp_find_unused_parameters,
        init_sync=cfg.ddp_init_sync,
    )


def _wrap_fsdp(model: nn.Module, cfg: DistributedConfig) -> nn.Module:
    try:
        from torch.distributed.fsdp import BackwardPrefetch, CPUOffload, FullyShardedDataParallel, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    except Exception as exc:  # pragma: no cover - torch build dependent
        raise RuntimeError("FSDP is not available in this torch build") from exc

    strategy = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[cfg.fsdp_sharding_strategy]
    mixed_precision = None
    if cfg.fsdp_mixed_precision and torch.cuda.is_available():
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    auto_wrap_policy = None
    if cfg.fsdp_auto_wrap_policy == "transformer_block":
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    backward_prefetch = {
        "pre": BackwardPrefetch.BACKWARD_PRE,
        "post": BackwardPrefetch.BACKWARD_POST,
        "none": None,
    }[cfg.fsdp_backward_prefetch]
    return FullyShardedDataParallel(
        model,
        sharding_strategy=strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=cfg.fsdp_cpu_offload),
        use_orig_params=cfg.fsdp_use_orig_params,
        forward_prefetch=cfg.fsdp_forward_prefetch,
        backward_prefetch=backward_prefetch,
    )


def _maybe_compile(model: nn.Module, cfg: DistributedConfig) -> nn.Module:
    if not cfg.compile_model:
        return model
    mode = None if cfg.compile_mode == "default" else cfg.compile_mode
    return torch.compile(model, mode=mode)  # type: ignore[return-value]


def _compile_transformer_blocks(model: nn.Module, cfg: DistributedConfig) -> None:
    if not cfg.compile_model:
        return
    mode = None if cfg.compile_mode == "default" else cfg.compile_mode
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            module.forward = torch.compile(module.forward, mode=mode)  # type: ignore[method-assign]
