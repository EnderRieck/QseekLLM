from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class DistributedConfig(BaseModel):
    backend: Literal["none", "ddp", "fsdp", "zero"] = "none"
    activation_checkpointing: bool = False
    activation_checkpointing_interval: PositiveInt = Field(
        1,
        description="Checkpoint every Nth transformer block when activation_checkpointing is enabled.",
    )
    gradient_accumulation_no_sync: bool = True
    compile_model: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    init_method: str = "env://"
    timeout_seconds: PositiveInt = 1800
    fsdp_sharding_strategy: Literal["full_shard", "shard_grad_op", "no_shard"] = "full_shard"
    fsdp_auto_wrap_policy: Literal["none", "transformer_block"] = "transformer_block"
    fsdp_mixed_precision: bool = True
    fsdp_cpu_offload: bool = False
    fsdp_use_orig_params: bool = False
    fsdp_forward_prefetch: bool = True
    fsdp_backward_prefetch: Literal["pre", "post", "none"] = "pre"
    ddp_find_unused_parameters: bool = False
    profile_memory: bool = True
    profile_interval_steps: PositiveInt = Field(10, description="Collect heavier profiler signals every N steps.")

    model_config = {"extra": "forbid"}
