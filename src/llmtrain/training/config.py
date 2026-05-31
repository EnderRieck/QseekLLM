from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class OptimizerConfig(BaseModel):
    type: Literal["adamw"] = "adamw"
    lr: float = Field(3.0e-4, gt=0)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(0.1, ge=0)
    foreach: bool | None = None
    fused: bool | None = None

    model_config = {"extra": "forbid"}


class SchedulerConfig(BaseModel):
    type: Literal["cosine", "wsd"] = "cosine"
    warmup_tokens: int = Field(1_000_000_000, ge=0)
    decay_tokens: int | None = Field(None, gt=0)
    stable_tokens: int | None = Field(None, ge=0)
    min_lr_ratio: float = Field(0.1, ge=0, le=1)

    model_config = {"extra": "forbid"}


class TrainerConfig(BaseModel):
    micro_batch_size: PositiveInt = 1
    global_batch_size: PositiveInt = 1
    max_tokens: PositiveInt = 1_000_000
    max_steps: PositiveInt | None = None
    checkpoint_interval_steps: PositiveInt | None = None
    save_final_checkpoint: bool = True
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    grad_clip: float = Field(1.0, gt=0)

    model_config = {"extra": "forbid"}
