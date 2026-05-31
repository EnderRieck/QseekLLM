from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class CheckpointConfig(BaseModel):
    format: Literal["auto", "torch", "dcp"] = "auto"
    save_interval_minutes: PositiveInt = 30
    milestone_interval_tokens: PositiveInt = 1_000_000_000
    keep_latest: PositiveInt = 3
    save_best: bool = True
    monitor_metric: str = "loss"
    mode: str = "min"
    schema_version: str = "1.0"

    model_config = {"extra": "forbid"}
