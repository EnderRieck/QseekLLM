from __future__ import annotations

from pydantic import BaseModel, Field, PositiveInt


class ObservabilityConfig(BaseModel):
    metrics_jsonl: bool = True
    events_jsonl: bool = True
    data_metrics_jsonl: bool = True
    heartbeat: bool = True
    console_interval_steps: PositiveInt = 10
    heartbeat_interval_seconds: PositiveInt = 60
    data_metrics_interval_seconds: PositiveInt = 30
    tensorboard: bool = False
    wandb: bool = False
    collect_histograms: bool = False
    collect_tokenizer_stats: bool = False
    loss_spike_window: PositiveInt = 10
    loss_spike_sigma: float = Field(5.0, gt=0)

    model_config = {"extra": "forbid"}
