from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt, model_validator


class ValidationConfig(BaseModel):
    """Per-N-tokens in-training evaluation against a held-out val manifest."""

    enabled: bool = False
    val_manifest: Path | None = None
    interval_tokens: PositiveInt = 5_000_000_000
    max_tokens_per_source: PositiveInt = 6_000_000
    seq_len: PositiveInt = 4096
    batch_size: PositiveInt = 4
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    run_at_start: bool = False

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _require_manifest_when_enabled(self) -> "ValidationConfig":
        if self.enabled and self.val_manifest is None:
            raise ValueError("validation.val_manifest must be set when validation.enabled=true")
        return self
