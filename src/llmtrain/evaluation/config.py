from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PositiveInt

from llmtrain.inference.config import RuntimeConfig
from llmtrain.utils.config import apply_overrides


class EvalRunConfig(BaseModel):
    name: str
    output_dir: Path = Path("runs/eval")
    seed: int = 42

    model_config = {"extra": "forbid"}


class EvalModelConfig(BaseModel):
    train_config: Path
    checkpoint: str | Path = "latest"

    model_config = {"extra": "forbid"}


class EvalDatasetConfig(BaseModel):
    root_dir: Path = Path("eval/datasets")
    wplc_prepared_path: Path = Path("eval/datasets/chinese_wplc/dev.jsonl")
    offline: bool = True

    model_config = {"extra": "forbid"}


class EvalHarnessConfig(BaseModel):
    include_path: Path = Path("eval/tasks")
    batch_size: PositiveInt | str = 1
    max_batch_size: PositiveInt | None = None
    limit: PositiveInt | float | None = None
    num_fewshot: int = Field(0, ge=0)
    bootstrap_iters: int = Field(0, ge=0)
    log_samples: bool = True
    cache_requests: bool = True
    use_cache: Path | None = None
    verbosity: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    model_config = {"extra": "forbid"}


class EvalConfig(BaseModel):
    run: EvalRunConfig
    model: EvalModelConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    datasets: EvalDatasetConfig = Field(default_factory=EvalDatasetConfig)
    harness: EvalHarnessConfig = Field(default_factory=EvalHarnessConfig)
    tasks: list[str] = Field(default_factory=lambda: ["chinese_wplc", "lambada_openai"])

    model_config = {"extra": "forbid"}


def load_eval_config(path: str | Path, overrides: list[str] | None = None) -> EvalConfig:
    p = Path(path).expanduser()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Eval config must be a mapping: {p}")
    return EvalConfig.model_validate(apply_overrides(raw, overrides))
