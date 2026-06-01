from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class DataSourceConfig(BaseModel):
    name: str
    domain: str
    language: str = "multi"
    source_filter: str | None = None
    weight: float = Field(1.0, ge=0.0)
    paths: list[Path] = Field(default_factory=list)
    target_ratio: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class ReaderConfig(BaseModel):
    world_size: PositiveInt = 1
    rank: int = Field(0, ge=0)
    num_workers: PositiveInt = 1
    worker_id: int = Field(0, ge=0)
    quarantine_threshold: PositiveInt = 5
    parquet_batch_size: PositiveInt = 8192

    model_config = {"extra": "forbid"}


class MixerConfig(BaseModel):
    temperature: float = Field(1.0, gt=0.0)
    seed: int = 42

    model_config = {"extra": "forbid"}


class QualityFilterConfig(BaseModel):
    enabled: bool = False
    ccnet_model_path: str | None = None
    min_length: int = 50
    max_length: int = 100000
    max_perplexity: float = 1000.0
    min_quality_score: float = 0.3

    model_config = {"extra": "forbid"}


class PipelineConfig(BaseModel):
    min_chars: int = Field(1, ge=0)
    max_chars: int | None = Field(default=None, ge=1)
    min_quality_score: float | None = Field(default=None, ge=0.0)
    normalize_whitespace: bool = True

    model_config = {"extra": "forbid"}


class PackingConfig(BaseModel):
    seq_len: PositiveInt = 4096
    batch_size: PositiveInt = 1
    prefetch_records: PositiveInt = 8192
    async_tokenization: bool = False
    producer_workers: PositiveInt = 2
    queue_max_batches: PositiveInt = 32

    model_config = {"extra": "forbid"}


class TokenizerSamplingConfig(BaseModel):
    byte_budget: PositiveInt = 200_000_000_000
    output_dir: Path | None = None
    output_shard_bytes: PositiveInt = 1_073_741_824
    num_workers: PositiveInt = 1
    ratios: dict[str, float] = Field(default_factory=dict)
    seed: int = 42

    model_config = {"extra": "forbid"}


class DataConfig(BaseModel):
    manifest_path: Path
    validate_hashes: bool = True
    sources: list[DataSourceConfig] = Field(default_factory=list)
    reader: ReaderConfig = Field(default_factory=ReaderConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    mixer: MixerConfig = Field(default_factory=MixerConfig)
    packing: PackingConfig = Field(default_factory=PackingConfig)
    tokenizer_sampling: TokenizerSamplingConfig = Field(default_factory=TokenizerSamplingConfig)
    allowed_formats: list[Literal["jsonl", "parquet"]] = Field(default_factory=lambda: ["jsonl", "parquet"])

    model_config = {"extra": "forbid"}
