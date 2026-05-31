from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt, model_validator


class PreprocessSourceConfig(BaseModel):
    name: str
    enabled: bool = True
    type: Literal["jsonl", "parquet", "remote_jsonl", "remote_parquet", "hf_dataset", "html", "text", "wiki_xml", "git", "pdf"]
    domain: str
    language: str = "multi"
    license: str = "unknown"
    paths: list[Path] = Field(default_factory=list)
    weight: float = Field(1.0, gt=0)
    text_field: str = "text"
    id_field: str | None = "id"
    metadata_fields: list[str] = Field(default_factory=list)
    exclude_metadata_values: dict[str, list[str]] = Field(default_factory=dict)
    urls: list[str] = Field(default_factory=list)
    url_list_path: Path | None = None
    url_list_url: str | None = None
    hf_repo_id: str | None = None
    hf_revision: str = "main"
    hf_include_patterns: list[str] = Field(default_factory=list)
    hf_name: str | None = None
    hf_config: str | None = None
    hf_split: str = "train"
    hf_streaming: bool = True
    include_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".java",
            ".go",
            ".rs",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".md",
            ".txt",
        ]
    )
    exclude_dirs: list[str] = Field(
        default_factory=lambda: [".git", "node_modules", "vendor", "dist", "build", "__pycache__"]
    )
    max_file_bytes: PositiveInt = 1_000_000
    limit: PositiveInt | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_source_locator(self) -> "PreprocessSourceConfig":
        if self.type in {"remote_jsonl", "remote_parquet"} and not (
            self.urls or self.url_list_path or self.url_list_url or self.hf_repo_id
        ):
            raise ValueError(f"{self.type} source requires one of: urls, url_list_path, url_list_url, hf_repo_id")
        if self.type == "hf_dataset" and not self.hf_name:
            raise ValueError("hf_dataset source requires hf_name")
        if self.type in {"jsonl", "parquet", "html", "text", "wiki_xml", "git", "pdf"} and not self.paths:
            raise ValueError(f"{self.type} source requires paths")
        return self


class CleaningConfig(BaseModel):
    min_chars: int = Field(50, ge=0)
    max_chars: int | None = Field(default=200_000, ge=1)
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    remove_boilerplate: bool = True
    min_alpha_or_cjk_ratio: float = Field(0.15, ge=0.0, le=1.0)
    max_repeated_line_ratio: float = Field(0.35, ge=0.0, le=1.0)
    max_control_char_ratio: float = Field(0.02, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class DedupConfig(BaseModel):
    exact: bool = True
    simhash: bool = True
    simhash_bits: PositiveInt = 64
    simhash_threshold: int = Field(3, ge=0)
    index_workers: PositiveInt = 32
    exact_shards: PositiveInt = 256
    simhash_shards: PositiveInt = 256
    simhash_max_group_size: PositiveInt = 4096
    state_dir: Path | None = None
    persist_state: bool = True
    load_existing_state: bool = False

    model_config = {"extra": "forbid"}


class QualityConfig(BaseModel):
    scorer: Literal["heuristic"] = "heuristic"
    min_score: float = Field(0.45, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class PreprocessWriterConfig(BaseModel):
    output_dir: Path
    output_format: Literal["jsonl", "parquet"] = "jsonl"
    shard_prefix: str = "shard"
    shard_max_bytes: PositiveInt = 128 * 1024 * 1024
    rejected_path: Path | None = None

    model_config = {"extra": "forbid"}


class PreprocessConfig(BaseModel):
    sources: list[PreprocessSourceConfig]
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    writer: PreprocessWriterConfig
    checkpoint_interval_records: PositiveInt = 1000
    num_workers: int = Field(0, ge=0)
    worker_chunk_files: PositiveInt = 1
    candidate_dir: Path | None = None
    keep_candidates: bool = False
    cleanup_dedup_work: bool = True
    progress_interval_records: PositiveInt = 1000
    progress_interval_seconds: float = Field(5.0, gt=0)

    model_config = {"extra": "forbid"}
