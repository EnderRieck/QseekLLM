from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, Field, PositiveInt

from llmtrain.utils.config import sha256_file


class ShardInfo(BaseModel):
    id: str
    uri: str
    source: str
    domain: str
    language: str
    format: Literal["jsonl", "parquet"]
    num_records: int = Field(ge=0)
    bytes: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    sha256: str
    weight: float = Field(gt=0)
    license: str = "unknown"
    created_at: str
    record_start: int = Field(default=0, ge=0)
    record_end: int | None = Field(default=None)

    model_config = {"extra": "forbid"}

    @property
    def effective_record_end(self) -> int:
        return self.num_records if self.record_end is None else self.record_end

    @property
    def slice_record_count(self) -> int:
        return max(0, self.effective_record_end - self.record_start)


class ManifestMeta(BaseModel):
    manifest_version: str
    created_at: str
    manifest_sha256: str
    num_shards: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    total_records: int = Field(ge=0)
    total_estimated_tokens: int = Field(ge=0)

    model_config = {"extra": "forbid"}


@dataclass(frozen=True)
class ManifestPaths:
    manifest: Path
    meta: Path


def infer_format(path: Path) -> Literal["jsonl", "parquet"]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return "jsonl"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(f"Unsupported shard extension: {path}")


def estimate_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4)


def _inspect_jsonl(path: Path) -> tuple[int, int]:
    records = 0
    tokens = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            records += 1
            tokens += estimate_tokens(str(obj.get("text", "")))
    return records, tokens


def _inspect_parquet(path: Path) -> tuple[int, int]:
    pf = pq.ParquetFile(path)
    records = pf.metadata.num_rows
    tokens = 0
    for batch in pf.iter_batches(columns=["text"], batch_size=4096):
        for text in batch.column(0).to_pylist():
            tokens += estimate_tokens(str(text or ""))
    return records, tokens


def inspect_shard(
    path: Path,
    *,
    source: str,
    domain: str,
    language: str,
    weight: float = 1.0,
    license: str = "unknown",
) -> ShardInfo:
    fmt = infer_format(path)
    num_records, estimated = _inspect_jsonl(path) if fmt == "jsonl" else _inspect_parquet(path)
    return ShardInfo(
        id=path.stem,
        uri=str(path.resolve()),
        source=source,
        domain=domain,
        language=language,
        format=fmt,
        num_records=num_records,
        bytes=path.stat().st_size,
        estimated_tokens=estimated,
        sha256=sha256_file(path),
        weight=weight,
        license=license,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def write_manifest(
    shards: Iterable[ShardInfo],
    output_dir: str | Path,
    *,
    manifest_version: str = "0.1.0",
) -> ManifestPaths:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    meta_path = out / "manifest.meta.json"
    shard_list = list(shards)
    with manifest_path.open("w", encoding="utf-8") as f:
        for shard in shard_list:
            f.write(json.dumps(shard.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")
    meta = ManifestMeta(
        manifest_version=manifest_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha256_file(manifest_path),
        num_shards=len(shard_list),
        total_bytes=sum(s.bytes for s in shard_list),
        total_records=sum(s.num_records for s in shard_list),
        total_estimated_tokens=sum(s.estimated_tokens for s in shard_list),
    )
    meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return ManifestPaths(manifest=manifest_path, meta=meta_path)


def merge_manifests(
    manifest_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    manifest_version: str = "0.1.0",
    validate_inputs: bool = True,
) -> ManifestPaths:
    merged: list[ShardInfo] = []
    seen_keys: set[tuple[str, int, int | None]] = set()
    for manifest_path in manifest_paths:
        if validate_inputs:
            validate_manifest(manifest_path, validate_shards=True)
        for shard in load_manifest(manifest_path):
            key = (shard.uri, shard.record_start, shard.record_end)
            if key in seen_keys:
                raise ValueError(
                    f"Duplicate (uri, record_range) while merging manifests: {shard.uri} [{shard.record_start},{shard.record_end}]"
                )
            seen_keys.add(key)
            merged.append(shard)
    return write_manifest(merged, output_dir, manifest_version=manifest_version)


def load_manifest(path: str | Path) -> list[ShardInfo]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        return [ShardInfo.model_validate_json(line) for line in f if line.strip()]


def manifest_meta_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name("manifest.meta.json")


def validate_manifest(path: str | Path, *, validate_shards: bool = True) -> ManifestMeta:
    manifest_path = Path(path)
    meta = ManifestMeta.model_validate_json(manifest_meta_path(manifest_path).read_text(encoding="utf-8"))
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != meta.manifest_sha256:
        raise ValueError(
            f"Manifest hash mismatch: expected {meta.manifest_sha256}, got {actual_manifest_hash}"
        )
    shards = load_manifest(manifest_path)
    if len(shards) != meta.num_shards:
        raise ValueError(f"Manifest shard count mismatch: expected {meta.num_shards}, got {len(shards)}")
    if validate_shards:
        for shard in shards:
            actual = sha256_file(Path(shard.uri))
            if actual != shard.sha256:
                raise ValueError(f"Shard hash mismatch for {shard.id}: expected {shard.sha256}, got {actual}")
    return meta


def deterministic_assignment(shard: ShardInfo, world_size: int, num_workers: int) -> int:
    return int(shard.sha256, 16) % (world_size * num_workers)


def assigned_shards(shards: Iterable[ShardInfo], world_size: int, rank: int, num_workers: int, worker_id: int) -> list[ShardInfo]:
    target = rank * num_workers + worker_id
    return [s for s in shards if deterministic_assignment(s, world_size, num_workers) == target]


def discover_shards(inputs: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if any(ch in str(p) for ch in "*?[]"):
            paths.extend(sorted(Path().glob(str(p))))
        elif p.is_dir():
            paths.extend(sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in {".jsonl", ".json", ".parquet", ".pq"}))
        else:
            paths.append(p)
    return paths
