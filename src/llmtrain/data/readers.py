from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from llmtrain.data.manifest import ManifestMeta, ShardInfo, assigned_shards, load_manifest, validate_manifest
from llmtrain.data.schemas import validate_record
from llmtrain.interfaces import Record


class IncompatibleDataState(RuntimeError):
    pass


class ShardReader:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        world_size: int = 1,
        rank: int = 0,
        num_workers: int = 1,
        worker_id: int = 0,
        validate_hashes: bool = True,
        shuffle_seed: int | None = None,
        parquet_batch_size: int = 8192,
        source_filter: str | None = None,
        domain_filter: str | None = None,
        manifest_meta: ManifestMeta | None = None,
        exclude_uris: set[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.meta = manifest_meta or validate_manifest(self.manifest_path, validate_shards=validate_hashes)
        all_shards = load_manifest(self.manifest_path)
        if source_filter is not None:
            all_shards = [shard for shard in all_shards if shard.source == source_filter]
        if domain_filter is not None:
            all_shards = [shard for shard in all_shards if shard.domain == domain_filter]
        # Auto-dedup vs a warm-start source: drop shards a previous run consumed.
        # Keyed on uri (the unique shard identifier; shard id is not unique).
        if exclude_uris:
            all_shards = [shard for shard in all_shards if shard.uri not in exclude_uris]
        self.shards = assigned_shards(all_shards, world_size, rank, num_workers, worker_id)
        self.source_filter = source_filter
        self.domain_filter = domain_filter
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.shards)
        self.world_size = world_size
        self.rank = rank
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.parquet_batch_size = parquet_batch_size
        self.shard_index = 0
        self.shard_byte_offset = 0
        self.shard_record_offset = 0

    @property
    def current_shard_id(self) -> str:
        if self.shard_index >= len(self.shards):
            return ""
        return self.shards[self.shard_index].id

    def __iter__(self) -> Iterator[Record]:
        while self.shard_index < len(self.shards):
            shard = self.shards[self.shard_index]
            if shard.format == "jsonl":
                yield from self._iter_jsonl(shard)
            else:
                yield from self._iter_parquet(shard)
            self.shard_index += 1
            self.shard_byte_offset = 0
            self.shard_record_offset = 0

    def _iter_jsonl(self, shard: ShardInfo) -> Iterator[Record]:
        slice_start = shard.record_start
        slice_end = shard.effective_record_end
        with Path(shard.uri).open("r", encoding="utf-8") as f:
            if self.shard_byte_offset:
                f.seek(self.shard_byte_offset)
            absolute_idx = slice_start + self.shard_record_offset
            # Walk records until absolute_idx; if we've started fresh (shard_byte_offset=0)
            # we need to skip the first slice_start records.
            if self.shard_byte_offset == 0 and slice_start > 0:
                skipped = 0
                while skipped < slice_start:
                    line = f.readline()
                    if not line:
                        return
                    if line.strip():
                        skipped += 1
            while True:
                if absolute_idx >= slice_end:
                    return
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    self.shard_byte_offset = f.tell()
                    continue
                data = json.loads(line)
                record = validate_record(data)
                self.shard_byte_offset = f.tell()
                self.shard_record_offset += 1
                absolute_idx += 1
                yield record

    def _iter_parquet(self, shard: ShardInfo) -> Iterator[Record]:
        slice_start = shard.record_start
        slice_end = shard.effective_record_end
        seen = 0
        pf = pq.ParquetFile(shard.uri)
        columns = ["id", "text", "source", "domain", "language", "metadata"]
        for batch in pf.iter_batches(columns=columns, batch_size=self.parquet_batch_size):
            rows = batch.to_pylist()
            for row in rows:
                if seen < slice_start:
                    seen += 1
                    continue
                if seen >= slice_end:
                    return
                if seen - slice_start < self.shard_record_offset:
                    seen += 1
                    continue
                if row.get("metadata") is None:
                    row["metadata"] = {}
                elif isinstance(row.get("metadata"), str):
                    row["metadata"] = json.loads(row["metadata"]) if row["metadata"] else {}
                self.shard_record_offset += 1
                seen += 1
                yield validate_record(row)

    def state_dict(self) -> dict[str, Any]:
        return {
            "manifest_hash": self.meta.manifest_sha256,
            "world_size": self.world_size,
            "rank": self.rank,
            "num_workers": self.num_workers,
            "worker_id": self.worker_id,
            "parquet_batch_size": self.parquet_batch_size,
            "source_filter": self.source_filter,
            "domain_filter": self.domain_filter,
            "current_shard_id": self.current_shard_id,
            "shard_index": self.shard_index,
            "shard_byte_offset": self.shard_byte_offset,
            "consumed_records": self.shard_record_offset,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        for key, value in {
            "manifest_hash": self.meta.manifest_sha256,
            "world_size": self.world_size,
            "rank": self.rank,
            "num_workers": self.num_workers,
            "worker_id": self.worker_id,
            "source_filter": self.source_filter,
            "domain_filter": self.domain_filter,
        }.items():
            if sd.get(key) != value:
                raise IncompatibleDataState(f"{key} mismatch: expected {value}, got {sd.get(key)}")
        current_shard_id = sd.get("current_shard_id")
        if current_shard_id:
            shard_index = int(sd["shard_index"])
            if shard_index >= len(self.shards) or self.shards[shard_index].id != current_shard_id:
                raise IncompatibleDataState(
                    f"current_shard_id mismatch: expected {self.shards[shard_index].id if shard_index < len(self.shards) else ''}, "
                    f"got {current_shard_id}"
                )
        self.shard_index = int(sd["shard_index"])
        self.shard_byte_offset = int(sd["shard_byte_offset"])
        self.shard_record_offset = int(sd["consumed_records"])
