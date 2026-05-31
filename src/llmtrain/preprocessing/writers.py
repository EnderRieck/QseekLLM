from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from llmtrain.data.manifest import ShardInfo, inspect_shard
from llmtrain.data.schemas import record_to_dict
from llmtrain.interfaces import Record
from llmtrain.preprocessing.config import PreprocessWriterConfig


class RollingShardWriter:
    def __init__(
        self,
        cfg: PreprocessWriterConfig,
        *,
        source: str,
        domain: str,
        language: str,
        weight: float,
        license: str,
        start_index: int = 0,
        shard_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.source = source
        self.domain = domain
        self.language = language
        self.weight = weight
        self.license = license
        self.shard_dir = shard_dir or (cfg.output_dir / "shards")
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.shard_index = start_index
        self.current_bytes = 0
        self.current_records = 0
        self._jsonl_file = None
        self._parquet_rows: list[dict[str, Any]] = []
        self.shards: list[ShardInfo] = []
        self._open_next()

    def write(self, record: Record) -> None:
        payload = record_to_dict(record)
        if self.cfg.output_format == "jsonl":
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            if self.current_records and self.current_bytes + len(encoded.encode("utf-8")) > self.cfg.shard_max_bytes:
                self.rotate()
            assert self._jsonl_file is not None
            self._jsonl_file.write(encoded)
            self.current_bytes += len(encoded.encode("utf-8"))
            self.current_records += 1
        else:
            self._parquet_rows.append(payload)
            self.current_bytes += len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            self.current_records += 1
            if self.current_bytes >= self.cfg.shard_max_bytes:
                self.rotate()

    def rotate(self) -> None:
        if self.current_records == 0:
            return
        path = self._current_path()
        if self.cfg.output_format == "jsonl":
            assert self._jsonl_file is not None
            self._jsonl_file.close()
            self._jsonl_file = None
        else:
            pq.write_table(pa.Table.from_pylist(self._parquet_rows), path)
            self._parquet_rows = []
        self.shards.append(
            inspect_shard(
                path,
                source=self.source,
                domain=self.domain,
                language=self.language,
                weight=self.weight,
                license=self.license,
            )
        )
        self.shard_index += 1
        self.current_bytes = 0
        self.current_records = 0
        self._open_next()

    def close(self) -> list[ShardInfo]:
        self.rotate()
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        current = self._current_path()
        if self.current_records == 0 and current.exists() and current.stat().st_size == 0:
            current.unlink()
        return list(self.shards)

    def _open_next(self) -> None:
        if self.cfg.output_format == "jsonl":
            self._jsonl_file = self._current_path().open("w", encoding="utf-8")

    def _current_path(self) -> Path:
        suffix = "jsonl" if self.cfg.output_format == "jsonl" else "parquet"
        return self.shard_dir / f"{self.cfg.shard_prefix}_{self.source}_{self.shard_index:06d}.{suffix}"


class RejectedWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, doc_id: str, reason: str, metadata: dict[str, Any] | None = None) -> None:
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": doc_id, "reason": reason, "metadata": metadata or {}}, ensure_ascii=False) + "\n")
