#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

import _bootstrap  # noqa: F401


DEFAULT_LANGUAGES = ["Python", "Java", "JavaScript", "TypeScript", "C", "C++", "Go", "Rust", "Shell", "SQL"]
DEFAULT_COLUMNS = [
    "blob_id",
    "content_id",
    "repo_name",
    "path",
    "detected_licenses",
    "license_type",
    "src_encoding",
    "language",
    "extension",
    "length_bytes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve The Stack v2 SWH blob metadata into local JSONL source-code shards.")
    parser.add_argument("--output-dir", default="/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/datasets/the_stack_v2_resolved/jsonl")
    parser.add_argument("--metadata-dir", default="/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/datasets/the_stack_v2_metadata")
    parser.add_argument("--repo-id", default="bigcode/the-stack-v2")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--language", action="append", default=[], help="The Stack v2 language/config to resolve. Repeatable.")
    parser.add_argument("--metadata-limit-files", type=int, default=None, help="Debug only: cap metadata parquet files per run.")
    parser.add_argument("--limit", type=int, default=None, help="Debug only: max metadata rows to attempt.")
    parser.add_argument("--max-file-bytes", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--shard-records", type=int, default=50_000)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata_dir = Path(args.metadata_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    languages = args.language or DEFAULT_LANGUAGES
    resolved_ids = _load_resolved_ids(output_dir) if args.skip_existing else set()
    parquet_paths = _download_metadata_files(args.repo_id, languages, metadata_dir, args.endpoint, limit_files=args.metadata_limit_files)

    writer = JsonlShardWriter(output_dir=output_dir, shard_records=args.shard_records)
    stats = {"attempted": 0, "written": 0, "skipped": 0, "failed": 0}
    try:
        rows = _iter_metadata_rows(parquet_paths, max_file_bytes=args.max_file_bytes)
        if args.limit is not None:
            rows = _take(rows, args.limit)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {}
            with tqdm(desc="stack-v2 blobs", total=args.limit) as progress:
                for row in rows:
                    blob_id = str(row.get("blob_id") or "")
                    if not blob_id or blob_id in resolved_ids:
                        stats["skipped"] += 1
                        progress.update(1)
                        continue
                    futures[pool.submit(_resolve_row, row, args.timeout, args.retries)] = blob_id
                    stats["attempted"] += 1
                    if len(futures) >= max(1, args.workers) * 4:
                        _drain_completed(futures, writer, resolved_ids, stats, progress)
                while futures:
                    _drain_completed(futures, writer, resolved_ids, stats, progress)
    finally:
        writer.close()

    summary = {"output_dir": str(output_dir), "metadata_dir": str(metadata_dir), **stats}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _download_metadata_files(repo_id: str, languages: list[str], metadata_dir: Path, endpoint: str, *, limit_files: int | None) -> list[Path]:
    api = HfApi(endpoint=endpoint)
    info = api.dataset_info(repo_id, token=True)
    wanted = set(languages)
    siblings = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
    selected = [
        filename
        for filename in siblings
        if len(filename.split("/")) >= 3 and filename.split("/")[1] in wanted
    ]
    if not selected:
        raise RuntimeError(f"No The Stack v2 parquet files matched languages: {languages}")
    if limit_files is not None:
        selected = selected[: max(0, limit_files)]

    out: list[Path] = []
    for filename in tqdm(selected, desc="metadata parquet"):
        path = hf_hub_download(
            repo_id,
            filename,
            repo_type="dataset",
            revision="main",
            endpoint=endpoint,
            local_dir=metadata_dir,
            token=True,
        )
        out.append(Path(path))
    return out


def _iter_metadata_rows(paths: list[Path], *, max_file_bytes: int):
    columns = DEFAULT_COLUMNS
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=columns, batch_size=2048):
            for row in batch.to_pylist():
                length = row.get("length_bytes")
                if length is not None and int(length) > max_file_bytes:
                    continue
                yield row


def _take(rows, limit: int):
    for index, row in enumerate(rows):
        if index >= limit:
            return
        yield row


def _resolve_row(row: dict[str, Any], timeout: float, retries: int) -> dict[str, Any] | None:
    blob_id = str(row["blob_id"])
    encoding = row.get("src_encoding") or "utf-8"
    content = _download_swh_blob(blob_id, str(encoding), timeout=timeout, retries=retries)
    if content is None or not content.strip():
        return None
    doc_id = f"the_stack_v2/{blob_id}"
    return {
        "id": doc_id,
        "text": content,
        "blob_id": blob_id,
        "content_id": row.get("content_id"),
        "repo_name": row.get("repo_name"),
        "path": row.get("path"),
        "language": row.get("language"),
        "extension": row.get("extension"),
        "detected_licenses": row.get("detected_licenses"),
        "license_type": row.get("license_type"),
        "src_encoding": row.get("src_encoding"),
        "length_bytes": row.get("length_bytes"),
    }


def _download_swh_blob(blob_id: str, encoding: str, *, timeout: float, retries: int) -> str | None:
    try:
        from smart_open import open as smart_open
        import boto3
    except ImportError as exc:
        raise RuntimeError("The Stack v2 resolver requires boto3 and smart_open[s3]. Install them before resolving contents.") from exc

    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )
    client = session.client("s3", config=_boto_config(timeout))
    s3_url = f"s3://softwareheritage/content/{blob_id}"
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            with smart_open(s3_url, "rb", compression=".gz", transport_params={"client": client}) as fin:
                return fin.read().decode(encoding or "utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2**attempt, 30))
    print(f"failed to resolve {blob_id}: {last_error}", file=sys.stderr, flush=True)
    return None


def _boto_config(timeout: float):
    try:
        from botocore.config import Config
    except ImportError:
        return None
    return Config(connect_timeout=timeout, read_timeout=timeout, retries={"max_attempts": 1})


def _drain_completed(futures, writer, resolved_ids: set[str], stats: dict[str, int], progress: tqdm) -> None:
    done, pending = wait(futures, timeout=2.0, return_when=FIRST_COMPLETED)
    if not done:
        progress.refresh()
        return
    for future in done:
        blob_id = futures.pop(future)
        doc = future.result()
        if doc is None:
            stats["failed"] += 1
        else:
            writer.write(doc)
            resolved_ids.add(blob_id)
            stats["written"] += 1
        progress.update(1)


def _load_resolved_ids(output_dir: Path) -> set[str]:
    out: set[str] = set()
    for path in sorted(output_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                blob_id = row.get("blob_id")
                if blob_id:
                    out.add(str(blob_id))
    return out


class JsonlShardWriter:
    def __init__(self, *, output_dir: Path, shard_records: int) -> None:
        self.output_dir = output_dir
        self.shard_records = max(1, shard_records)
        self.shard_index = self._next_shard_index()
        self.record_count = 0
        self.current = None

    def write(self, row: dict[str, Any]) -> None:
        if self.current is None or self.record_count >= self.shard_records:
            self._open_next()
        assert self.current is not None
        self.current.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.record_count += 1

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None

    def _open_next(self) -> None:
        self.close()
        path = self.output_dir / f"stack_v2_{self.shard_index:06d}.jsonl"
        self.current = path.open("a", encoding="utf-8")
        self.record_count = _count_lines(path)
        self.shard_index += 1

    def _next_shard_index(self) -> int:
        existing = sorted(self.output_dir.glob("stack_v2_*.jsonl"))
        if not existing:
            return 0
        return max(int(path.stem.rsplit("_", 1)[-1]) for path in existing) + 1


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


if __name__ == "__main__":
    main()
