"""Build train/val parquet files from the cleaned RL jsonl pool.

The input pool is large, so this script streams it twice:
1. count valid reward rows and deterministically choose validation row indices;
2. write train/val parquet shards with pyarrow.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 50_000


def _valid_row(o: dict) -> bool:
    return bool(o.get("reward_model", {}).get("ground_truth"))


def _to_rl_row(o: dict) -> dict:
    return {
        "prompt": o["prompt"],
        "reward_model": o["reward_model"],
        "data_source": o["data_source"],
        "ability": o["ability"],
        "extra_info": json.dumps(o.get("extra_info", {}), ensure_ascii=False),
    }


def _count_valid(jsonl_path: Path) -> int:
    n = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if _valid_row(json.loads(line)):
                n += 1
    return n


def _choose_val_indices(n_rows: int, val_size: int, seed: int) -> set[int]:
    if val_size <= 0:
        return set()
    val_size = min(val_size, n_rows)
    rng = random.Random(seed)
    return set(rng.sample(range(n_rows), val_size))


class StreamingParquetWriter:
    def __init__(self, path: Path):
        self.path = path
        self.tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.writer = None
        self.buf = []
        self.n = 0

    def write(self, row: dict):
        self.buf.append(row)
        if len(self.buf) >= CHUNK:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        table = pa.Table.from_pylist(self.buf)
        if self.writer is None:
            self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.tmp_path, table.schema)
        self.writer.write_table(table)
        self.n += len(self.buf)
        self.buf = []

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()
        os.replace(self.tmp_path, self.path)


def write_split(jsonl_path: Path, train_out: Path, val_out: Path, val_indices: set[int]) -> tuple[int, int]:
    train_writer = StreamingParquetWriter(train_out)
    val_writer = StreamingParquetWriter(val_out)
    valid_i = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if not _valid_row(o):
                continue
            row = _to_rl_row(o)
            if valid_i in val_indices:
                val_writer.write(row)
            else:
                train_writer.write(row)
            valid_i += 1
            if valid_i % 100_000 == 0:
                print(f"processed valid rows: {valid_i:,}", flush=True)
    train_writer.close()
    val_writer.close()
    return train_writer.n, val_writer.n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-jsonl", default="/data/zilu/data_unified_v2/train_rl_s4clean.jsonl")
    ap.add_argument("--out-dir", default="/data/zilu/data_unified_v2/parquet")
    ap.add_argument("--train-name", default="train_rl_s4clean.parquet")
    ap.add_argument("--val-name", default="val_rl_s4clean.parquet")
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    jsonl_path = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)
    train_out = out_dir / args.train_name
    val_out = out_dir / args.val_name

    print(f"input: {jsonl_path}", flush=True)
    print(f"train_out: {train_out}", flush=True)
    print(f"val_out: {val_out}", flush=True)
    print(f"val_size: {args.val_size}, seed: {args.seed}", flush=True)

    n_valid = _count_valid(jsonl_path)
    print(f"valid rows: {n_valid:,}", flush=True)
    val_indices = _choose_val_indices(n_valid, args.val_size, args.seed)
    n_train, n_val = write_split(jsonl_path, train_out, val_out, val_indices)
    print(f"done: train={n_train:,}, val={n_val:,}", flush=True)


if __name__ == "__main__":
    main()
