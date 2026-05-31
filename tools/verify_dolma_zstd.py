#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from tqdm import tqdm

import _bootstrap  # noqa: F401


DEFAULT_ROOT = Path("/mnt/paper2any/ziyi/llmTrain/datasets/dolma/files/olmo-data.org/dolma-v1_7")
DEFAULT_OUTPUT_DIR = Path("/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/dolma_verify")
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
GZIP_MAGIC = b"\x1f\x8b"
BASE_URL = "https://olmo-data.org/dolma-v1_7"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Dolma zstd-wrapped .json.gz shards with zstd -t.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Dolma v1_7 root containing shard subdirectories.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for verify outputs.")
    parser.add_argument("--workers", type=int, default=16, help="Parallel zstd -t workers.")
    parser.add_argument("--force", action="store_true", help="Retest files even if an existing matching result is present.")
    parser.add_argument("--include-gzip", action="store_true", help="Also test gzip-header files with gzip -t.")
    parser.add_argument("--limit", type=int, default=None, help="Debug only: test at most this many files.")
    args = parser.parse_args()

    if shutil.which("zstd") is None:
        raise SystemExit("zstd command not found")
    if args.include_gzip and shutil.which("gzip") is None:
        raise SystemExit("gzip command not found")
    if not args.root.exists():
        raise SystemExit(f"root does not exist: {args.root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "zstd_results.jsonl"
    ok_path = args.output_dir / "zstd_ok.txt"
    bad_path = args.output_dir / "zstd_bad.txt"
    skip_path = args.output_dir / "zstd_skipped.txt"
    bad_urls_path = args.output_dir / "zstd_bad_urls.txt"

    existing = {} if args.force else _load_existing_results(result_path)
    files = sorted(args.root.rglob("*.json.gz"))
    if args.limit is not None:
        files = files[: args.limit]

    tasks: list[Path] = []
    skipped_resume = 0
    for path in files:
        stat = path.stat()
        old = existing.get(str(path))
        if old and old.get("size_bytes") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
            skipped_resume += 1
            continue
        tasks.append(path)

    print(
        json.dumps(
            {
                "root": str(args.root),
                "output_dir": str(args.output_dir),
                "total_files": len(files),
                "pending": len(tasks),
                "skipped_by_resume": skipped_resume,
                "workers": args.workers,
            },
            ensure_ascii=False,
        )
    )

    counts: dict[str, int] = {}
    with result_path.open("a", encoding="utf-8") as results, ok_path.open("a", encoding="utf-8") as ok, bad_path.open(
        "a", encoding="utf-8"
    ) as bad, skip_path.open("a", encoding="utf-8") as skipped, bad_urls_path.open(
        "a", encoding="utf-8"
    ) as bad_urls:
        progress = tqdm(total=len(tasks), desc="verify dolma", unit="file")
        pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
        futures = []
        try:
            futures = [pool.submit(_verify_one, path, args.root, args.include_gzip) for path in tasks]
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    status = str(row["status"])
                    counts[status] = counts.get(status, 0) + 1
                    results.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    results.flush()
                    rel = str(row["relative_path"])
                    if status == "ok":
                        ok.write(rel + "\n")
                        ok.flush()
                    elif status == "bad":
                        bad.write(rel + "\n")
                        bad.flush()
                        bad_urls.write(f"{BASE_URL}/{rel}\n")
                        bad_urls.flush()
                    else:
                        skipped.write(f"{status}\t{rel}\n")
                        skipped.flush()
                    progress.update(1)
                    progress.set_postfix(counts, refresh=False)
        finally:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            progress.close()

    _write_summary(args.output_dir / "summary.json", result_path, args.root)


def _load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            existing[str(row["path"])] = row
    return existing


def _verify_one(path: Path, root: Path, include_gzip: bool) -> dict[str, Any]:
    stat = path.stat()
    with path.open("rb") as f:
        header = f.read(4)
    rel = path.relative_to(root)
    if header == ZSTD_MAGIC:
        cmd = ["zstd", "-q", "-t", str(path)]
        kind = "zstd"
    elif header.startswith(GZIP_MAGIC):
        if not include_gzip:
            return _row(path, rel, stat, "skipped_gzip", "gzip header; use --include-gzip to test with gzip -t")
        cmd = ["gzip", "-t", str(path)]
        kind = "gzip"
    else:
        return _row(path, rel, stat, "skipped_unknown_header", f"header={header.hex()}")

    started = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - started
    status = "ok" if proc.returncode == 0 else "bad"
    message = proc.stderr.strip() or proc.stdout.strip()
    row = _row(path, rel, stat, status, message)
    row.update({"kind": kind, "elapsed_seconds": round(elapsed, 3), "returncode": proc.returncode})
    return row


def _row(path: Path, rel: Path, stat: Any, status: str, message: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": str(rel),
        "status": status,
        "message": message,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _write_summary(path: Path, result_path: Path, root: Path) -> None:
    counts: dict[str, int] = {}
    size_by_status: dict[str, int] = {}
    latest: dict[str, dict[str, Any]] = {}
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                latest[str(row["path"])] = row
    for row in latest.values():
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
        size_by_status[status] = size_by_status.get(status, 0) + int(row.get("size_bytes", 0))
    summary = {
        "root": str(root),
        "result_path": str(result_path),
        "counts": counts,
        "size_tb_by_status": {k: round(v / 1024**4, 4) for k, v in size_by_status.items()},
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
