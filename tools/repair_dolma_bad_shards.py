#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import unquote, urlparse

from tqdm import tqdm

import _bootstrap  # noqa: F401


DEFAULT_URL_LIST = Path("/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/dolma_verify/zstd_bad_urls.txt")
DEFAULT_FILES_ROOT = Path("/mnt/paper2any/ziyi/llmTrain/datasets/dolma/files")
DEFAULT_OUTPUT_DIR = Path("/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/dolma_repair")
GZIP_MAGIC = b"\x1f\x8b"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-download and verify corrupt Dolma shards.")
    parser.add_argument("--url-list", type=Path, default=DEFAULT_URL_LIST, help="File containing one bad Dolma URL per line.")
    parser.add_argument("--files-root", type=Path, default=DEFAULT_FILES_ROOT, help="Local root matching URL host/path layout.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for repair logs.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel files to repair.")
    parser.add_argument("--connections", type=int, default=16, help="aria2c connections per file, max 16.")
    parser.add_argument("--tries", type=int, default=5, help="Download attempts per file.")
    parser.add_argument("--retry-wait", type=int, default=10, help="aria2c retry wait seconds.")
    parser.add_argument("--keep-bad-backup", action="store_true", help="Rename old corrupt files to *.bad-* instead of deleting.")
    parser.add_argument("--use-env-proxy", action="store_true", help="Let aria2c inherit proxy environment variables.")
    parser.add_argument("--force", action="store_true", help="Repair even if current local file already passes validation.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned targets without downloading.")
    parser.add_argument("--limit", type=int, default=None, help="Debug only: process at most this many URLs.")
    args = parser.parse_args()

    if shutil.which("aria2c") is None:
        raise SystemExit("aria2c command not found")
    if shutil.which("gzip") is None:
        raise SystemExit("gzip command not found")
    if shutil.which("zstd") is None:
        raise SystemExit("zstd command not found")
    if not args.url_list.exists():
        raise SystemExit(f"url list does not exist: {args.url_list}")

    urls = _read_urls(args.url_list)
    if args.limit is not None:
        urls = urls[: args.limit]
    args.files_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_path = args.output_dir / "repair_results.jsonl"
    existing = _load_successes(result_path)
    jobs = []
    for url in urls:
        target = _target_for_url(url, args.files_root)
        if not args.force and existing.get(url) == _file_signature(target):
            continue
        jobs.append((url, target))

    print(
        json.dumps(
            {
                "url_list": str(args.url_list),
                "files_root": str(args.files_root),
                "output_dir": str(args.output_dir),
                "total_urls": len(urls),
                "pending": len(jobs),
                "workers": args.workers,
                "connections": min(max(1, args.connections), 16),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.dry_run:
        for url, target in jobs[:50]:
            print(f"{url} -> {target}")
        if len(jobs) > 50:
            print(f"... {len(jobs) - 50} more")
        return

    progress = tqdm(total=len(jobs), desc="repair dolma", unit="file")
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    futures = [
        pool.submit(
            _repair_one,
            url,
            target,
            args.output_dir / "tmp",
            min(max(1, args.connections), 16),
            max(1, args.tries),
            max(1, args.retry_wait),
            args.keep_bad_backup,
            args.use_env_proxy,
            args.force,
        )
        for url, target in jobs
    ]
    counts: dict[str, int] = {}
    try:
        with result_path.open("a", encoding="utf-8") as out:
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    status = str(row["status"])
                    counts[status] = counts.get(status, 0) + 1
                    out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    out.flush()
                    progress.update(1)
                    progress.set_postfix(counts, refresh=False)
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        progress.close()

    _write_summary(args.output_dir / "summary.json", result_path)


def _read_urls(path: Path) -> list[str]:
    urls = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#") or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _load_successes(path: Path) -> dict[str, str]:
    successes: dict[str, str] = {}
    if not path.exists():
        return successes
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "repaired":
                successes[str(row["url"])] = str(row.get("target_signature") or "")
    return successes


def _file_signature(path: Path) -> str:
    if not path.exists():
        return ""
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _repair_one(
    url: str,
    target: Path,
    tmp_root: Path,
    connections: int,
    tries: int,
    retry_wait: int,
    keep_bad_backup: bool,
    use_env_proxy: bool,
    force: bool,
) -> dict[str, object]:
    started = time.time()
    if target.exists() and not force:
        valid, message = _validate_compressed_file(target)
        if valid:
            return _result(url, target, "already_ok", started, message)

    tmp_root.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_root / (target.name + f".{os.getpid()}.{time.time_ns()}.download")
    try:
        _download_with_aria2(url, tmp_path, connections, tries, retry_wait, use_env_proxy)
        valid, message = _validate_compressed_file(tmp_path)
        if not valid:
            tmp_path.unlink(missing_ok=True)
            return _result(url, target, "download_failed_validation", started, message)
        if target.exists():
            if keep_bad_backup:
                backup = target.with_name(target.name + f".bad-{time.strftime('%Y%m%dT%H%M%S')}")
                target.replace(backup)
            else:
                target.unlink()
        tmp_path.replace(target)
        return _result(url, target, "repaired", started, message)
    except Exception as exc:  # noqa: BLE001
        tmp_path.unlink(missing_ok=True)
        return _result(url, target, "failed", started, f"{type(exc).__name__}: {exc}")


def _download_with_aria2(url: str, output: Path, connections: int, tries: int, retry_wait: int, use_env_proxy: bool) -> None:
    output.unlink(missing_ok=True)
    Path(str(output) + ".aria2").unlink(missing_ok=True)
    cmd = [
        "aria2c",
        "--continue=true",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--file-allocation=none",
        "--summary-interval=30",
        "--console-log-level=warn",
        "--show-console-readout=false",
        f"--max-tries={tries}",
        f"--retry-wait={retry_wait}",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        "--min-split-size=8M",
        "--dir",
        str(output.parent),
        "--out",
        output.name,
        url,
    ]
    subprocess.run(cmd, check=True, env=_aria2_env(use_env_proxy))
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"download produced empty file: {output}")


def _aria2_env(use_env_proxy: bool) -> dict[str, str]:
    env = os.environ.copy()
    if use_env_proxy:
        return env
    for key in [
        "all_proxy",
        "ALL_PROXY",
        "http_proxy",
        "HTTP_PROXY",
        "https_proxy",
        "HTTPS_PROXY",
        "ftp_proxy",
        "FTP_PROXY",
        "no_proxy",
        "NO_PROXY",
    ]:
        env.pop(key, None)
    return env


def _validate_compressed_file(path: Path) -> tuple[bool, str]:
    with path.open("rb") as f:
        header = f.read(4)
    if header.startswith(GZIP_MAGIC):
        cmd = ["gzip", "-t", str(path)]
    elif header == ZSTD_MAGIC:
        cmd = ["zstd", "-q", "-t", str(path)]
    else:
        return False, f"unknown compression header: {header.hex()}"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    message = proc.stderr.strip() or proc.stdout.strip()
    return proc.returncode == 0, message


def _target_for_url(url: str, files_root: Path) -> Path:
    parsed = urlparse(url)
    path = unquote(parsed.path).lstrip("/")
    if not path or path.endswith("/"):
        path += "download"
    return files_root / parsed.netloc / path


def _result(url: str, target: Path, status: str, started: float, message: str) -> dict[str, object]:
    return {
        "url": url,
        "target": str(target),
        "status": status,
        "message": message,
        "elapsed_seconds": round(time.time() - started, 3),
        "target_signature": _file_signature(target),
        "target_size_bytes": target.stat().st_size if target.exists() else 0,
    }


def _write_summary(path: Path, result_path: Path) -> None:
    counts: dict[str, int] = {}
    bytes_by_status: dict[str, int] = {}
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                status = str(row["status"])
                counts[status] = counts.get(status, 0) + 1
                bytes_by_status[status] = bytes_by_status.get(status, 0) + int(row.get("target_size_bytes") or 0)
    summary = {
        "result_path": str(result_path),
        "counts": counts,
        "size_tb_by_status": {k: round(v / 1024**4, 4) for k, v in bytes_by_status.items()},
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
