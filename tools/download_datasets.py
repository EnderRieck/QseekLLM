#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
import yaml
from huggingface_hub import snapshot_download
from tqdm import tqdm

import _bootstrap  # noqa: F401
from llmtrain.preprocessing.config import PreprocessSourceConfig
from llmtrain.preprocessing.parsers import _proxies_from_env
from llmtrain.utils.config import load_config


DEFAULT_HF_PATTERNS = [
    "data/**",
    "**/*.parquet",
    "*.parquet",
    "**/*.jsonl",
    "*.jsonl",
    "**/*.jsonl.gz",
    "*.jsonl.gz",
    "**/*.json.gz",
    "*.json.gz",
]
DEFAULT_REMOTE_JSON_PATTERNS = ["**/*.jsonl", "*.jsonl", "**/*.jsonl.gz", "*.jsonl.gz", "**/*.json.gz", "*.json.gz"]
DEFAULT_REMOTE_PARQUET_PATTERNS = ["**/*.parquet", "*.parquet"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download configured datasets to local storage for preprocessing.")
    parser.add_argument("--config", required=True, help="Preprocess config to read data sources from.")
    parser.add_argument("--output-dir", default="/mnt/paper2any/datasets/llmtrain_raw", help="Local dataset root.")
    parser.add_argument("--source", action="append", default=[], help="Source name to download. Repeatable. Defaults to all remote/HF sources.")
    parser.add_argument("--skip-source", action="append", default=[], help="Source name to skip. Repeatable.")
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel download workers.")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"), help="HF endpoint for snapshot_download.")
    parser.add_argument("--cache-dir", default=None, help="HF cache dir. Defaults to <repo>/hf_cache.")
    parser.add_argument("--local-config", default=None, help="Where to write a resolved config that reads downloaded local files.")
    parser.add_argument("--force", action="store_true", help="Force re-download HF files and URL files.")
    parser.add_argument("--no-env-proxy", action="store_true", help="Ignore HTTP_PROXY/HTTPS_PROXY for URL-list downloads.")
    parser.add_argument("--limit-url-files", type=int, default=None, help="Debug only: cap files downloaded from URL lists.")
    parser.add_argument("--snapshot-retries", type=int, default=5, help="Retry count for HF snapshot_download failures.")
    parser.add_argument("--retry-sleep", type=float, default=10.0, help="Initial sleep seconds between retries.")
    parser.add_argument("--tail-threshold", type=int, default=16, help="Use aggressive per-file strategy when URL downloads have this many files left.")
    parser.add_argument("--tail-file-workers", type=int, default=6, help="Parallel files to download in aggressive tail mode.")
    parser.add_argument("--tail-connections", type=int, default=16, help="Connections per file for aggressive tail URL downloads. aria2c supports at most 16.")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    cfg, _ = load_config(args.config)
    if cfg.preprocess is None:
        raise SystemExit("preprocess config section is required")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else Path(__file__).resolve().parents[1] / "hf_cache"
    selected = set(args.source)
    skipped = set(args.skip_source)
    sources = [
        source
        for source in cfg.preprocess.sources
        if source.name not in skipped
        and (source.enabled or source.name in selected)
        and (not selected or source.name in selected)
        and source.type in {"hf_dataset", "remote_parquet", "remote_jsonl"}
    ]

    manifest_path = output_dir / "download_manifest.json"
    local_config_path = Path(args.local_config).expanduser().resolve() if args.local_config else output_dir / "preprocess.local.yaml"
    results = _load_download_manifest(manifest_path)
    for source in sources:
        print(f"\n==> {source.name} ({source.type})")
        results[source.name] = _download_source(
            source,
            output_dir=output_dir,
            cache_dir=cache_dir,
            endpoint=args.endpoint,
            max_workers=args.max_workers,
            force=args.force,
            use_env_proxy=not args.no_env_proxy,
            limit_url_files=args.limit_url_files,
            snapshot_retries=args.snapshot_retries,
            retry_sleep=args.retry_sleep,
            tail_threshold=args.tail_threshold,
            tail_file_workers=args.tail_file_workers,
            tail_connections=args.tail_connections,
        )
        _write_download_outputs(cfg.model_dump(mode="json"), results, manifest_path, local_config_path)

    _write_download_outputs(cfg.model_dump(mode="json"), results, manifest_path, local_config_path)

    print(
        json.dumps(
            {
                "download_root": str(output_dir),
                "manifest": str(manifest_path),
                "local_config": str(local_config_path),
                "sources": {name: {"type": item["local_type"], "paths": item["local_paths"]} for name, item in results.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_download_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"download manifest must be a mapping: {path}")
    return data


def _write_download_outputs(config: dict[str, Any], results: dict[str, dict[str, Any]], manifest_path: Path, local_config_path: Path) -> None:
    _atomic_write_text(manifest_path, json.dumps(results, ensure_ascii=False, indent=2))
    _write_local_config(config, results, local_config_path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _download_source(
    source: PreprocessSourceConfig,
    *,
    output_dir: Path,
    cache_dir: Path,
    endpoint: str,
    max_workers: int,
    force: bool,
    use_env_proxy: bool,
    limit_url_files: int | None,
    snapshot_retries: int,
    retry_sleep: float,
    tail_threshold: int,
    tail_file_workers: int,
    tail_connections: int,
) -> dict[str, Any]:
    source_dir = output_dir / source.name
    source_dir.mkdir(parents=True, exist_ok=True)

    if source.type == "hf_dataset":
        if not source.hf_name:
            raise ValueError(f"{source.name}: hf_dataset requires hf_name")
        patterns = source.hf_include_patterns or DEFAULT_HF_PATTERNS
        local_dir = _snapshot_dataset(
            repo_id=source.hf_name,
            revision=source.hf_revision,
            allow_patterns=patterns,
            local_dir=source_dir,
            cache_dir=cache_dir,
            endpoint=endpoint,
            max_workers=max_workers,
            force=force,
            retries=snapshot_retries,
            retry_sleep=retry_sleep,
        )
        return _local_result(source, local_dir, requested_patterns=patterns)

    if source.hf_repo_id:
        patterns = source.hf_include_patterns or (
            DEFAULT_REMOTE_PARQUET_PATTERNS if source.type == "remote_parquet" else DEFAULT_REMOTE_JSON_PATTERNS
        )
        local_dir = _snapshot_dataset(
            repo_id=source.hf_repo_id,
            revision=source.hf_revision,
            allow_patterns=patterns,
            local_dir=source_dir,
            cache_dir=cache_dir,
            endpoint=endpoint,
            max_workers=max_workers,
            force=force,
            retries=snapshot_retries,
            retry_sleep=retry_sleep,
        )
        return _local_result(source, local_dir, requested_patterns=patterns)

    urls = _source_urls(source, endpoint=endpoint, use_env_proxy=use_env_proxy)
    if limit_url_files is not None:
        urls = urls[:limit_url_files]
    files_dir = source_dir / "files"
    downloaded = _download_urls(
        urls,
        files_dir=files_dir,
        max_workers=max_workers,
        force=force,
        use_env_proxy=use_env_proxy,
        tail_threshold=tail_threshold,
        tail_file_workers=tail_file_workers,
        tail_connections=tail_connections,
    )
    local_type = "parquet" if source.type == "remote_parquet" else "jsonl"
    globs = _local_globs(files_dir, local_type)
    return {
        "source_type": source.type,
        "local_type": local_type,
        "download_dir": str(source_dir),
        "file_count": len(downloaded),
        "local_paths": globs,
        "downloaded_files": [str(path) for path in downloaded],
    }


def _snapshot_dataset(
    *,
    repo_id: str,
    revision: str,
    allow_patterns: list[str],
    local_dir: Path,
    cache_dir: Path,
    endpoint: str,
    max_workers: int,
    force: bool,
    retries: int,
    retry_sleep: float,
) -> Path:
    attempts = max(1, retries)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=allow_patterns,
                local_dir=local_dir,
                cache_dir=cache_dir,
                endpoint=endpoint,
                max_workers=max_workers,
                force_download=force and attempt == 1,
                token=os.environ.get("HF_TOKEN") or None,
            )
            return Path(path)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts:
                break
            sleep = retry_sleep * (2 ** (attempt - 1))
            print(
                f"snapshot_download failed for {repo_id} on attempt {attempt}/{attempts}: {exc}. "
                f"Retrying in {sleep:.1f}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep)
    raise RuntimeError(f"snapshot_download failed for {repo_id} after {attempts} attempts: {last_error}") from last_error


def _source_urls(source: PreprocessSourceConfig, *, endpoint: str, use_env_proxy: bool) -> list[str]:
    urls = list(source.urls)
    proxies = _proxies_from_env() if use_env_proxy else None
    if source.url_list_path:
        urls.extend(line.strip() for line in source.url_list_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if source.url_list_url:
        url_list_url = _with_hf_endpoint(source.url_list_url, endpoint)
        with requests.get(url_list_url, timeout=60, proxies=proxies) as response:
            response.raise_for_status()
            urls.extend(line.strip() for line in response.text.splitlines() if line.strip())
    if not urls:
        raise ValueError(f"{source.name}: no urls/url_list_path/url_list_url to download")
    return urls


def _with_hf_endpoint(url: str, endpoint: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc != "huggingface.co":
        return url
    endpoint = endpoint.rstrip("/")
    return endpoint + parsed.path + (f"?{parsed.query}" if parsed.query else "")


def _download_urls(
    urls: list[str],
    *,
    files_dir: Path,
    max_workers: int,
    force: bool,
    use_env_proxy: bool,
    tail_threshold: int,
    tail_file_workers: int,
    tail_connections: int,
) -> list[Path]:
    files_dir.mkdir(parents=True, exist_ok=True)
    proxies = _proxies_from_env() if use_env_proxy else None
    out: list[Path] = []
    pending: list[str] = []
    for url in urls:
        target = _target_for_url(url, files_dir)
        if target.exists() and target.stat().st_size > 0 and not force:
            out.append(target)
        else:
            pending.append(url)
    tail_threshold = max(0, tail_threshold)
    regular_count = max(0, len(pending) - tail_threshold)
    regular_urls, tail_urls = pending[:regular_count], pending[regular_count:]
    with tqdm(total=len(urls), initial=len(out), desc="url files") as progress:
        if regular_urls:
            out.extend(_download_url_batch(regular_urls, files_dir, proxies, force, max_workers, progress))
        if tail_urls:
            out.extend(
                _download_url_tail_batch(
                    tail_urls,
                    files_dir,
                    proxies,
                    force,
                    max(1, tail_file_workers),
                    tail_connections,
                    progress,
                )
            )
    return out


def _download_url_tail_batch(
    urls: list[str],
    files_dir: Path,
    proxies: dict[str, str] | None,
    force: bool,
    tail_file_workers: int,
    tail_connections: int,
    progress: tqdm,
) -> list[Path]:
    workers = min(max(1, tail_file_workers), len(urls))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one_url_aggressive, url, files_dir, proxies, force, tail_connections) for url in urls}
        out: list[Path] = []
        last_bytes, last_time = _download_disk_bytes(files_dir), time.monotonic()
        while futures:
            done, futures = wait(futures, timeout=2.0, return_when=FIRST_COMPLETED)
            for future in done:
                out.append(future.result())
            if done:
                progress.update(len(done))
            now = time.monotonic()
            current_bytes = _download_disk_bytes(files_dir)
            elapsed = max(now - last_time, 1e-6)
            speed = max(0.0, (current_bytes - last_bytes) / elapsed)
            progress.set_postfix_str(
                f"disk={_format_bytes(current_bytes)} parts={_count_part_files(files_dir)} "
                f"tail=aggressive tail_pending={len(futures)} speed={_format_bytes(speed)}/s",
                refresh=True,
            )
            last_bytes, last_time = current_bytes, now
        return out


def _download_url_batch(
    urls: list[str],
    files_dir: Path,
    proxies: dict[str, str] | None,
    force: bool,
    max_workers: int,
    progress: tqdm,
) -> list[Path]:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one_url, url, files_dir, proxies, force) for url in urls}
        out: list[Path] = []
        last_bytes, last_time = _download_disk_bytes(files_dir), time.monotonic()
        while futures:
            done, futures = wait(futures, timeout=2.0, return_when=FIRST_COMPLETED)
            for future in done:
                out.append(future.result())
            if done:
                progress.update(len(done))
            now = time.monotonic()
            current_bytes = _download_disk_bytes(files_dir)
            elapsed = max(now - last_time, 1e-6)
            speed = max(0.0, (current_bytes - last_bytes) / elapsed)
            progress.set_postfix_str(
                f"disk={_format_bytes(current_bytes)} parts={_count_part_files(files_dir)} "
                f"pending={len(futures)} speed={_format_bytes(speed)}/s",
                refresh=True,
            )
            last_bytes, last_time = current_bytes, now
        return out


def _download_disk_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def _count_part_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*.part") if path.is_file())


def _format_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{value:.0f}{unit}"
        value /= 1024
    return f"{value:.1f}PB"


def _download_one_url(url: str, files_dir: Path, proxies: dict[str, str] | None, force: bool) -> Path:
    target = _target_for_url(url, files_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not force:
        return target

    part = target.with_suffix(target.suffix + ".part")
    if force and part.exists():
        part.unlink()

    last_error: Exception | None = None
    for attempt in range(5):
        headers = {}
        mode = "wb"
        if part.exists() and not force:
            size = part.stat().st_size
            if size > 0:
                headers["Range"] = f"bytes={size}-"
                mode = "ab"
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers, proxies=proxies) as response:
                if response.status_code == 416 and part.exists():
                    part.replace(target)
                    return target
                response.raise_for_status()
                if headers.get("Range") and response.status_code != 206:
                    mode = "wb"
                with part.open(mode + ("" if "b" in mode else "b")) as f:
                    shutil.copyfileobj(response.raw, f, length=1024 * 1024)
            part.replace(target)
            return target
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"failed to download {url}: {last_error}") from last_error


def _download_one_url_aggressive(
    url: str,
    files_dir: Path,
    proxies: dict[str, str] | None,
    force: bool,
    connections: int,
) -> Path:
    target = _target_for_url(url, files_dir)
    if target.exists() and target.stat().st_size > 0 and not force:
        return target
    aria2c = shutil.which("aria2c")
    if aria2c:
        try:
            return _download_one_url_aria2(url, files_dir, proxies, force, max(1, connections))
        except KeyboardInterrupt:
            raise
        except subprocess.CalledProcessError as exc:
            if exc.returncode in {7, 130, -signal.SIGINT}:
                raise
            print(f"aria2c failed for {url}: {exc}. Falling back to range downloader.", file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"aria2c failed for {url}: {exc}. Falling back to range downloader.", file=sys.stderr, flush=True)
    return _download_one_url_ranges(url, files_dir, proxies, force, max(1, connections))


def _download_one_url_aria2(url: str, files_dir: Path, proxies: dict[str, str] | None, force: bool, connections: int) -> Path:
    target = _target_for_url(url, files_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    connections = min(max(1, connections), 16)
    part = target.with_suffix(target.suffix + ".part")
    if force:
        target.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        Path(str(part) + ".aria2").unlink(missing_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    cmd = [
        "aria2c",
        "--continue=true",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        "--min-split-size=8M",
        "--summary-interval=30",
        "--console-log-level=warn",
        "--show-console-readout=false",
        "--dir",
        str(target.parent),
        "--out",
        part.name,
        url,
    ]
    if proxies:
        if proxies.get("http"):
            cmd.extend(["--http-proxy", proxies["http"]])
        if proxies.get("https"):
            cmd.extend(["--https-proxy", proxies["https"]])
    subprocess.run(cmd, check=True)
    if not part.exists() or part.stat().st_size == 0:
        raise RuntimeError(f"aria2c completed but did not create {part}")
    part.replace(target)
    Path(str(part) + ".aria2").unlink(missing_ok=True)
    return target


def _download_one_url_ranges(url: str, files_dir: Path, proxies: dict[str, str] | None, force: bool, connections: int) -> Path:
    target = _target_for_url(url, files_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if force:
        target.unlink(missing_ok=True)
        target.with_suffix(target.suffix + ".part").unlink(missing_ok=True)
        for old_part in target.parent.glob(target.name + ".range.*.part"):
            old_part.unlink()
    if target.exists() and target.stat().st_size > 0:
        return target

    size, accepts_ranges = _remote_file_info(url, proxies)
    if not size or not accepts_ranges:
        return _download_one_url(url, files_dir, proxies, force)

    part = target.with_suffix(target.suffix + ".part")
    prefix_size = part.stat().st_size if part.exists() else 0
    if prefix_size >= size:
        part.replace(target)
        for old_part in target.parent.glob(target.name + ".range.*.part"):
            old_part.unlink()
        return target
    if prefix_size > 0:
        for old_part in target.parent.glob(target.name + ".range.*.part"):
            old_part.unlink()

    range_count = _range_count(size, connections, start=prefix_size)
    ranges = _missing_ranges(target, size, connections, start=prefix_size)
    if ranges:
        with ThreadPoolExecutor(max_workers=min(connections, len(ranges))) as pool:
            futures = [
                pool.submit(_download_range_part, url, target, index, start, end, proxies)
                for index, (start, end) in enumerate(ranges)
            ]
            for future in futures:
                future.result()
    _assemble_range_parts(target, size, range_count, prefix=part if prefix_size > 0 else None)
    return target


def _remote_file_info(url: str, proxies: dict[str, str] | None) -> tuple[int | None, bool]:
    response = requests.head(url, timeout=60, allow_redirects=True, proxies=proxies)
    if response.status_code >= 400:
        response = requests.get(url, timeout=60, headers={"Range": "bytes=0-0"}, stream=True, proxies=proxies)
    response.raise_for_status()
    size = response.headers.get("Content-Length")
    if response.status_code == 206:
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            size = content_range.rsplit("/", 1)[-1]
    accepts_ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes" or response.status_code == 206
    return (int(size) if size and size.isdigit() else None), accepts_ranges


def _missing_ranges(target: Path, size: int, connections: int, *, start: int = 0) -> list[tuple[int, int]]:
    range_count = _range_count(size, connections, start=start)
    if range_count == 0:
        return []
    remaining = size - start
    chunk = (remaining + connections - 1) // connections
    ranges = []
    for index in range(range_count):
        range_start = start + index * chunk
        end = min(size - 1, range_start + chunk - 1)
        part = _range_part_path(target, index)
        if part.exists() and part.stat().st_size == end - range_start + 1:
            continue
        ranges.append((range_start, end))
    return ranges


def _range_count(size: int, connections: int, *, start: int = 0) -> int:
    remaining = size - start
    if remaining <= 0:
        return 0
    return min(connections, remaining)


def _download_range_part(
    url: str,
    target: Path,
    index: int,
    start: int,
    end: int,
    proxies: dict[str, str] | None,
) -> Path:
    part = _range_part_path(target, index)
    offset = part.stat().st_size if part.exists() else 0
    if offset >= end - start + 1:
        return part
    headers = {"Range": f"bytes={start + offset}-{end}"}
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers, proxies=proxies) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"range request returned HTTP {response.status_code}")
                with part.open("ab") as f:
                    shutil.copyfileobj(response.raw, f, length=1024 * 1024)
            expected = end - start + 1
            if part.stat().st_size != expected:
                raise RuntimeError(f"range part size mismatch: {part.stat().st_size} != {expected}")
            return part
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={start + offset}-{end}"}
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"failed range download {url} [{start}-{end}]: {last_error}") from last_error


def _assemble_range_parts(target: Path, size: int, connections: int, *, prefix: Path | None = None) -> None:
    assembled = target.with_suffix(target.suffix + ".assembling")
    with assembled.open("wb") as out:
        if prefix is not None and prefix.exists():
            with prefix.open("rb") as f:
                shutil.copyfileobj(f, out, length=1024 * 1024)
        for index in range(connections):
            part = _range_part_path(target, index)
            if not part.exists():
                continue
            with part.open("rb") as f:
                shutil.copyfileobj(f, out, length=1024 * 1024)
    if assembled.stat().st_size != size:
        assembled.unlink(missing_ok=True)
        raise RuntimeError(f"assembled file size mismatch: {assembled.stat().st_size} != {size}")
    assembled.replace(target)
    for index in range(connections):
        _range_part_path(target, index).unlink(missing_ok=True)
    target.with_suffix(target.suffix + ".part").unlink(missing_ok=True)


def _range_part_path(target: Path, index: int) -> Path:
    return target.with_name(f"{target.name}.range.{index:04d}.part")


def _target_for_url(url: str, files_dir: Path) -> Path:
    parsed = urlparse(url)
    path = unquote(parsed.path).lstrip("/")
    if not path or path.endswith("/"):
        path = path + "download"
    return files_dir / parsed.netloc / path


def _local_result(source: PreprocessSourceConfig, local_dir: Path, *, requested_patterns: list[str]) -> dict[str, Any]:
    files = [path for path in local_dir.rglob("*") if path.is_file()]
    parquet = [path for path in files if path.suffix == ".parquet"]
    jsonl = [
        path
        for path in files
        if path.name.endswith((".jsonl", ".jsonl.gz", ".json.gz"))
    ]
    if parquet and not jsonl:
        local_type = "parquet"
    elif jsonl and not parquet:
        local_type = "jsonl"
    elif source.type == "remote_parquet":
        local_type = "parquet"
    elif source.type == "remote_jsonl":
        local_type = "jsonl"
    elif parquet:
        local_type = "parquet"
    elif jsonl:
        local_type = "jsonl"
    else:
        raise ValueError(f"{source.name}: downloaded no supported data files under {local_dir}")

    return {
        "source_type": source.type,
        "local_type": local_type,
        "download_dir": str(local_dir),
        "file_count": len(parquet if local_type == "parquet" else jsonl),
        "local_paths": _local_paths_from_patterns(local_dir, requested_patterns, local_type),
    }


def _local_paths_from_patterns(local_dir: Path, patterns: list[str], local_type: str) -> list[str]:
    suffix_patterns = DEFAULT_REMOTE_PARQUET_PATTERNS if local_type == "parquet" else DEFAULT_REMOTE_JSON_PATTERNS
    candidates = patterns or suffix_patterns
    out = [str(local_dir / pattern) for pattern in candidates if _pattern_matches_type(pattern, local_type)]
    return out or _local_globs(local_dir, local_type)


def _pattern_matches_type(pattern: str, local_type: str) -> bool:
    if local_type == "parquet":
        return "parquet" in pattern
    return "jsonl" in pattern or "json.gz" in pattern


def _local_globs(root: Path, local_type: str) -> list[str]:
    if local_type == "parquet":
        return [str(root / "**/*.parquet")]
    return [str(root / "**/*.jsonl"), str(root / "**/*.jsonl.gz"), str(root / "**/*.json.gz")]


def _write_local_config(config: dict[str, Any], results: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config.pop("extends", None)
    preprocess = config.get("preprocess") or {}
    for source in preprocess.get("sources", []):
        result = results.get(source.get("name"))
        if not result:
            continue
        source["type"] = result["local_type"]
        source["paths"] = result["local_paths"]
        for key in [
            "urls",
            "url_list_path",
            "url_list_url",
            "hf_repo_id",
            "hf_revision",
            "hf_include_patterns",
            "hf_name",
            "hf_config",
            "hf_split",
            "hf_streaming",
        ]:
            source.pop(key, None)
    _atomic_write_text(path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
