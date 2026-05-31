from __future__ import annotations

import json
import io
import heapq
import hashlib
import os
import signal
import shutil
import sqlite3
import time
from collections import Counter, OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm
import zstandard as zstd

from llmtrain.data.manifest import ShardInfo, write_manifest
from llmtrain.data.schemas import validate_record
from llmtrain.preprocessing.cleaners import TextCleaner
from llmtrain.preprocessing.config import PreprocessConfig, PreprocessSourceConfig
from llmtrain.preprocessing.dedup import exact_hash, hamming_distance, normalize_for_dedup, simhash
from llmtrain.preprocessing.documents import RawDocument
from llmtrain.preprocessing.parsers import iter_documents
from llmtrain.preprocessing.quality import HeuristicQualityScorer
from llmtrain.preprocessing.sources import expand_paths
from llmtrain.preprocessing.writers import RejectedWriter, RollingShardWriter


@dataclass(frozen=True)
class WorkerTask:
    task_id: str
    source: PreprocessSourceConfig
    paths: list[Path]
    candidate_path: Path
    rejected_path: Path
    done_path: Path
    stats_path: Path
    progress_path: Path


WORKER_WRITE_BUFFER_BYTES = 8 * 1024 * 1024
DEDUP_INDEX_WRITE_BUFFER_BYTES = 256 * 1024


def run_parallel_preprocess(cfg: PreprocessConfig, *, resume: bool = False) -> dict[str, Any]:
    cfg.writer.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = cfg.candidate_dir or (cfg.writer.output_dir / "candidates")
    if candidate_dir.exists() and not resume:
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    parallel_sources, fallback_sources = _split_sources(cfg.sources)
    if fallback_sources:
        from llmtrain.preprocessing.pipeline import run_stream_preprocess

        return run_stream_preprocess(cfg.model_copy(update={"num_workers": 0}), resume=resume)
    if not parallel_sources:
        from llmtrain.preprocessing.pipeline import run_stream_preprocess

        return run_stream_preprocess(cfg.model_copy(update={"num_workers": 0}), resume=resume)

    tasks = _build_tasks(parallel_sources, candidate_dir, cfg.worker_chunk_files)
    worker_stats: dict[str, Counter] = {}
    pending_tasks = [task for task in tasks if not _task_complete(task)]
    for task in tasks:
        if _task_complete(task):
            result = _load_task_result(task)
            worker_stats.setdefault(result["source"], Counter()).update(result["stats"])
    workers = max(1, cfg.num_workers or min(os.cpu_count() or 1, len(tasks)))
    if pending_tasks:
        progress = _progress_bar(total=sum(len(task.paths) for task in pending_tasks), desc="preprocess workers", unit="file")
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_ignore_parent_interrupts)
        futures = []
        try:
            futures = [pool.submit(_run_worker_task, task, cfg) for task in pending_tasks]
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
                if progress is not None:
                    _refresh_worker_progress(progress, pending_tasks)
                for future in done:
                    result = future.result()
                    stats = worker_stats.setdefault(result["source"], Counter())
                    stats.update(result["stats"])
                    if progress is not None:
                        progress.update(result["files"])
                        progress.set_postfix(source=result["source"], refresh=False)
        except BaseException:
            for future in futures:
                future.cancel()
            _shutdown_process_pool(pool, kill_after=8.0)
            raise
        else:
            pool.shutdown(wait=True, cancel_futures=True)
        finally:
            if progress is not None:
                progress.close()

    final_summary = _reduce_candidates(cfg, tasks, worker_stats, resume=resume)
    if not cfg.keep_candidates:
        shutil.rmtree(candidate_dir, ignore_errors=True)
    return final_summary


def _split_sources(sources: list[PreprocessSourceConfig]) -> tuple[list[PreprocessSourceConfig], list[PreprocessSourceConfig]]:
    parallel: list[PreprocessSourceConfig] = []
    fallback: list[PreprocessSourceConfig] = []
    for source in sources:
        if not source.enabled:
            continue
        if source.type in {"jsonl", "parquet"} and source.limit is None:
            parallel.append(source)
        else:
            fallback.append(source)
    return parallel, fallback


def _build_tasks(sources: list[PreprocessSourceConfig], candidate_dir: Path, chunk_files: int) -> list[WorkerTask]:
    tasks: list[WorkerTask] = []
    for source in sources:
        paths = expand_paths(source.paths)
        for task_index, chunk in enumerate(_chunks(paths, chunk_files)):
            safe_name = _safe_name(source.name)
            task_id = f"{safe_name}_{task_index:06d}"
            tasks.append(
                WorkerTask(
                    task_id=task_id,
                    source=source.model_copy(update={"paths": chunk, "limit": None}),
                    paths=chunk,
                    candidate_path=candidate_dir / f"{task_id}.jsonl.zst",
                    rejected_path=candidate_dir / f"{task_id}.rejected.jsonl",
                    done_path=candidate_dir / f"{task_id}.done.json",
                    stats_path=candidate_dir / f"{task_id}.stats.json",
                    progress_path=candidate_dir / f"{task_id}.progress.json",
                )
            )
    return tasks


def _chunks(paths: list[Path], chunk_size: int) -> list[list[Path]]:
    return [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]


def _run_worker_task(task: WorkerTask, cfg: PreprocessConfig) -> dict[str, Any]:
    try:
        return _run_worker_task_inner(task, cfg)
    except Exception as exc:
        paths = ", ".join(str(path) for path in task.paths)
        raise RuntimeError(
            f"preprocess worker task failed: task_id={task.task_id}, "
            f"source={task.source.name}, paths=[{paths}], error={type(exc).__name__}: {exc}"
        ) from None


def _run_worker_task_inner(task: WorkerTask, cfg: PreprocessConfig) -> dict[str, Any]:
    cleaner = TextCleaner(cfg.cleaning)
    scorer = HeuristicQualityScorer(cfg.quality)
    stats = Counter()
    task.candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_tmp = task.candidate_path.with_suffix(task.candidate_path.suffix + ".tmp")
    rejected_tmp = task.rejected_path.with_suffix(task.rejected_path.suffix + ".tmp")
    progress_state: dict[str, float | int] = {"last_seen": 0, "last_time": 0.0}
    for path in [
        task.candidate_path,
        _legacy_candidate_path(task),
        candidate_tmp,
        _legacy_candidate_path(task).with_suffix(_legacy_candidate_path(task).suffix + ".tmp"),
        rejected_tmp,
        task.done_path,
        task.stats_path,
        task.progress_path,
    ]:
        path.unlink(missing_ok=True)
    with _open_candidate_writer(candidate_tmp) as out, rejected_tmp.open(
        "w", encoding="utf-8", buffering=WORKER_WRITE_BUFFER_BYTES
    ) as rejected:
        for doc in iter_documents(task.source):
            stats["seen"] += 1
            excluded_field = _excluded_metadata_field(task.source, doc.metadata)
            if excluded_field is not None:
                stats[f"rejected_excluded_metadata_{excluded_field}"] += 1
                _write_rejected(rejected, doc.id, f"excluded_metadata_{excluded_field}", doc.metadata)
                _maybe_write_task_progress(task, cfg, stats, candidate_tmp, rejected_tmp, progress_state)
                continue
            cleaned, reason = cleaner.clean(doc)
            if cleaned is None:
                stats[f"rejected_{reason}"] += 1
                _write_rejected(rejected, doc.id, reason or "cleaning", doc.metadata)
                _maybe_write_task_progress(task, cfg, stats, candidate_tmp, rejected_tmp, progress_state)
                continue
            quality = scorer.score(cleaned)
            if quality.score < cfg.quality.min_score:
                stats["rejected_low_quality"] += 1
                _write_rejected(rejected, cleaned.id, "low_quality", {"score": quality.score, "signals": quality.signals})
                _maybe_write_task_progress(task, cfg, stats, candidate_tmp, rejected_tmp, progress_state)
                continue
            metadata = dict(cleaned.metadata)
            metadata["quality_score"] = quality.score
            metadata["quality_signals"] = quality.signals
            if cfg.dedup.exact:
                metadata["dedup_hash"] = exact_hash(cleaned.text)
            if cfg.dedup.simhash:
                metadata["simhash"] = str(simhash(normalize_for_dedup(cleaned.text), bits=cfg.dedup.simhash_bits))
            record = cleaned.__class__(cleaned.id, cleaned.text, cleaned.source, cleaned.domain, cleaned.language, metadata).to_record()
            out.write(json.dumps(record.__dict__, ensure_ascii=False, sort_keys=True) + "\n")
            stats["candidate"] += 1
            _maybe_write_task_progress(task, cfg, stats, candidate_tmp, rejected_tmp, progress_state)
    _write_task_progress(task, stats, candidate_tmp, rejected_tmp)
    candidate_tmp.replace(task.candidate_path)
    rejected_tmp.replace(task.rejected_path)
    result = {"task_id": task.task_id, "source": task.source.name, "files": len(task.paths), "stats": dict(stats)}
    task.stats_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    task.done_path.write_text(json.dumps({"task_id": task.task_id, "paths": [str(p) for p in task.paths]}, ensure_ascii=False, indent=2), encoding="utf-8")
    task.progress_path.unlink(missing_ok=True)
    return result


def _excluded_metadata_field(source: PreprocessSourceConfig, metadata: dict[str, Any]) -> str | None:
    for field, excluded_values in source.exclude_metadata_values.items():
        value = metadata.get(field)
        if value is None:
            continue
        value_text = str(value).strip().casefold()
        if value_text in {str(item).strip().casefold() for item in excluded_values}:
            return field
    return None


def _reduce_candidates(cfg: PreprocessConfig, tasks: list[WorkerTask], worker_stats: dict[str, Counter], *, resume: bool) -> dict[str, Any]:
    _clear_final_outputs(cfg)
    work_dir = cfg.writer.output_dir / "dedup_work"
    if work_dir.exists() and not resume:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    source_by_name = {source.name: source for source in cfg.sources}
    per_source = {name: Counter(stats) for name, stats in worker_stats.items()}
    candidate_files = [_candidate_read_path(task) for task in tasks if _task_complete(task)]
    total_candidates = sum(int(stats.get("candidate", 0)) for stats in worker_stats.values())
    workers = max(1, cfg.num_workers or min(os.cpu_count() or 1, len(candidate_files) or 1))
    index_summary = _build_dedup_index(candidate_files, cfg, work_dir, workers, total_candidates=total_candidates)
    drop_files: list[Path] = []
    if cfg.dedup.exact:
        exact_results = _run_resumable_parallel_jobs(
                index_summary["exact_bucket_jobs"],
                _reduce_exact_bucket,
                workers=workers,
                desc="exact dedup",
                unit="bucket",
                is_done=lambda job: _done_marker_valid(job.drop_path.with_suffix(job.drop_path.suffix + ".done.json")),
                load_result=lambda job: _load_json(job.drop_path.with_suffix(job.drop_path.suffix + ".result.json")),
        )
        drop_files.extend(Path(result["drop_path"]) for result in exact_results if result.get("drop_path"))
    if cfg.dedup.simhash:
        simhash_results = _run_resumable_parallel_jobs(
                index_summary["simhash_shard_jobs"],
                _reduce_simhash_shard,
                workers=workers,
                desc="simhash dedup",
                unit="shard",
                is_done=lambda job: _done_marker_valid(job.drop_path.with_suffix(job.drop_path.suffix + ".done.json")),
                load_result=lambda job: _load_json(job.drop_path.with_suffix(job.drop_path.suffix + ".result.json")),
        )
        drop_files.extend(Path(result["drop_path"]) for result in simhash_results if result.get("drop_path"))
    drop_db_path = _build_drop_db(drop_files, work_dir / "drops.sqlite")
    final_summary = _materialize_deduped_candidates(
        cfg,
        candidate_files,
        source_by_name,
        per_source,
        drop_db_path,
        workers=workers,
        work_dir=work_dir,
    )
    summary = dict(final_summary)
    summary.update(
        {
            "parallel": True,
            "candidate_files": len(candidate_files),
            "parallel_dedup": True,
            "parallel_dedup_index": True,
            "dedup_index_jobs": index_summary["index_jobs"],
            "dedup_index_workers": index_summary["index_workers"],
            "dedup_index_exact_shards": index_summary["exact_shards"],
            "dedup_index_simhash_shards": index_summary["simhash_shards"],
            "exact_bucket_files": len(index_summary["exact_bucket_jobs"]),
            "simhash_shard_files": len(index_summary["simhash_shard_jobs"]),
        }
    )
    (cfg.writer.output_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if cfg.cleanup_dedup_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    return summary


@dataclass(frozen=True)
class _ExactBucketJob:
    paths: list[Path]
    drop_path: Path


@dataclass(frozen=True)
class _SimhashShardJob:
    paths: list[Path]
    drop_path: Path
    bits: int
    threshold: int
    max_group_size: int


@dataclass(frozen=True)
class _MaterializeJob:
    job_id: int
    files: list[tuple[int, Path]]
    output_dir: Path
    shard_dir: Path
    rejected_path: Path
    result_path: Path
    done_path: Path
    writer: PreprocessWriterConfig
    sources: dict[str, dict[str, Any]]
    persist_state: bool
    write_exact_state: bool
    write_simhash_state: bool


@dataclass(frozen=True)
class _DedupIndexJob:
    job_id: int
    files: list[tuple[int, Path]]
    exact_dir: Path
    simhash_dir: Path
    result_path: Path
    done_path: Path
    exact: bool
    simhash: bool
    exact_shards: int
    simhash_shards: int
    bands: list[tuple[int, int]]


def _build_dedup_index(
    candidate_files: list[Path],
    cfg: PreprocessConfig,
    work_dir: Path,
    workers: int,
    *,
    total_candidates: int,
) -> dict[str, Any]:
    exact_parts_dir = work_dir / "exact_index_parts"
    simhash_parts_dir = work_dir / "simhash_index_parts"
    exact_drops_dir = work_dir / "exact_drops"
    simhash_drops_dir = work_dir / "simhash_drops"
    exact_parts_dir.mkdir(parents=True, exist_ok=True)
    simhash_parts_dir.mkdir(parents=True, exist_ok=True)
    exact_drops_dir.mkdir(parents=True, exist_ok=True)
    simhash_drops_dir.mkdir(parents=True, exist_ok=True)
    index_done_dir = work_dir / "index_done"
    index_done_dir.mkdir(parents=True, exist_ok=True)
    index_workers = max(1, min(workers, _env_int("LLMTRAIN_DEDUP_INDEX_WORKERS", cfg.dedup.index_workers)))
    exact_shards = _env_int("LLMTRAIN_DEDUP_EXACT_SHARDS", cfg.dedup.exact_shards)
    simhash_shards = _env_int("LLMTRAIN_DEDUP_SIMHASH_SHARDS", cfg.dedup.simhash_shards)
    bands = _simhash_bands(cfg.dedup.simhash_bits, cfg.dedup.simhash_threshold)
    index_jobs = [
        _DedupIndexJob(
            job_id=job_id,
            files=files,
            exact_dir=exact_parts_dir / f"part_{job_id:05d}",
            simhash_dir=simhash_parts_dir / f"part_{job_id:05d}",
            result_path=index_done_dir / f"part_{job_id:05d}.result.json",
            done_path=index_done_dir / f"part_{job_id:05d}.done.json",
            exact=cfg.dedup.exact,
            simhash=cfg.dedup.simhash,
            exact_shards=exact_shards,
            simhash_shards=simhash_shards,
            bands=bands,
        )
        for job_id, files in enumerate(_balanced_candidate_chunks(candidate_files, index_workers))
    ]
    _run_resumable_parallel_jobs(
        index_jobs,
        _build_dedup_index_part,
        workers=index_workers,
        desc="dedup index",
        unit="cand",
        progress_total=max(1, total_candidates),
        progress_step=lambda result: max(1, int(result.get("records", 0))),
        is_done=lambda job: _done_marker_valid(job.done_path),
        load_result=lambda job: _load_json(job.result_path),
    )
    return {
        "index_jobs": len(index_jobs),
        "index_workers": index_workers,
        "exact_shards": exact_shards,
        "simhash_shards": simhash_shards,
        "exact_bucket_jobs": _group_index_part_files(exact_parts_dir, exact_drops_dir),
        "simhash_shard_jobs": [
            _SimhashShardJob(
                paths=job.paths,
                drop_path=job.drop_path,
                bits=cfg.dedup.simhash_bits,
                threshold=cfg.dedup.simhash_threshold,
                max_group_size=cfg.dedup.simhash_max_group_size,
            )
            for job in _group_index_part_files(simhash_parts_dir, simhash_drops_dir)
        ],
    }


def _balanced_candidate_chunks(candidate_files: list[Path], workers: int) -> list[list[tuple[int, Path]]]:
    if not candidate_files:
        return []
    chunk_count = max(1, min(workers, len(candidate_files)))
    heap: list[tuple[int, int, list[tuple[int, Path]]]] = [(0, index, []) for index in range(chunk_count)]
    heapq.heapify(heap)
    files_by_size = sorted(
        enumerate(candidate_files),
        key=lambda item: item[1].stat().st_size if item[1].exists() else 0,
        reverse=True,
    )
    for file_index, path in files_by_size:
        total_size, chunk_index, files = heapq.heappop(heap)
        files.append((file_index, path))
        size = path.stat().st_size if path.exists() else 0
        heapq.heappush(heap, (total_size + size, chunk_index, files))
    return [files for _, _, files in sorted(heap, key=lambda item: item[1]) if files]


def _build_dedup_index_part(job: _DedupIndexJob) -> dict[str, Any]:
    writers = _LineWriterCache(
        max_open=max(1, job.exact_shards + job.simhash_shards),
        buffering=DEDUP_INDEX_WRITE_BUFFER_BYTES,
    )
    records = 0
    if _done_marker_valid(job.done_path):
        return _load_json(job.result_path)
    try:
        if job.exact_dir.exists():
            shutil.rmtree(job.exact_dir)
        if job.simhash_dir.exists():
            shutil.rmtree(job.simhash_dir)
        job.done_path.unlink(missing_ok=True)
        job.result_path.unlink(missing_ok=True)
        for file_index, path in job.files:
            with _open_candidate_reader(path) as f:
                for line_index, line in enumerate(f):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    metadata = row.get("metadata") or {}
                    exact = metadata.get("dedup_hash")
                    simhash_value = metadata.get("simhash")
                    entry = {
                        "file_index": file_index,
                        "line_index": line_index,
                        "id": str(row.get("id", "")),
                        "source": str(row.get("source", "")),
                    }
                    if job.exact and exact:
                        exact_text = str(exact)
                        exact_entry = dict(entry)
                        exact_entry["exact"] = exact_text
                        exact_shard = _stable_hex_shard(exact_text, job.exact_shards)
                        writers.write(job.exact_dir / f"{exact_shard:04d}.jsonl", exact_entry)
                    if job.simhash and simhash_value is not None:
                        simhash_int = int(simhash_value)
                        for band_index, band_value in _simhash_band_values(simhash_int, job.bands):
                            shard = (band_value + band_index * 1_000_003) % job.simhash_shards
                            sim_entry = dict(entry)
                            sim_entry.update({"band": band_index, "band_value": band_value, "simhash": str(simhash_int)})
                            writers.write(job.simhash_dir / f"{shard:04d}.jsonl", sim_entry)
                    records += 1
    finally:
        writers.close()
    result = {"job_id": job.job_id, "files": len(job.files), "records": records}
    _write_json_atomic(job.result_path, result)
    _write_json_atomic(job.done_path, {"job_id": job.job_id, "completed": True, "result_path": str(job.result_path)})
    return result


def _group_index_part_files(parts_dir: Path, drop_dir: Path) -> list[_ExactBucketJob]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(parts_dir.glob("part_*/*.jsonl")):
        grouped.setdefault(path.name, []).append(path)
    return [
        _ExactBucketJob(paths=paths, drop_path=drop_dir / f"{Path(name).stem}.drops.jsonl")
        for name, paths in sorted(grouped.items())
    ]


def _reduce_exact_bucket(job: _ExactBucketJob) -> dict[str, Any]:
    done_path = job.drop_path.with_suffix(job.drop_path.suffix + ".done.json")
    result_path = job.drop_path.with_suffix(job.drop_path.suffix + ".result.json")
    if _done_marker_valid(done_path):
        return _load_json(result_path)
    seen: dict[str, dict[str, Any]] = {}
    drop_path = job.drop_path
    tmp_drop_path = drop_path.with_suffix(drop_path.suffix + ".tmp")
    dropped = 0
    drop_path.unlink(missing_ok=True)
    tmp_drop_path.unlink(missing_ok=True)
    with tmp_drop_path.open("w", encoding="utf-8") as out:
        for path in job.paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    exact = row.get("exact")
                    if not exact:
                        continue
                    first = seen.get(exact)
                    if first is None:
                        seen[exact] = row
                        continue
                    _write_drop(
                        out,
                        row,
                        "exact_duplicate",
                        {"dedup_hash": exact, "first_id": first.get("id")},
                    )
                    dropped += 1
    if dropped == 0:
        tmp_drop_path.unlink(missing_ok=True)
        drop_path.unlink(missing_ok=True)
        result = {"drop_path": None, "dropped": 0}
    else:
        tmp_drop_path.replace(drop_path)
        result = {"drop_path": str(drop_path), "dropped": dropped}
    _write_json_atomic(result_path, result)
    _write_json_atomic(done_path, {"completed": True, "result_path": str(result_path)})
    return result


def _reduce_simhash_shard(job: _SimhashShardJob) -> dict[str, Any]:
    done_path = job.drop_path.with_suffix(job.drop_path.suffix + ".done.json")
    result_path = job.drop_path.with_suffix(job.drop_path.suffix + ".result.json")
    if _done_marker_valid(done_path):
        return _load_json(result_path)
    max_group_size = _env_int("LLMTRAIN_SIMHASH_MAX_GROUP_SIZE", job.max_group_size)
    groups: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
    drop_path = job.drop_path
    tmp_drop_path = drop_path.with_suffix(drop_path.suffix + ".tmp")
    dropped = 0
    capped = 0
    records = 0
    drop_path.unlink(missing_ok=True)
    tmp_drop_path.unlink(missing_ok=True)
    with tmp_drop_path.open("w", encoding="utf-8") as out:
        for path in job.paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    records += 1
                    simhash_value = int(row["simhash"])
                    group_key = (int(row["band"]), int(row["band_value"]))
                    group = groups.setdefault(group_key, [])
                    duplicate_of = None
                    for old_simhash, old_row in group:
                        if hamming_distance(simhash_value, old_simhash) <= job.threshold:
                            duplicate_of = old_row
                            break
                    if duplicate_of is not None:
                        _write_drop(
                            out,
                            row,
                            "near_duplicate",
                            {"simhash": str(simhash_value), "first_id": duplicate_of.get("id")},
                        )
                        dropped += 1
                        continue
                    if len(group) < max_group_size:
                        group.append((simhash_value, row))
                    else:
                        capped += 1
    if dropped == 0:
        tmp_drop_path.unlink(missing_ok=True)
        drop_path.unlink(missing_ok=True)
        result = {
            "drop_path": None,
            "dropped": 0,
            "records": records,
            "groups": len(groups),
            "max_group_size": max_group_size,
            "capped": capped,
        }
    else:
        tmp_drop_path.replace(drop_path)
        result = {
            "drop_path": str(drop_path),
            "dropped": dropped,
            "records": records,
            "groups": len(groups),
            "max_group_size": max_group_size,
            "capped": capped,
        }
    _write_json_atomic(result_path, result)
    _write_json_atomic(done_path, {"completed": True, "result_path": str(result_path)})
    return result


def _write_drop(handle: Any, row: dict[str, Any], reason: str, metadata: dict[str, Any]) -> None:
    handle.write(
        json.dumps(
            {
                "file_index": int(row["file_index"]),
                "line_index": int(row["line_index"]),
                "id": row.get("id", ""),
                "source": row.get("source", ""),
                "reason": reason,
                "metadata": metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )


def _build_drop_db(drop_files: list[Path], db_path: Path) -> Path:
    done_path = db_path.with_suffix(db_path.suffix + ".done.json")
    if _done_marker_valid(done_path) and db_path.exists():
        return db_path
    db_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA cache_size=-1048576")
    conn.execute(
        "CREATE TABLE drops (file_index INTEGER NOT NULL, line_index INTEGER NOT NULL, "
        "reason TEXT NOT NULL, metadata TEXT NOT NULL)"
    )
    insert_sql = "INSERT INTO drops(file_index, line_index, reason, metadata) VALUES (?, ?, ?, ?)"
    ordered_drop_files = sorted(drop_files, key=lambda path: 0 if "exact_buckets" in path.parts else 1)
    conn.execute("BEGIN")
    for path in ordered_drop_files:
        batch = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                batch.append(
                    (
                        int(row["file_index"]),
                        int(row["line_index"]),
                        str(row["reason"]),
                        json.dumps(row.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
                    )
                )
                if len(batch) >= 10_000:
                    conn.executemany(insert_sql, batch)
                    batch.clear()
            if batch:
                conn.executemany(insert_sql, batch)
    conn.commit()
    conn.execute("CREATE INDEX drops_file_line_idx ON drops(file_index, line_index)")
    conn.commit()
    conn.close()
    _write_json_atomic(done_path, {"completed": True, "drop_files": len(drop_files)})
    return db_path


def _materialize_deduped_candidates(
    cfg: PreprocessConfig,
    candidate_files: list[Path],
    source_by_name: dict[str, PreprocessSourceConfig],
    per_source: dict[str, Counter],
    drop_db_path: Path,
    *,
    workers: int,
    work_dir: Path,
) -> dict[str, Any]:
    materialize_dir = work_dir / "materialize"
    materialize_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = materialize_dir / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        _MaterializeJob(
            job_id=job_id,
            files=files,
            output_dir=cfg.writer.output_dir,
            shard_dir=cfg.writer.output_dir / "shards" / f"part_{job_id:05d}",
            rejected_path=rejected_dir / f"part_{job_id:05d}.rejected.jsonl",
            result_path=materialize_dir / f"part_{job_id:05d}.result.json",
            done_path=materialize_dir / f"part_{job_id:05d}.done.json",
            writer=cfg.writer,
            sources={name: source.model_dump(mode="json") for name, source in source_by_name.items()},
            persist_state=cfg.dedup.persist_state,
            write_exact_state=bool(cfg.dedup.persist_state and cfg.dedup.exact),
            write_simhash_state=bool(cfg.dedup.persist_state and cfg.dedup.simhash),
        )
        for job_id, files in enumerate(_balanced_candidate_chunks(candidate_files, max(1, min(workers, len(candidate_files)))))
    ]
    results = _run_resumable_parallel_jobs(
        jobs,
        _materialize_job,
        workers=workers,
        desc="write deduped",
        unit="file",
        progress_total=len(candidate_files),
        progress_step=lambda result: max(1, int(result.get("files", 0))),
        is_done=lambda job: _done_marker_valid(job.done_path),
        load_result=lambda job: _load_json(job.result_path),
        extra_arg=drop_db_path,
    )
    all_shards = [ShardInfo.model_validate(item) for result in results for item in result.get("shards", [])]
    _merge_rejected_parts([Path(result["rejected_path"]) for result in results], cfg.writer.rejected_path or cfg.writer.output_dir / "rejected.jsonl")
    _merge_dedup_state_parts(cfg, [Path(result["state_dir"]) for result in results if result.get("state_dir")])
    for result in results:
        for name, stats in (result.get("per_source") or {}).items():
            per_source.setdefault(name, Counter()).update(stats)
    totals = Counter()
    for stats in per_source.values():
        totals.update(stats)
    manifest = write_manifest(all_shards, cfg.writer.output_dir)
    return {
        "output_dir": str(cfg.writer.output_dir),
        "manifest": str(manifest.manifest),
        "manifest_meta": str(manifest.meta),
        "num_shards": len(all_shards),
        "totals": dict(totals),
        "sources": {k: dict(v) for k, v in per_source.items()},
    }


def _materialize_job(job: _MaterializeJob, drop_db_path: Path) -> dict[str, Any]:
    if _done_marker_valid(job.done_path):
        return _load_json(job.result_path)
    if job.shard_dir.exists():
        shutil.rmtree(job.shard_dir)
    job.shard_dir.mkdir(parents=True, exist_ok=True)
    job.rejected_path.unlink(missing_ok=True)
    state_dir = job.result_path.parent / f"state_part_{job.job_id:05d}"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    source_by_name = {name: PreprocessSourceConfig.model_validate(data) for name, data in job.sources.items()}
    writer_cfg = job.writer.model_copy(update={"output_dir": job.output_dir})
    rejected = RejectedWriter(job.rejected_path)
    writers: dict[str, RollingShardWriter] = {}
    per_source: dict[str, Counter] = {}
    state_handles = _open_materialize_state_writers(state_dir, exact=job.write_exact_state, simhash=job.write_simhash_state)
    conn = sqlite3.connect(f"file:{drop_db_path}?mode=ro", uri=True)
    try:
        for file_index, path in job.files:
            drops = _load_file_drops(conn, file_index)
            with _open_candidate_reader(path) as f:
                for line_index, line in enumerate(f):
                    if not line.strip():
                        continue
                    record = validate_record(json.loads(line))
                    source = source_by_name[record.source]
                    stats = per_source.setdefault(source.name, Counter())
                    drop = drops.get(line_index)
                    if drop is not None:
                        reason, metadata = drop
                        stats[f"rejected_{reason}"] += 1
                        rejected.write(record.id, reason, metadata)
                        continue
                    writer = writers.get(source.name)
                    if writer is None:
                        writer = RollingShardWriter(
                            writer_cfg,
                            source=source.name,
                            domain=source.domain,
                            language=source.language,
                            weight=source.weight,
                            license=source.license,
                            shard_dir=job.shard_dir,
                        )
                        writers[source.name] = writer
                    writer.write(record)
                    _write_dedup_state_record(state_handles, record.metadata)
                    stats["written"] += 1
    finally:
        conn.close()
        for handle in state_handles:
            handle.close()
    shards = []
    for writer in writers.values():
        shards.extend(writer.close())
    result = {
        "job_id": job.job_id,
        "files": len(job.files),
        "shards": [shard.model_dump(mode="json") for shard in shards],
        "per_source": {name: dict(stats) for name, stats in per_source.items()},
        "rejected_path": str(job.rejected_path),
        "state_dir": str(state_dir),
    }
    _write_json_atomic(job.result_path, result)
    _write_json_atomic(job.done_path, {"job_id": job.job_id, "completed": True, "result_path": str(job.result_path)})
    return result


def _load_file_drops(conn: sqlite3.Connection, file_index: int) -> dict[int, tuple[str, dict[str, Any]]]:
    rows = conn.execute(
        "SELECT line_index, reason, metadata FROM drops WHERE file_index = ? ORDER BY rowid DESC",
        (file_index,),
    )
    return {int(line_index): (str(reason), json.loads(metadata)) for line_index, reason, metadata in rows}


def _open_dedup_state_writers(cfg: PreprocessConfig) -> list[Any]:
    if not cfg.dedup.persist_state or not cfg.dedup.state_dir:
        return []
    cfg.dedup.state_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    if cfg.dedup.exact:
        handles.append((cfg.dedup.state_dir / "exact_hashes.txt").open("w", encoding="utf-8"))
    if cfg.dedup.simhash:
        handles.append((cfg.dedup.state_dir / "simhashes.jsonl").open("w", encoding="utf-8"))
    return handles


def _open_materialize_state_writers(state_dir: Path, *, exact: bool, simhash: bool) -> list[Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    if exact:
        handles.append((state_dir / "exact_hashes.txt").open("w", encoding="utf-8"))
    if simhash:
        handles.append((state_dir / "simhashes.jsonl").open("w", encoding="utf-8"))
    return handles


def _merge_rejected_parts(parts: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for path in sorted(parts):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                shutil.copyfileobj(f, out, length=1024 * 1024)
    tmp.replace(output_path)


def _merge_dedup_state_parts(cfg: PreprocessConfig, state_dirs: list[Path]) -> None:
    if not cfg.dedup.persist_state or not cfg.dedup.state_dir:
        return
    cfg.dedup.state_dir.mkdir(parents=True, exist_ok=True)
    if cfg.dedup.exact:
        _concat_files(
            [path / "exact_hashes.txt" for path in sorted(state_dirs)],
            cfg.dedup.state_dir / "exact_hashes.txt",
        )
    if cfg.dedup.simhash:
        _concat_files(
            [path / "simhashes.jsonl" for path in sorted(state_dirs)],
            cfg.dedup.state_dir / "simhashes.jsonl",
        )


def _concat_files(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for path in inputs:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                shutil.copyfileobj(f, out, length=1024 * 1024)
    tmp.replace(output)


def _write_dedup_state_record(handles: list[Any], metadata: dict[str, Any]) -> None:
    for handle in handles:
        if handle.name.endswith("exact_hashes.txt"):
            value = metadata.get("dedup_hash")
            if value:
                handle.write(f"{value}\n")
        elif handle.name.endswith("simhashes.jsonl"):
            value = metadata.get("simhash")
            if value is not None:
                handle.write(json.dumps({"simhash": str(value)}) + "\n")


def _run_parallel_jobs(
    jobs: list[Any],
    fn: Any,
    *,
    workers: int,
    desc: str,
    unit: str,
    progress_total: int | None = None,
    progress_step: Any | None = None,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    progress = _progress_bar(total=progress_total if progress_total is not None else len(jobs), desc=desc, unit=unit)
    results: list[dict[str, Any]] = []
    pool = ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs))), initializer=_worker_ignore_parent_interrupts)
    futures = []
    try:
        futures = [pool.submit(fn, job) for job in jobs]
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                results.append(result)
                if progress is not None:
                    step = progress_step(result) if progress_step is not None else 1
                    progress.update(max(1, int(step)))
    except BaseException:
        for future in futures:
            future.cancel()
        _shutdown_process_pool(pool, kill_after=8.0)
        raise
    else:
        pool.shutdown(wait=True, cancel_futures=True)
    finally:
        if progress is not None:
            progress.close()
    return results


def _run_resumable_parallel_jobs(
    jobs: list[Any],
    fn: Any,
    *,
    workers: int,
    desc: str,
    unit: str,
    is_done: Any,
    load_result: Any,
    progress_total: int | None = None,
    progress_step: Any | None = None,
    extra_arg: Any | None = None,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    results: list[dict[str, Any]] = []
    pending_jobs = []
    completed_progress = 0
    for job in jobs:
        if is_done(job):
            result = load_result(job)
            results.append(result)
            completed_progress += progress_step(result) if progress_step is not None else 1
        else:
            pending_jobs.append(job)
    progress = _progress_bar(total=progress_total if progress_total is not None else len(jobs), desc=desc, unit=unit)
    if progress is not None and completed_progress:
        progress.update(min(int(completed_progress), progress.total or int(completed_progress)))
    if not pending_jobs:
        if progress is not None:
            progress.close()
        return results
    pool = ProcessPoolExecutor(max_workers=max(1, min(workers, len(pending_jobs))), initializer=_worker_ignore_parent_interrupts)
    futures = []
    try:
        if extra_arg is None:
            futures = [pool.submit(fn, job) for job in pending_jobs]
        else:
            futures = [pool.submit(fn, job, extra_arg) for job in pending_jobs]
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                results.append(result)
                if progress is not None:
                    step = progress_step(result) if progress_step is not None else 1
                    progress.update(max(1, int(step)))
    except BaseException:
        for future in futures:
            future.cancel()
        _shutdown_process_pool(pool, kill_after=8.0)
        raise
    else:
        pool.shutdown(wait=True, cancel_futures=True)
    finally:
        if progress is not None:
            progress.close()
    return results


def _simhash_bands(bits: int, threshold: int) -> list[tuple[int, int]]:
    # Pigeonhole split: if distance <= threshold, at least one of threshold+1 bands is identical.
    num_bands = max(1, min(bits, threshold + 1))
    base_width = bits // num_bands
    remainder = bits % num_bands
    bands = []
    offset = 0
    for band in range(num_bands):
        width = base_width + (1 if band < remainder else 0)
        mask = ((1 << width) - 1) << offset
        bands.append((offset, mask))
        offset += width
    return bands


def _simhash_band_values(value: int, bands: list[tuple[int, int]]) -> Iterator[tuple[int, int]]:
    for index, (offset, mask) in enumerate(bands):
        yield index, (value & mask) >> offset


class _LineWriterCache:
    def __init__(self, *, max_open: int, buffering: int = -1) -> None:
        self.max_open = max_open
        self.buffering = buffering
        self._handles: OrderedDict[Path, Any] = OrderedDict()

    def write(self, path: Path, payload: dict[str, Any]) -> None:
        handle = self._get(path)
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def close(self) -> None:
        while self._handles:
            _, handle = self._handles.popitem(last=False)
            handle.close()

    def _get(self, path: Path) -> Any:
        handle = self._handles.get(path)
        if handle is not None:
            self._handles.move_to_end(path)
            return handle
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8", buffering=self.buffering)
        self._handles[path] = handle
        if len(self._handles) > self.max_open:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle


def _stable_hex_shard(value: str, shards: int) -> int:
    prefix = value[:8]
    try:
        number = int(prefix, 16)
    except ValueError:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        number = int(digest, 16)
    return number % shards


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _done_marker_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("completed"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _write_rejected(handle: Any, doc_id: str, reason: str, metadata: dict[str, Any] | None = None) -> None:
    handle.write(json.dumps({"id": doc_id, "reason": reason, "metadata": metadata or {}}, ensure_ascii=False) + "\n")


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


@contextmanager
def _open_candidate_writer(path: Path) -> Iterator[Any]:
    if path.name.endswith(".zst.tmp") or path.suffix == ".zst":
        with path.open("wb", buffering=WORKER_WRITE_BUFFER_BYTES) as raw:
            compressor = zstd.ZstdCompressor(level=3)
            with compressor.stream_writer(raw) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                    yield text
        return
    with path.open("w", encoding="utf-8", buffering=WORKER_WRITE_BUFFER_BYTES) as text:
        yield text


@contextmanager
def _open_candidate_reader(path: Path) -> Iterator[Any]:
    if path.suffix == ".zst":
        with path.open("rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as decompressed:
                with io.TextIOWrapper(decompressed, encoding="utf-8") as text:
                    yield text
        return
    with path.open("r", encoding="utf-8") as text:
        yield text


def _progress_enabled() -> bool:
    flag = os.environ.get("LLMTRAIN_PROGRESS")
    if flag is not None:
        return flag.lower() in {"1", "true", "yes", "on"}
    return True


def _progress_bar(*, total: int, desc: str, unit: str) -> Any | None:
    if not _progress_enabled():
        return None
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _shutdown_process_pool(pool: ProcessPoolExecutor, *, kill_after: float) -> None:
    processes = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + kill_after
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join(timeout=1.0)


def _worker_ignore_parent_interrupts() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, signal.SIG_DFL)


def _write_task_progress(task: WorkerTask, stats: Counter, candidate_tmp: Path, rejected_tmp: Path) -> None:
    payload = {
        "task_id": task.task_id,
        "source": task.source.name,
        "stats": dict(stats),
        "candidate_tmp_bytes": candidate_tmp.stat().st_size if candidate_tmp.exists() else 0,
        "rejected_tmp_bytes": rejected_tmp.stat().st_size if rejected_tmp.exists() else 0,
        "updated_at": time.time(),
    }
    tmp = task.progress_path.with_suffix(task.progress_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(task.progress_path)


def _maybe_write_task_progress(
    task: WorkerTask,
    cfg: PreprocessConfig,
    stats: Counter,
    candidate_tmp: Path,
    rejected_tmp: Path,
    progress_state: dict[str, float | int],
) -> None:
    seen = int(stats["seen"])
    if seen % cfg.progress_interval_records != 0:
        return
    now = time.time()
    if (
        seen != int(progress_state.get("last_seen", 0))
        and now - float(progress_state.get("last_time", 0.0)) >= cfg.progress_interval_seconds
    ):
        _write_task_progress(task, stats, candidate_tmp, rejected_tmp)
        progress_state["last_seen"] = seen
        progress_state["last_time"] = now


def _refresh_worker_progress(progress: Any, tasks: list[WorkerTask]) -> None:
    total_candidate = 0
    total_rejected = 0
    tmp_bytes = 0
    active = 0
    now = time.time()
    completed_cache = getattr(progress, "_llmtrain_completed_task_stats", None)
    if completed_cache is None:
        completed_cache = {}
        progress._llmtrain_completed_task_stats = completed_cache
    for task in tasks:
        if _task_complete(task):
            stats = completed_cache.get(task.task_id)
            if stats is None:
                stats = _load_task_result(task).get("stats") or {}
                completed_cache[task.task_id] = stats
            total_candidate += int(stats.get("candidate", 0))
            total_rejected += sum(int(v) for k, v in stats.items() if str(k).startswith("rejected_"))
            continue
        if not task.progress_path.exists():
            continue
        try:
            payload = json.loads(task.progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stats = payload.get("stats") or {}
        total_candidate += int(stats.get("candidate", 0))
        total_rejected += sum(int(v) for k, v in stats.items() if str(k).startswith("rejected_"))
        tmp_bytes += int(payload.get("candidate_tmp_bytes", 0))
        if now - float(payload.get("updated_at", 0.0)) < 30:
            active += 1
    previous_time = getattr(progress, "_llmtrain_candidate_rate_time", None)
    previous_candidate = getattr(progress, "_llmtrain_candidate_rate_value", None)
    candidates_per_second = 0.0
    if previous_time is not None and previous_candidate is not None and now > previous_time:
        candidates_per_second = max(0.0, (total_candidate - previous_candidate) / (now - previous_time))
    progress._llmtrain_candidate_rate_time = now
    progress._llmtrain_candidate_rate_value = total_candidate
    progress.set_postfix(
        {
            "active": active,
            "cand": total_candidate,
            "cands/s": f"{candidates_per_second:.1f}",
            "rej": total_rejected,
            "tmp_gb": f"{tmp_bytes / (1024 ** 3):.1f}",
        },
        refresh=True,
    )


def _task_complete(task: WorkerTask) -> bool:
    return (
        task.done_path.exists()
        and task.stats_path.exists()
        and _candidate_read_path(task).exists()
        and task.rejected_path.exists()
    )


def _candidate_read_path(task: WorkerTask) -> Path:
    if task.candidate_path.exists():
        return task.candidate_path
    return _legacy_candidate_path(task)


def _legacy_candidate_path(task: WorkerTask) -> Path:
    if task.candidate_path.name.endswith(".jsonl.zst"):
        return task.candidate_path.with_suffix("")
    return task.candidate_path


def _load_task_result(task: WorkerTask) -> dict[str, Any]:
    return json.loads(task.stats_path.read_text(encoding="utf-8"))


def _clear_final_outputs(cfg: PreprocessConfig) -> None:
    shard_dir = cfg.writer.output_dir / "shards"
    if shard_dir.exists():
        for path in shard_dir.iterdir():
            if path.is_file() and path.name.startswith(f"{cfg.writer.shard_prefix}_"):
                path.unlink()
    for path in [
        cfg.writer.output_dir / "manifest.jsonl",
        cfg.writer.output_dir / "manifest.meta.json",
        cfg.writer.output_dir / "stats.json",
        cfg.writer.rejected_path or cfg.writer.output_dir / "rejected.jsonl",
    ]:
        path.unlink(missing_ok=True)
    if cfg.dedup.state_dir and cfg.dedup.state_dir.exists():
        shutil.rmtree(cfg.dedup.state_dir)
