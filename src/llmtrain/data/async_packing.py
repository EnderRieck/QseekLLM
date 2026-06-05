from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch

from llmtrain.data.config import PipelineConfig
from llmtrain.data.manifest import ManifestMeta
from llmtrain.data.packing import PackedDataIterator
from llmtrain.data.stream import TrainingRecordStream
from llmtrain.interfaces import Batch
from llmtrain.observability.sinks import JsonlSink
from llmtrain.tokenizer.adapter import load_tokenizer
from llmtrain.tokenizer.config import TokenizerConfig


_STAT_BATCHES = 0
_STAT_TOKENS = 1
_STAT_QUEUE_PUT_WAIT_MS = 2
_STAT_QUEUE_PUTS = 3
_STATS_FIELDS = 4


class AsyncPackedDataIterator:
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        tokenizer_cfg: TokenizerConfig | dict[str, Any],
        pipeline_cfg: PipelineConfig | dict[str, Any],
        seq_len: int,
        batch_size: int = 1,
        producer_workers: int = 2,
        queue_max_batches: int = 32,
        world_size: int = 1,
        rank: int = 0,
        validate_hashes: bool = True,
        shuffle_seed: int | None = None,
        parquet_batch_size: int = 8192,
        sources: list[dict[str, Any]] | None = None,
        manifest_meta: dict[str, Any] | None = None,
        mixer_temperature: float = 1.0,
        metrics_path: str | Path | None = None,
        metrics_interval_seconds: int = 30,
        emit_metrics: bool = True,
        exclude_uris: set[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.exclude_uris = exclude_uris
        self.tokenizer_cfg = _dump_model(tokenizer_cfg)
        self.pipeline_cfg = _dump_model(pipeline_cfg)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.producer_workers = producer_workers
        self.queue_max_batches = queue_max_batches
        self.world_size = world_size
        self.rank = rank
        self.validate_hashes = validate_hashes
        self.shuffle_seed = shuffle_seed
        self.parquet_batch_size = parquet_batch_size
        self.sources = sources or []
        self.manifest_meta = manifest_meta
        self.mixer_temperature = mixer_temperature
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        self.metrics_interval_seconds = metrics_interval_seconds
        self.emit_metrics = emit_metrics
        self._committed_worker_states: dict[int, dict[str, Any]] = {}
        self._loaded_state: dict[str, Any] | None = None

    def __iter__(self) -> Iterator[Batch]:
        ctx = mp.get_context("spawn")
        queue: mp.Queue[Any] = ctx.Queue(maxsize=self.queue_max_batches)
        stop = ctx.Event()
        worker_stats = ctx.Array("Q", self.producer_workers * _STATS_FIELDS, lock=False)
        metrics = _RankDataMetrics(
            path=self.metrics_path,
            interval_seconds=self.metrics_interval_seconds,
            rank=self.rank,
            world_size=self.world_size,
            producer_workers=self.producer_workers,
            queue_max_batches=self.queue_max_batches,
            worker_stats=worker_stats,
            enabled=self.emit_metrics and self.metrics_path is not None,
        )
        workers = [
            ctx.Process(
                target=_producer_main,
                args=(
                    self.manifest_path,
                    self.tokenizer_cfg,
                    self.pipeline_cfg,
                    self.seq_len,
                    self.batch_size,
                    self.producer_workers,
                    worker_id,
                    self.world_size,
                    self.rank,
                    self.validate_hashes,
                    self.shuffle_seed,
                    self.parquet_batch_size,
                    self.sources,
                    self.manifest_meta,
                    self.mixer_temperature,
                    queue,
                    stop,
                    worker_stats,
                    self._committed_worker_states.get(worker_id),
                    self.exclude_uris,
                ),
                daemon=True,
            )
            for worker_id in range(self.producer_workers)
        ]
        completed = 0
        try:
            for worker in workers:
                worker.start()
            while completed < self.producer_workers:
                try:
                    metrics.before_queue_get()
                    item = queue.get(timeout=1.0)
                    metrics.after_queue_get(queue, completed)
                except Empty:
                    metrics.on_queue_get_timeout(queue, completed, workers)
                    if not any(worker.is_alive() for worker in workers):
                        break
                    continue
                kind = item.get("kind")
                if kind == "error":
                    stop.set()
                    raise RuntimeError(f"Async producer failed in worker {item.get('worker_id')}:\n{item.get('traceback')}")
                if kind == "eof":
                    completed += 1
                    worker_id = int(item["worker_id"])
                    self._committed_worker_states[worker_id] = item.get("state", {})
                    metrics.maybe_emit(queue, completed, workers, force=True)
                    continue
                payload = item
                worker_id = int(payload["worker_id"])
                worker_state = payload.get("state")
                if worker_state is not None:
                    self._committed_worker_states[worker_id] = worker_state
                metrics.on_batch_consumed(int(payload["consumed_tokens"]), queue, completed, workers)
                yield Batch(
                    input_ids=torch.from_numpy(payload["input_ids"]),
                    document_ids=torch.from_numpy(payload["document_ids"]),
                    consumed_tokens=int(payload["consumed_tokens"]),
                )
        finally:
            stop.set()
            for worker in workers:
                if worker.is_alive():
                    worker.join(timeout=1.0)
                    if worker.is_alive():
                        worker.terminate()
            queue.close()
            queue.join_thread()
            metrics.close()

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": "async_tokenization",
            "manifest_path": str(self.manifest_path),
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "producer_workers": self.producer_workers,
            "queue_max_batches": self.queue_max_batches,
            "world_size": self.world_size,
            "rank": self.rank,
            "validate_hashes": self.validate_hashes,
            "shuffle_seed": self.shuffle_seed,
            "parquet_batch_size": self.parquet_batch_size,
            "tokenizer_cfg": self.tokenizer_cfg,
            "pipeline_cfg": self.pipeline_cfg,
            "metrics_path": str(self.metrics_path) if self.metrics_path is not None else None,
            "metrics_interval_seconds": self.metrics_interval_seconds,
            "committed_worker_states": dict(self._committed_worker_states),
            "loaded_state": self._loaded_state,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        if not sd:
            return
        self._loaded_state = sd
        committed = sd.get("committed_worker_states") or {}
        self._committed_worker_states = {int(k): v for k, v in committed.items()}


def _producer_main(
    manifest_path: Path,
    tokenizer_cfg: dict[str, Any],
    pipeline_cfg: dict[str, Any],
    seq_len: int,
    batch_size: int,
    producer_workers: int,
    worker_id: int,
    world_size: int,
    rank: int,
    validate_hashes: bool,
    shuffle_seed: int | None,
    parquet_batch_size: int,
    sources: list[dict[str, Any]],
    manifest_meta: dict[str, Any] | None,
    mixer_temperature: float,
    queue: mp.Queue[Any],
    stop: mp.Event,
    worker_stats: Any,
    resume_state: dict[str, Any] | None,
    exclude_uris: set[str] | None = None,
) -> None:
    try:
        tokenizer = load_tokenizer(TokenizerConfig.model_validate(tokenizer_cfg))
        manifest_meta_model = ManifestMeta.model_validate(manifest_meta) if isinstance(manifest_meta, dict) else manifest_meta
        stream = TrainingRecordStream(
            manifest_path=manifest_path,
            sources=sources,
            pipeline_cfg=PipelineConfig.model_validate(pipeline_cfg),
            world_size=world_size,
            rank=rank,
            num_workers=producer_workers,
            worker_id=worker_id,
            validate_hashes=validate_hashes,
            shuffle_seed=shuffle_seed,
            parquet_batch_size=parquet_batch_size,
            mixer_temperature=mixer_temperature,
            manifest_meta=manifest_meta_model,
            exclude_uris=exclude_uris,
        )
        records = iter(stream)
        packed = PackedDataIterator(
            records,
            tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            upstream_state_getter=stream.state_dict,
            upstream_state_loader=stream.load_state_dict,
            prefetch_records=0,
        )
        if resume_state:
            try:
                packed.load_state_dict(resume_state)
            except Exception as exc:
                queue.put(
                    {
                        "kind": "error",
                        "worker_id": worker_id,
                        "traceback": f"failed to restore worker {worker_id} state: {exc}",
                    }
                )
                return
        for batch in packed:
            if stop.is_set():
                break
            consumed_tokens = int(batch.consumed_tokens)
            payload = {
                "worker_id": worker_id,
                "input_ids": np.asarray(batch.input_ids.cpu().numpy(), dtype=np.int64),
                "document_ids": np.asarray(batch.document_ids.cpu().numpy(), dtype=np.int64),
                "consumed_tokens": consumed_tokens,
                "state": packed.state_dict(),
            }
            _add_worker_stat(worker_stats, worker_id, _STAT_BATCHES, 1)
            _add_worker_stat(worker_stats, worker_id, _STAT_TOKENS, consumed_tokens)
            put_start = time.monotonic()
            queue.put(payload)
            put_wait_ms = int((time.monotonic() - put_start) * 1000.0)
            _add_worker_stat(worker_stats, worker_id, _STAT_QUEUE_PUT_WAIT_MS, put_wait_ms)
            _add_worker_stat(worker_stats, worker_id, _STAT_QUEUE_PUTS, 1)
        queue.put({"kind": "eof", "worker_id": worker_id, "state": packed.state_dict()})
    except Exception:
        queue.put(
            {
                "kind": "error",
                "worker_id": worker_id,
                "traceback": traceback.format_exc(),
            }
        )


def _dump_model(value: TokenizerConfig | PipelineConfig | dict[str, Any]) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")  # type: ignore[no-any-return]
    return dict(value)


def _add_worker_stat(worker_stats: Any, worker_id: int, field: int, value: int) -> None:
    worker_stats[worker_id * _STATS_FIELDS + field] += value


def _read_worker_stats(worker_stats: Any, producer_workers: int) -> dict[str, int]:
    batches = 0
    tokens = 0
    queue_put_wait_ms = 0
    queue_puts = 0
    for worker_id in range(producer_workers):
        offset = worker_id * _STATS_FIELDS
        batches += int(worker_stats[offset + _STAT_BATCHES])
        tokens += int(worker_stats[offset + _STAT_TOKENS])
        queue_put_wait_ms += int(worker_stats[offset + _STAT_QUEUE_PUT_WAIT_MS])
        queue_puts += int(worker_stats[offset + _STAT_QUEUE_PUTS])
    return {
        "produced_batches": batches,
        "produced_tokens": tokens,
        "queue_put_wait_ms": queue_put_wait_ms,
        "queue_puts": queue_puts,
    }


class _RankDataMetrics:
    def __init__(
        self,
        *,
        path: Path | None,
        interval_seconds: int,
        rank: int,
        world_size: int,
        producer_workers: int,
        queue_max_batches: int,
        worker_stats: Any,
        enabled: bool,
    ) -> None:
        self.sink = JsonlSink(path) if enabled and path is not None else None
        self.interval_seconds = max(1, interval_seconds)
        self.rank = rank
        self.world_size = world_size
        self.producer_workers = producer_workers
        self.queue_max_batches = queue_max_batches
        self.worker_stats = worker_stats
        self.start_time = time.monotonic()
        self.last_emit_time = self.start_time
        self.last_produced_tokens = 0
        self.last_consumed_tokens = 0
        self.last_queue_get_wait_seconds = 0.0
        self.total_queue_get_wait_seconds = 0.0
        self.consumed_batches = 0
        self.consumed_tokens = 0
        self.queue_gets = 0
        self.queue_get_timeouts = 0
        self._queue_get_started_at: float | None = None

    def before_queue_get(self) -> None:
        self._queue_get_started_at = time.monotonic()

    def after_queue_get(self, queue: mp.Queue[Any], completed_workers: int) -> None:
        if self._queue_get_started_at is None:
            return
        waited = time.monotonic() - self._queue_get_started_at
        self._queue_get_started_at = None
        self.last_queue_get_wait_seconds = waited
        self.total_queue_get_wait_seconds += waited
        self.queue_gets += 1
        self.maybe_emit(queue, completed_workers, workers=None)

    def on_queue_get_timeout(self, queue: mp.Queue[Any], completed_workers: int, workers: list[mp.Process]) -> None:
        self.queue_get_timeouts += 1
        self.last_queue_get_wait_seconds = 1.0
        self.total_queue_get_wait_seconds += 1.0
        self.maybe_emit(queue, completed_workers, workers, force=True)

    def on_batch_consumed(self, tokens: int, queue: mp.Queue[Any], completed_workers: int, workers: list[mp.Process]) -> None:
        self.consumed_batches += 1
        self.consumed_tokens += tokens
        self.maybe_emit(queue, completed_workers, workers)

    def maybe_emit(
        self,
        queue: mp.Queue[Any],
        completed_workers: int,
        workers: list[mp.Process] | None,
        *,
        force: bool = False,
    ) -> None:
        if self.sink is None:
            return
        now = time.monotonic()
        if not force and now - self.last_emit_time < self.interval_seconds:
            return
        stats = _read_worker_stats(self.worker_stats, self.producer_workers)
        elapsed = max(1.0e-6, now - self.start_time)
        window_seconds = max(1.0e-6, now - self.last_emit_time)
        produced_delta = stats["produced_tokens"] - self.last_produced_tokens
        consumed_delta = self.consumed_tokens - self.last_consumed_tokens
        record = {
            "rank": self.rank,
            "world_size": self.world_size,
            "producer_workers_per_rank": self.producer_workers,
            "completed_workers": completed_workers,
            "alive_workers": _alive_workers(workers) if workers is not None else None,
            "queue_depth_batches": _safe_qsize(queue),
            "queue_max_batches": self.queue_max_batches,
            "produced_batches": stats["produced_batches"],
            "produced_tokens": stats["produced_tokens"],
            "consumed_batches": self.consumed_batches,
            "consumed_tokens": self.consumed_tokens,
            "producer_tokens_per_sec": produced_delta / window_seconds,
            "consumer_tokens_per_sec": consumed_delta / window_seconds,
            "producer_tokens_per_sec_total": stats["produced_tokens"] / elapsed,
            "consumer_tokens_per_sec_total": self.consumed_tokens / elapsed,
            "queue_gets": self.queue_gets,
            "queue_get_timeouts": self.queue_get_timeouts,
            "last_queue_get_wait_seconds": self.last_queue_get_wait_seconds,
            "avg_queue_get_wait_seconds": self.total_queue_get_wait_seconds / max(1, self.queue_gets + self.queue_get_timeouts),
            "queue_puts": stats["queue_puts"],
            "avg_queue_put_wait_seconds": (stats["queue_put_wait_ms"] / 1000.0) / max(1, stats["queue_puts"]),
        }
        self.sink.write(record)
        self.last_emit_time = now
        self.last_produced_tokens = stats["produced_tokens"]
        self.last_consumed_tokens = self.consumed_tokens

    def close(self) -> None:
        self.sink = None


def _safe_qsize(queue: mp.Queue[Any]) -> int | None:
    try:
        return int(queue.qsize())
    except (NotImplementedError, OSError):
        return None


def _alive_workers(workers: list[mp.Process] | None) -> int | None:
    if workers is None:
        return None
    return sum(1 for worker in workers if worker.is_alive())
