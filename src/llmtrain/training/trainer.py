from __future__ import annotations

import math
import threading
import time
from contextlib import nullcontext
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from llmtrain.checkpointing.manager import CheckpointManager
from llmtrain.distributed.env import DistributedContext, barrier
from llmtrain.interfaces import Batch
from llmtrain.observability.callbacks import Phase1RunLogger
from llmtrain.training.schedule import TokenCosineScheduler
from llmtrain.training.state import TrainerState
from llmtrain.utils.config import Config


class Trainer:
    def __init__(
        self,
        *,
        cfg: Config,
        chain: list[dict[str, str]],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        data_iterator: Iterable[Batch],
        checkpoint_manager: CheckpointManager,
        manifest_metadata: dict[str, Any],
        tokenizer_metadata: dict[str, Any],
        device: torch.device,
        logger: Phase1RunLogger | None = None,
        distributed: DistributedContext | None = None,
        validation_callback: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.chain = chain
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_iterator = data_iterator
        self.checkpoint_manager = checkpoint_manager
        self.manifest_metadata = manifest_metadata
        self.tokenizer_metadata = tokenizer_metadata
        self.pad_token_id = _token_id_from_metadata(tokenizer_metadata, "<pad>", "pad_id")
        self.device = device
        self.logger = logger
        self.distributed = distributed
        self.validation_callback = validation_callback
        self.state = TrainerState()
        self.data_parallel_world_size = distributed.world_size if distributed is not None and distributed.enabled else 1
        self.global_micro_batch_size = cfg.trainer.micro_batch_size * self.data_parallel_world_size
        self.grad_accum_steps = max(1, math.ceil(cfg.trainer.global_batch_size / self.global_micro_batch_size))
        self._last_checkpoint_time = time.monotonic()
        self._train_start_time = self._last_checkpoint_time
        self._train_start_tokens = self.state.consumed_tokens
        self._last_log_tokens = 0
        self._last_log_time = self._last_checkpoint_time
        self._last_heartbeat_time = self._last_checkpoint_time
        self._last_milestone_bucket = 0
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._latest_loss: float | None = None

    def fit(self, *, resume_from: str | Path | None = None) -> TrainerState:
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        self._reset_progress_window()
        self.model.train()
        accum = 0
        loss_sum_tensor: torch.Tensor | None = None
        window_start = time.monotonic()
        self.optimizer.zero_grad(set_to_none=True)
        self._start_heartbeat_thread()
        try:
            self._sync_global_consumed_tokens()
            for batch in self.data_iterator:
                if self._done():
                    break
                batch = self._to_device(batch)
                with self._gradient_sync_context(accum):
                    with self._autocast():
                        output = self.model(
                            batch.input_ids,
                            document_ids=batch.document_ids,
                            pad_token_id=self.pad_token_id,
                        )
                    if output.loss is None:
                        raise FloatingPointError(f"Missing loss at step {self.state.global_step}")
                    loss = output.loss / self.grad_accum_steps
                    loss.backward()
                accum += 1
                loss_detached = output.loss.detach()
                loss_sum_tensor = loss_detached if loss_sum_tensor is None else loss_sum_tensor + loss_detached
                self.state.consumed_tokens += batch.consumed_tokens
                self.state.consumed_samples += int(batch.input_ids.shape[0])
                if accum >= self.grad_accum_steps:
                    avg_loss, grad_norm_val = self._finalize_step(loss_sum_tensor, accum)
                    self._sync_global_consumed_tokens()
                    self.scheduler.step(self.state.global_consumed_tokens)
                    self._latest_loss = avg_loss
                    self._log_step(avg_loss, grad_norm_val, window_start)
                    self._maybe_checkpoint(avg_loss)
                    self._maybe_validate()
                    accum = 0
                    loss_sum_tensor = None
                    window_start = time.monotonic()
            if accum > 0 and not self._done() and loss_sum_tensor is not None:
                avg_loss, grad_norm_val = self._finalize_step(loss_sum_tensor, accum)
                self._sync_global_consumed_tokens()
                self.scheduler.step(self.state.global_consumed_tokens)
                self._latest_loss = avg_loss
                self._log_step(avg_loss, grad_norm_val, window_start)
            if self.cfg.trainer.save_final_checkpoint:
                self._save_final_milestone()
                self.save_checkpoint("latest", metrics={"final": True})
            self._run_final_validation()
        finally:
            self._stop_heartbeat_thread()
        return self.state

    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        meta = self.checkpoint_manager.load(
            checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.state = TrainerState.from_state_dict(meta.get("trainer_state"))
        if hasattr(self.data_iterator, "load_state_dict"):
            self.data_iterator.load_state_dict(self._data_state_for_current_rank(meta.get("data_state", {})))
        if self.validation_callback is not None:
            self.validation_callback.load_state_dict(
                {"last_bucket": self.state.last_validation_bucket, "has_run_once": self.state.last_validation_bucket > 0}
            )
        if self._is_main and self.logger:
            self.logger.event("checkpoint_loaded", path=str(checkpoint_dir), global_step=self.state.global_step)

    def save_checkpoint(self, name: str, *, metrics: dict[str, Any] | None = None) -> Path:
        distributed = self.distributed is not None and self.distributed.enabled
        checkpoint_format = self.checkpoint_manager.resolve_format(distributed=distributed)
        if distributed and checkpoint_format == "torch":
            barrier()
            if not self._is_main:
                barrier()
                return self.checkpoint_manager.root / name
        path = self.checkpoint_manager.save(
            name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            trainer_state=self.state.state_dict(),
            data_state=self._checkpoint_data_state(distributed),
            cfg=self.cfg,
            chain=self.chain,
            manifest_metadata=self.manifest_metadata,
            tokenizer_metadata=self.tokenizer_metadata,
            metrics=metrics,
            distributed=distributed and checkpoint_format == "dcp",
            is_main=self._is_main,
        )
        if self._is_main:
            self._last_checkpoint_time = time.monotonic()
        if self._is_main and self.logger:
            self.logger.event("checkpoint_saved", path=str(path), global_step=self.state.global_step)
        if distributed and checkpoint_format == "torch":
            barrier()
        return path

    def _checkpoint_data_state(self, distributed: bool) -> dict[str, Any]:
        local_state = self.data_iterator.state_dict() if hasattr(self.data_iterator, "state_dict") else {}
        if not distributed or not (dist.is_available() and dist.is_initialized()):
            return local_state
        gathered: list[Any] | None = [None for _ in range(self.data_parallel_world_size)] if self._is_main else None
        dist.gather_object(local_state, object_gather_list=gathered, dst=0)
        if not self._is_main:
            return {}
        return {
            "mode": "distributed_data_state",
            "world_size": self.data_parallel_world_size,
            "rank_states": {str(rank): state for rank, state in enumerate(gathered or [])},
        }

    def _data_state_for_current_rank(self, data_state: dict[str, Any]) -> dict[str, Any]:
        if not data_state:
            return {}
        distributed = self.distributed is not None and self.distributed.enabled
        if data_state.get("mode") == "distributed_data_state":
            expected_world_size = self.data_parallel_world_size
            if data_state.get("world_size") != expected_world_size:
                raise RuntimeError(
                    "Distributed data checkpoint world_size mismatch: "
                    f"expected {expected_world_size}, got {data_state.get('world_size')}"
                )
            rank = self.distributed.rank if self.distributed is not None else 0
            rank_state = data_state.get("rank_states", {}).get(str(rank))
            if rank_state is None:
                raise RuntimeError(f"Missing data checkpoint state for rank {rank}")
            return rank_state
        if distributed and not self._is_main:
            return {}
        return data_state

    def _finalize_step(self, loss_sum_tensor: torch.Tensor, accum: int) -> tuple[float, float]:
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.grad_clip)
        if not isinstance(grad_norm, torch.Tensor):
            grad_norm = torch.tensor(float(grad_norm), device=loss_sum_tensor.device)
        avg_loss_tensor = (loss_sum_tensor / accum).detach()
        stacked = torch.stack([avg_loss_tensor.to(torch.float32), grad_norm.detach().to(torch.float32)]).cpu()
        avg_loss = float(stacked[0].item())
        grad_norm_val = float(stacked[1].item())
        if not math.isfinite(avg_loss):
            raise FloatingPointError(f"Non-finite loss at step {self.state.global_step}: {avg_loss}")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.global_step += 1
        self.state.optimizer_steps += 1
        return avg_loss, grad_norm_val

    def _to_device(self, batch: Batch) -> Batch:
        return Batch(
            input_ids=batch.input_ids.to(self.device, non_blocking=True),
            document_ids=batch.document_ids.to(self.device, non_blocking=True),
            consumed_tokens=batch.consumed_tokens,
        )

    def _autocast(self):
        if self.device.type != "cuda" or self.cfg.trainer.precision == "fp32":
            return torch.autocast(device_type=self.device.type, enabled=False)
        dtype = torch.bfloat16 if self.cfg.trainer.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _gradient_sync_context(self, accum: int):
        if not self.cfg.distributed.gradient_accumulation_no_sync:
            return nullcontext()
        if self.grad_accum_steps <= 1:
            return nullcontext()
        if self.distributed is None or not self.distributed.enabled:
            return nullcontext()
        is_sync_step = (accum + 1) >= self.grad_accum_steps
        if is_sync_step:
            return nullcontext()
        no_sync = getattr(self.model, "no_sync", None)
        if no_sync is None:
            return nullcontext()
        return no_sync()

    def _done(self) -> bool:
        if self.state.global_consumed_tokens >= self.cfg.trainer.max_tokens:
            return True
        return self.cfg.trainer.max_steps is not None and self.state.global_step >= self.cfg.trainer.max_steps

    def _log_step(self, loss: float, grad_norm: float, start_time: float) -> None:
        now = time.monotonic()
        elapsed = max(1.0e-6, now - start_time)
        local_tokens = self.state.global_consumed_tokens - self._last_log_tokens
        local_elapsed = max(1.0e-6, now - self._last_log_time)
        total_elapsed = max(1.0e-6, now - self._train_start_time)
        session_tokens = max(0, self.state.global_consumed_tokens - self._train_start_tokens)
        total_tokens_per_sec = session_tokens / total_elapsed
        tokens_remaining = max(0, self.cfg.trainer.max_tokens - self.state.global_consumed_tokens)
        progress_fraction = min(1.0, self.state.global_consumed_tokens / self.cfg.trainer.max_tokens)
        eta_seconds = tokens_remaining / total_tokens_per_sec if total_tokens_per_sec > 0 else None
        record = {
            "global_step": self.state.global_step,
            "consumed_tokens": self.state.global_consumed_tokens,
            "local_consumed_tokens": self.state.consumed_tokens,
            "max_tokens": self.cfg.trainer.max_tokens,
            "progress_fraction": progress_fraction,
            "progress_percent": progress_fraction * 100.0,
            "tokens_remaining": tokens_remaining,
            "eta_seconds": eta_seconds,
            "loss": loss,
            "lr": self.scheduler.get_lr(),
            "grad_norm": grad_norm,
            "step_seconds": elapsed,
            "tokens_per_sec": local_tokens / local_elapsed,
            "tokens_per_sec_total": total_tokens_per_sec,
            "samples_per_sec": self.global_micro_batch_size * self.grad_accum_steps / elapsed,
            "global_micro_batch_size": self.global_micro_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            **self._memory_stats(),
        }
        self._last_log_tokens = self.state.global_consumed_tokens
        self._last_log_time = now
        if self._is_main and self.logger:
            self.logger.metric(**record)
        if self._is_main and self.cfg.observability.console_interval_steps and self.state.global_step % self.cfg.observability.console_interval_steps == 0:
            print(
                f"step={self.state.global_step} "
                f"progress={record['progress_percent']:.2f}% "
                f"tokens={_format_count(self.state.global_consumed_tokens)}/{_format_count(self.cfg.trainer.max_tokens)} "
                f"eta={_format_duration(eta_seconds)} "
                f"loss={loss:.4f} lr={self.scheduler.get_lr():.3e} grad_norm={grad_norm:.3f} "
                f"tok/s={_format_rate(record['tokens_per_sec'])}"
            )

    def _reset_progress_window(self) -> None:
        now = time.monotonic()
        self._train_start_time = now
        self._last_log_time = now
        self._last_heartbeat_time = now
        self._train_start_tokens = self.state.global_consumed_tokens
        self._last_log_tokens = self.state.global_consumed_tokens

    def _start_heartbeat_thread(self) -> None:
        if not self._is_main or self.logger is None or not self.cfg.observability.heartbeat:
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="llmtrain-heartbeat", daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat_thread(self) -> None:
        if self._heartbeat_thread is None:
            return
        self._heartbeat_stop.set()
        self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        interval = max(1, self.cfg.observability.heartbeat_interval_seconds)
        while not self._heartbeat_stop.wait(interval):
            self._write_heartbeat()

    def _write_heartbeat(self) -> None:
        if not self._is_main or self.logger is None or not self.cfg.observability.heartbeat:
            return
        now = time.monotonic()
        total_elapsed = max(1.0e-6, now - self._train_start_time)
        session_tokens = max(0, self.state.global_consumed_tokens - self._train_start_tokens)
        tokens_remaining = max(0, self.cfg.trainer.max_tokens - self.state.global_consumed_tokens)
        progress_fraction = min(1.0, self.state.global_consumed_tokens / self.cfg.trainer.max_tokens)
        total_tokens_per_sec = session_tokens / total_elapsed
        eta_seconds = tokens_remaining / total_tokens_per_sec if total_tokens_per_sec > 0 else None
        record = {
            "global_step": self.state.global_step,
            "consumed_tokens": self.state.global_consumed_tokens,
            "local_consumed_tokens": self.state.consumed_tokens,
            "max_tokens": self.cfg.trainer.max_tokens,
            "progress_fraction": progress_fraction,
            "progress_percent": progress_fraction * 100.0,
            "tokens_remaining": tokens_remaining,
            "eta_seconds": eta_seconds,
            "loss": self._latest_loss,
            "lr": self.scheduler.get_lr(),
            "step_seconds": now - self._last_heartbeat_time,
            "tokens_per_sec_total": total_tokens_per_sec,
            "samples_per_sec": self.global_micro_batch_size * self.grad_accum_steps / max(1.0e-6, now - self._last_heartbeat_time),
            "global_micro_batch_size": self.global_micro_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
        }
        self.logger.beat(**record)
        self._last_heartbeat_time = now

    def _maybe_checkpoint(self, loss: float) -> None:
        monitor_value = self._sync_monitor_value(loss)
        metrics = {"loss": monitor_value, "global_step": self.state.global_step}
        checkpoint_names = self._checkpoint_plan(monitor_value)
        for name in checkpoint_names:
            self.save_checkpoint(name, metrics=metrics)

    def _maybe_validate(self) -> None:
        """Run in-training validation if a callback is registered.

        The callback owns the bucket trigger; this method just forwards the
        current global token / step counters. All ranks must enter together
        because FSDP-sharded forward needs every rank's participation.
        """
        if self.validation_callback is None:
            return
        self.validation_callback.maybe_run(
            model=self.model,
            global_consumed_tokens=self.state.global_consumed_tokens,
            global_step=self.state.global_step,
        )
        st = self.validation_callback.state_dict()
        self.state.last_validation_bucket = int(st.get("last_bucket", 0))

    def _run_final_validation(self) -> None:
        if self.validation_callback is None:
            return
        cfg = getattr(self.validation_callback, "cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return
        current_bucket = self.state.global_consumed_tokens // max(1, cfg.interval_tokens)
        if current_bucket <= self.state.last_validation_bucket and self.state.last_validation_bucket > 0:
            return
        self.validation_callback._run(
            model=self.model,
            global_consumed_tokens=self.state.global_consumed_tokens,
            global_step=self.state.global_step,
            trigger="final",
        )

    def _save_final_milestone(self) -> None:
        bucket = self.state.global_consumed_tokens // self.cfg.checkpoint.milestone_interval_tokens
        name: str | None = None
        if self._is_main and bucket > self._last_milestone_bucket:
            self._last_milestone_bucket = bucket
            name = f"milestone_{bucket * self.cfg.checkpoint.milestone_interval_tokens:012d}"
        if self.distributed is not None and self.distributed.enabled:
            payload: list[Any] = [name, self._last_milestone_bucket]
            dist.broadcast_object_list(payload, src=0)
            name = payload[0]
            self._last_milestone_bucket = int(payload[1])
        if name is not None:
            self.save_checkpoint(
                name,
                metrics={
                    "final": True,
                    "global_step": self.state.global_step,
                    "consumed_tokens": self.state.global_consumed_tokens,
                },
            )

    def _checkpoint_plan(self, monitor_value: float) -> list[str]:
        names: list[str] = []
        next_best_metric = self.state.best_metric
        next_milestone_bucket = self._last_milestone_bucket

        if self._is_main:
            if self.cfg.trainer.checkpoint_interval_steps and self.state.global_step % self.cfg.trainer.checkpoint_interval_steps == 0:
                names.append(f"latest_step_{self.state.global_step:08d}")
            minutes = (time.monotonic() - self._last_checkpoint_time) / 60.0
            if minutes >= self.cfg.checkpoint.save_interval_minutes:
                names.append(f"latest_step_{self.state.global_step:08d}")
            bucket = self.state.global_consumed_tokens // self.cfg.checkpoint.milestone_interval_tokens
            if bucket > self._last_milestone_bucket:
                next_milestone_bucket = bucket
                names.append(f"milestone_{bucket * self.cfg.checkpoint.milestone_interval_tokens:012d}")
            if self.cfg.checkpoint.save_best and self._is_better_metric(monitor_value, self.state.best_metric):
                next_best_metric = monitor_value
                names.append("best")
            names = list(dict.fromkeys(names))

        if self.distributed is not None and self.distributed.enabled:
            payload: list[Any] = [names, next_best_metric, next_milestone_bucket]
            dist.broadcast_object_list(payload, src=0)
            names = list(payload[0])
            next_best_metric = payload[1]
            next_milestone_bucket = int(payload[2])

        self.state.best_metric = next_best_metric
        self._last_milestone_bucket = next_milestone_bucket
        return names

    def _sync_monitor_value(self, value: float) -> float:
        if self.distributed is None or not self.distributed.enabled:
            return value
        tensor = torch.tensor([value], dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float((tensor / self.distributed.world_size).item())

    def _is_better_metric(self, value: float, best: float | None) -> bool:
        if best is None:
            return True
        if self.cfg.checkpoint.mode == "min":
            return value < best
        if self.cfg.checkpoint.mode == "max":
            return value > best
        raise ValueError(f"Unsupported checkpoint.mode: {self.cfg.checkpoint.mode}")

    @property
    def _is_main(self) -> bool:
        return self.distributed is None or self.distributed.is_main

    def _memory_stats(self) -> dict[str, int]:
        if not self.cfg.distributed.profile_memory or self.device.type != "cuda":
            return {}
        if self.state.global_step % self.cfg.distributed.profile_interval_steps != 0:
            return {}
        device = self.device
        return {
            "gpu_mem_allocated": int(torch.cuda.memory_allocated(device)),
            "gpu_mem_reserved": int(torch.cuda.memory_reserved(device)),
            "gpu_max_mem_allocated": int(torch.cuda.max_memory_allocated(device)),
            "gpu_max_mem_reserved": int(torch.cuda.max_memory_reserved(device)),
        }

    def _sync_global_consumed_tokens(self) -> None:
        if self.distributed is None or not self.distributed.enabled:
            self.state.global_consumed_tokens = self.state.consumed_tokens
            return
        value = torch.tensor([self.state.consumed_tokens], dtype=torch.long, device=self.device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        self.state.global_consumed_tokens = int(value.item())


def _format_count(value: int) -> str:
    units = ["", "K", "M", "B", "T"]
    n = float(value)
    for unit in units:
        if abs(n) < 1000.0 or unit == units[-1]:
            return f"{n:.2f}{unit}" if unit else str(int(n))
        n /= 1000.0
    return str(value)


def _format_rate(value: float) -> str:
    return f"{_format_count(int(value))}/s"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    total = max(0, int(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _token_id_from_metadata(metadata: dict[str, Any], token: str, direct_key: str) -> int | None:
    value = metadata.get(direct_key)
    if value is not None:
        return int(value)
    special_token_ids = metadata.get("special_token_ids")
    if isinstance(special_token_ids, dict):
        value = special_token_ids.get(token)
        if value is not None:
            return int(value)
    return None
