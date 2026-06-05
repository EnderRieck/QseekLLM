from __future__ import annotations

import os
import pickle
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from llmtrain.distributed.config import DistributedConfig


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    backend: str
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    local_world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_multi_node(self) -> bool:
        return self.enabled and self.world_size > max(1, self.local_world_size)


_CHECKPOINT_BARRIER_GROUP: dist.ProcessGroup | None = None
_CHECKPOINT_BARRIER_COUNTER = 0
_CHECKPOINT_BARRIER_TIMEOUT_SECONDS = int(os.environ.get("LLMTRAIN_CHECKPOINT_BARRIER_TIMEOUT", "900"))


def init_distributed(cfg: DistributedConfig) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    device = _select_device(local_rank)
    enabled = cfg.backend in {"ddp", "fsdp", "zero"} and world_size > 1
    if enabled and not dist.is_initialized():
        dist_backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(
            backend=dist_backend,
            init_method=cfg.init_method,
            timeout=timedelta(seconds=cfg.timeout_seconds),
        )
        _init_checkpoint_barrier_group(world_size=world_size, local_world_size=local_world_size)
    return DistributedContext(
        enabled=enabled,
        backend=cfg.backend,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        local_world_size=local_world_size,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        global _CHECKPOINT_BARRIER_GROUP
        if _CHECKPOINT_BARRIER_GROUP is not None:
            dist.destroy_process_group(_CHECKPOINT_BARRIER_GROUP)
            _CHECKPOINT_BARRIER_GROUP = None
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def checkpoint_barrier(*, barrier_dir: str | Path | None = None, tag: str = "checkpoint") -> None:
    """Synchronize checkpoint save/load bookkeeping without using NCCL on multi-node runs."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    if _is_multi_node_process_group():
        if _CHECKPOINT_BARRIER_GROUP is not None:
            dist.barrier(group=_CHECKPOINT_BARRIER_GROUP)
            return
        if barrier_dir is not None:
            _file_barrier(Path(barrier_dir), tag)
            return
    barrier()


def checkpoint_gather_object(
    obj: object,
    *,
    object_gather_list: list[object | None] | None = None,
    dst: int = 0,
    barrier_dir: str | Path | None = None,
    tag: str = "checkpoint_gather",
) -> list[object | None] | None:
    """Gather Python checkpoint metadata without using NCCL on multi-node runs."""
    if not (dist.is_available() and dist.is_initialized()):
        return [obj] if dst == 0 else None
    if _is_multi_node_process_group():
        if _CHECKPOINT_BARRIER_GROUP is not None:
            dist.gather_object(
                obj,
                object_gather_list=object_gather_list,
                dst=dst,
                group=_CHECKPOINT_BARRIER_GROUP,
            )
            return object_gather_list
        if barrier_dir is not None:
            gathered = _file_gather_object(Path(barrier_dir), tag, obj, dst=dst)
            if dist.get_rank() == dst and object_gather_list is not None and gathered is not None:
                object_gather_list[:] = gathered
            return gathered
    dist.gather_object(obj, object_gather_list=object_gather_list, dst=dst)
    return object_gather_list


def checkpoint_process_group() -> dist.ProcessGroup | None:
    if dist.is_available() and dist.is_initialized() and _is_multi_node_process_group():
        return _CHECKPOINT_BARRIER_GROUP
    return None


def _init_checkpoint_barrier_group(*, world_size: int, local_world_size: int) -> None:
    global _CHECKPOINT_BARRIER_GROUP
    if world_size <= max(1, local_world_size):
        return
    if not dist.is_gloo_available():
        _warn_checkpoint_barrier("Gloo is unavailable; falling back to file/default checkpoint barrier.")
        return
    try:
        _CHECKPOINT_BARRIER_GROUP = dist.new_group(
            backend="gloo",
            timeout=timedelta(seconds=_CHECKPOINT_BARRIER_TIMEOUT_SECONDS),
        )
    except Exception as exc:  # pragma: no cover - depends on distributed backend build
        _CHECKPOINT_BARRIER_GROUP = None
        _warn_checkpoint_barrier(f"failed to initialize Gloo checkpoint barrier group: {exc}")


def _is_multi_node_process_group() -> bool:
    world_size = dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    return world_size > max(1, local_world_size)


def _file_barrier(root: Path, tag: str) -> None:
    global _CHECKPOINT_BARRIER_COUNTER
    _CHECKPOINT_BARRIER_COUNTER += 1
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    run_id = _barrier_run_id()
    safe_tag = _safe_barrier_tag(f"{_CHECKPOINT_BARRIER_COUNTER:06d}_{tag}")
    barrier_path = root / run_id / safe_tag
    barrier_path.mkdir(parents=True, exist_ok=True)
    (barrier_path / f"rank_{rank:08d}").write_text(
        f"host={socket.gethostname()} pid={os.getpid()} time={time.time()}\n",
        encoding="utf-8",
    )
    deadline = time.monotonic() + _CHECKPOINT_BARRIER_TIMEOUT_SECONDS
    while len(list(barrier_path.glob("rank_*"))) < world_size:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for checkpoint file barrier: {barrier_path}")
        time.sleep(0.2)
    release = barrier_path / "_release"
    if rank == 0:
        release.write_text("ok\n", encoding="utf-8")
    while not release.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for checkpoint file barrier release: {barrier_path}")
        time.sleep(0.2)


def _file_gather_object(root: Path, tag: str, obj: object, *, dst: int) -> list[object | None] | None:
    global _CHECKPOINT_BARRIER_COUNTER
    _CHECKPOINT_BARRIER_COUNTER += 1
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    run_id = _barrier_run_id()
    safe_tag = _safe_barrier_tag(f"{_CHECKPOINT_BARRIER_COUNTER:06d}_{tag}")
    gather_path = root / run_id / safe_tag
    gather_path.mkdir(parents=True, exist_ok=True)
    tmp_path = gather_path / f"rank_{rank:08d}.tmp"
    final_path = gather_path / f"rank_{rank:08d}.pkl"
    tmp_path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    tmp_path.replace(final_path)

    deadline = time.monotonic() + _CHECKPOINT_BARRIER_TIMEOUT_SECONDS
    while len(list(gather_path.glob("rank_*.pkl"))) < world_size:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for checkpoint file gather: {gather_path}")
        time.sleep(0.2)

    gathered: list[object | None] | None = None
    if rank == dst:
        gathered = [
            pickle.loads((gather_path / f"rank_{rank_id:08d}.pkl").read_bytes())
            for rank_id in range(world_size)
        ]
        (gather_path / "_release").write_text("ok\n", encoding="utf-8")

    release = gather_path / "_release"
    while not release.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for checkpoint file gather release: {gather_path}")
        time.sleep(0.2)
    return gathered


def _barrier_run_id() -> str:
    value = os.environ.get("LLMTRAIN_CHECKPOINT_BARRIER_RUN_ID") or os.environ.get("TORCHELASTIC_RUN_ID")
    if not value:
        value = f"{os.environ.get('MASTER_ADDR', 'local')}_{os.environ.get('MASTER_PORT', '0')}_{os.environ.get('WORLD_SIZE', '1')}"
    return _safe_barrier_tag(value)


def _safe_barrier_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_") or "barrier"


def _warn_checkpoint_barrier(message: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[llmtrain] {message}", file=sys.stderr, flush=True)


def _select_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")
