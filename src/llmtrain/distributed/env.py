from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

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

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(cfg: DistributedConfig) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = _select_device(local_rank)
    enabled = cfg.backend in {"ddp", "fsdp", "zero"} and world_size > 1
    if enabled and not dist.is_initialized():
        dist_backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(
            backend=dist_backend,
            init_method=cfg.init_method,
            timeout=timedelta(seconds=cfg.timeout_seconds),
        )
    return DistributedContext(
        enabled=enabled,
        backend=cfg.backend,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _select_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")
