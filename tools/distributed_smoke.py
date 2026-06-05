from __future__ import annotations

import torch
import torch.distributed as dist
import os
from datetime import timedelta


def _numel_from_env() -> int:
    if "SMOKE_NUMEL" in os.environ:
        return int(os.environ["SMOKE_NUMEL"])
    mib = int(os.environ.get("SMOKE_MIB", "1"))
    return mib * 1024 * 1024 // 4


def main() -> None:
    backend = os.environ.get("SMOKE_BACKEND", "nccl")
    op = os.environ.get("SMOKE_OP", "all_reduce")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    print("stage init_start", "backend", backend, "rank_env", os.environ.get("RANK"), "local_rank", local_rank, flush=True)
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    print("stage init_done", "backend", backend, "rank", dist.get_rank(), "local_rank", local_rank, flush=True)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    numel = _numel_from_env()
    value = torch.full((numel,), float(rank + 1), device=device)
    print(
        "stage collective_start",
        "backend", backend,
        "op", op,
        "rank", rank,
        "local_rank", local_rank,
        "numel", numel,
        flush=True,
    )
    if op == "all_reduce":
        dist.all_reduce(value)
        checksum = float(value[: min(numel, 1024)].sum().item())
    elif op == "all_gather":
        output = torch.empty((world_size * numel,), device=device)
        dist.all_gather_into_tensor(output, value)
        checksum = float(output[: min(output.numel(), 1024)].sum().item())
    elif op == "reduce_scatter":
        output = torch.empty((numel,), device=device)
        input_tensor = torch.cat([value + peer for peer in range(world_size)])
        dist.reduce_scatter_tensor(output, input_tensor)
        checksum = float(output[: min(numel, 1024)].sum().item())
    else:
        raise ValueError(f"Unsupported SMOKE_OP: {op}")
    print(
        "backend", backend,
        "op", op,
        "rank", rank,
        "local_rank", local_rank,
        "checksum", checksum,
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
