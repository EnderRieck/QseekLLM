from __future__ import annotations

import torch
import torch.distributed as dist
import os
from datetime import timedelta


def main() -> None:
    backend = os.environ.get("SMOKE_BACKEND", "nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    print("stage init_start", "backend", backend, "rank_env", os.environ.get("RANK"), "local_rank", local_rank, flush=True)
    dist.init_process_group(backend, timeout=timedelta(seconds=60))
    print("stage init_done", "backend", backend, "rank", dist.get_rank(), "local_rank", local_rank, flush=True)
    value = torch.ones(1, device=device)
    print("stage all_reduce_start", "backend", backend, "rank", dist.get_rank(), "local_rank", local_rank, flush=True)
    dist.all_reduce(value)
    print("backend", backend, "rank", dist.get_rank(), "local_rank", local_rank, "sum", value.item(), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
