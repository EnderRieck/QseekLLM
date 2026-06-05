from __future__ import annotations

import inspect
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def _get_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _build_mlp(dim: int, hidden: int, layers: int) -> nn.Module:
    modules: list[nn.Module] = [nn.Linear(dim, hidden), nn.GELU()]
    for _ in range(max(0, layers - 2)):
        modules.extend([nn.Linear(hidden, hidden), nn.GELU()])
    modules.append(nn.Linear(hidden, dim))
    return nn.Sequential(*modules)


def _wrap_ddp(model: nn.Module, device: torch.device, local_rank: int) -> nn.Module:
    kwargs: dict[str, object] = {}
    if device.type == "cuda":
        kwargs.update(device_ids=[local_rank], output_device=local_rank)
    signature = inspect.signature(DistributedDataParallel)
    if "init_sync" in signature.parameters:
        kwargs["init_sync"] = os.environ.get("TRAIN_SMOKE_DDP_INIT_SYNC", "1") not in {"0", "false", "False"}
    return DistributedDataParallel(model, **kwargs)


def _wrap_fsdp(model: nn.Module, device: torch.device) -> nn.Module:
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

    strategy_name = os.environ.get("TRAIN_SMOKE_FSDP_SHARDING", "full_shard").lower()
    if strategy_name in {"no_shard", "none"}:
        strategy = ShardingStrategy.NO_SHARD
    elif strategy_name in {"shard_grad_op", "grad"}:
        strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        strategy = ShardingStrategy.FULL_SHARD
    return FullyShardedDataParallel(model, sharding_strategy=strategy, device_id=device)


def _manual_average_grads(model: nn.Module, rank: int, step: int) -> None:
    world_size = dist.get_world_size()
    sync_before = os.environ.get("TRAIN_SMOKE_SYNC_BEFORE_COLLECTIVE", "0") in {"1", "true", "True"}
    for index, parameter in enumerate(model.parameters()):
        if parameter.grad is None:
            continue
        numel = parameter.grad.numel()
        mib = numel * parameter.grad.element_size() / 1024 / 1024
        print(
            "stage manual_grad_all_reduce_start",
            "rank", rank,
            "step", step,
            "param", index,
            "shape", tuple(parameter.grad.shape),
            "mib", f"{mib:.3f}",
            flush=True,
        )
        if sync_before and parameter.grad.is_cuda:
            torch.cuda.synchronize(parameter.grad.device)
        dist.all_reduce(parameter.grad)
        print(
            "stage manual_grad_all_reduce_done",
            "rank", rank,
            "step", step,
            "param", index,
            flush=True,
        )
        parameter.grad.div_(world_size)


def main() -> None:
    backend = os.environ.get("TRAIN_SMOKE_BACKEND", "nccl")
    mode = os.environ.get("TRAIN_SMOKE_MODE", "ddp").lower()
    timeout_seconds = _get_int("TRAIN_SMOKE_TIMEOUT", 120)
    local_rank = _get_int("LOCAL_RANK", 0)

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    print(
        "stage init_start",
        "backend", backend,
        "mode", mode,
        "rank_env", os.environ.get("RANK"),
        "local_rank", local_rank,
        flush=True,
    )
    dist.init_process_group(backend, timeout=timedelta(seconds=timeout_seconds))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print("stage init_done", "rank", rank, "world_size", world_size, "local_rank", local_rank, flush=True)

    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)

    dim = _get_int("TRAIN_SMOKE_DIM", 1024)
    hidden = _get_int("TRAIN_SMOKE_HIDDEN", 4096)
    layers = _get_int("TRAIN_SMOKE_LAYERS", 4)
    batch = _get_int("TRAIN_SMOKE_BATCH", 8)
    steps = _get_int("TRAIN_SMOKE_STEPS", 5)

    print("stage model_build_start", "rank", rank, "dim", dim, "hidden", hidden, "layers", layers, flush=True)
    model = _build_mlp(dim, hidden, layers).to(device)
    print("stage model_build_done", "rank", rank, flush=True)

    if os.environ.get("TRAIN_SMOKE_WARMUP_ALL_REDUCE", "0") in {"1", "true", "True"}:
        print("stage warmup_all_reduce_start", "rank", rank, flush=True)
        warmup = torch.ones(1, device=device)
        dist.all_reduce(warmup)
        print("stage warmup_all_reduce_done", "rank", rank, "sum", float(warmup.item()), flush=True)

    if mode == "ddp":
        print("stage wrap_ddp_start", "rank", rank, flush=True)
        train_model = _wrap_ddp(model, device, local_rank)
        print("stage wrap_ddp_done", "rank", rank, flush=True)
    elif mode == "fsdp":
        print("stage wrap_fsdp_start", "rank", rank, flush=True)
        train_model = _wrap_fsdp(model, device)
        print("stage wrap_fsdp_done", "rank", rank, flush=True)
    elif mode == "manual":
        train_model = model
    else:
        raise ValueError(f"Unsupported TRAIN_SMOKE_MODE: {mode}")

    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for step in range(steps):
        torch.manual_seed(10_000 + step)
        x = torch.randn(batch, dim, device=device)
        target = torch.randn(batch, dim, device=device)
        print("stage step_start", "rank", rank, "step", step, flush=True)
        optimizer.zero_grad(set_to_none=True)
        output = train_model(x)
        loss = loss_fn(output, target)
        print("stage backward_start", "rank", rank, "step", step, "loss", float(loss.detach().cpu()), flush=True)
        loss.backward()
        print("stage backward_done", "rank", rank, "step", step, flush=True)
        if mode == "manual":
            print("stage manual_all_reduce_start", "rank", rank, "step", step, flush=True)
            _manual_average_grads(train_model, rank, step)
            print("stage manual_all_reduce_done", "rank", rank, "step", step, flush=True)
        optimizer.step()
        print("stage step_done", "rank", rank, "step", step, flush=True)

    done = torch.tensor([1.0], device=device)
    dist.all_reduce(done)
    print("stage done", "rank", rank, "sum", float(done.item()), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
