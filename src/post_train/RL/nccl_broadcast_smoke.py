#!/usr/bin/env python3
"""Minimal Ray collective broadcast smoke test for local NCCL topology."""

from __future__ import annotations

import argparse
import os
import time

import ray
import ray.util.collective as collective
import torch


@ray.remote(num_gpus=1)
class BroadcastWorker:
    def __init__(self, group_name: str, rank: int, world_size: int, num_bytes: int):
        self.group_name = group_name
        self.rank = rank
        self.world_size = world_size
        self.num_bytes = num_bytes

    def init(self):
        collective.init_collective_group(self.world_size, self.rank, "nccl", self.group_name)
        collective.barrier(self.group_name)
        return {
            "rank": self.rank,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": torch.cuda.get_device_name(0),
        }

    def broadcast(self):
        if self.rank == 0:
            tensor = torch.arange(self.num_bytes, dtype=torch.uint8, device="cuda")
        else:
            tensor = torch.zeros(self.num_bytes, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        start = time.time()
        collective.broadcast(tensor, src_rank=0, group_name=self.group_name)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        checksum = int(tensor[: min(self.num_bytes, 1024)].sum().item())
        return {"rank": self.rank, "elapsed": elapsed, "checksum": checksum}

    def finalize(self):
        collective.destroy_collective_group(self.group_name)
        return self.rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--mb", type=int, default=512)
    parser.add_argument("--group-name", default="nccl_broadcast_smoke")
    args = parser.parse_args()

    env_vars = {
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
    }
    for name, value in os.environ.items():
        if name.startswith("NCCL_"):
            env_vars[name] = value

    ray.init(
        num_gpus=args.world_size,
        runtime_env={"env_vars": env_vars},
    )
    num_bytes = args.mb << 20
    workers = [
        BroadcastWorker.remote(args.group_name, rank, args.world_size, num_bytes)
        for rank in range(args.world_size)
    ]
    print(ray.get([w.init.remote() for w in workers]))
    print(ray.get([w.broadcast.remote() for w in workers]))
    print(ray.get([w.finalize.remote() for w in workers]))
    ray.shutdown()


if __name__ == "__main__":
    main()
