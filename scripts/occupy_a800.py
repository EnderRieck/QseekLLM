#!/usr/bin/env python3
"""Reserve memory on visible NVIDIA A800 GPUs.

Examples:
  CUDA_VISIBLE_DEVICES=0 python scripts/occupy_a800.py --mem-frac 0.9
  python scripts/occupy_a800.py --gpus 0,1 --mem-gb 70
  python scripts/occupy_a800.py --gpus 0 --mem-frac 0.8 --burn

The script intentionally keeps allocated tensors alive until interrupted.
It only uses GPUs whose name contains "A800" by default. Use
`--require-name ""` if you intentionally want to occupy other visible GPUs.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass

import torch


BYTES_PER_GIB = 1024**3


@dataclass
class HeldGpu:
    index: int
    name: str
    tensors: list[torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated visible GPU indices to occupy. Default: all visible GPUs.",
    )
    parser.add_argument(
        "--mem-frac",
        type=float,
        default=0.9,
        help="Fraction of free memory to reserve per GPU when --mem-gb is not set.",
    )
    parser.add_argument(
        "--mem-gb",
        type=float,
        default=None,
        help="GiB to reserve per GPU. Overrides --mem-frac.",
    )
    parser.add_argument(
        "--chunk-gb",
        type=float,
        default=1.0,
        help="Allocation chunk size in GiB. Lower this if allocation is unstable.",
    )
    parser.add_argument(
        "--require-name",
        default="A800",
        help='Only occupy GPUs whose device name contains this string. Set to "" to disable.',
    )
    parser.add_argument(
        "--burn",
        action="store_true",
        help="Run a small matmul loop to keep some GPU utilization after reserving memory.",
    )
    parser.add_argument(
        "--burn-size",
        type=int,
        default=4096,
        help="Matrix size for --burn. Increase for higher utilization.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=5.0,
        help="Sleep interval for the idle hold loop.",
    )
    return parser.parse_args()


def selected_gpus(raw: str | None) -> list[int]:
    count = torch.cuda.device_count()
    if raw is None:
        return list(range(count))

    result = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0 or index >= count:
            raise ValueError(f"GPU index {index} is outside visible range 0..{count - 1}")
        result.append(index)
    return result


def reserve_memory(index: int, mem_gb: float | None, mem_frac: float, chunk_gb: float) -> HeldGpu:
    torch.cuda.set_device(index)
    name = torch.cuda.get_device_name(index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    target_bytes = int(mem_gb * BYTES_PER_GIB) if mem_gb is not None else int(free_bytes * mem_frac)
    target_bytes = max(0, min(target_bytes, free_bytes - 256 * 1024**2))
    chunk_bytes = max(1, int(chunk_gb * BYTES_PER_GIB))

    tensors: list[torch.Tensor] = []
    allocated = 0
    while allocated < target_bytes:
        this_chunk = min(chunk_bytes, target_bytes - allocated)
        numel = this_chunk // torch.empty((), dtype=torch.uint8, device=f"cuda:{index}").element_size()
        if numel <= 0:
            break
        try:
            tensors.append(torch.empty(numel, dtype=torch.uint8, device=f"cuda:{index}"))
            allocated += numel
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk_bytes <= 64 * 1024**2:
                break
            chunk_bytes //= 2

    held_gib = allocated / BYTES_PER_GIB
    total_gib = total_bytes / BYTES_PER_GIB
    print(f"GPU {index}: holding {held_gib:.2f} GiB on {name} / total {total_gib:.2f} GiB", flush=True)
    return HeldGpu(index=index, name=name, tensors=tensors)


def burn_loop(held: list[HeldGpu], size: int, sleep_sec: float) -> None:
    mats = {}
    for gpu in held:
        with torch.cuda.device(gpu.index):
            a = torch.randn((size, size), device=f"cuda:{gpu.index}", dtype=torch.float16)
            b = torch.randn((size, size), device=f"cuda:{gpu.index}", dtype=torch.float16)
            mats[gpu.index] = (a, b)

    while True:
        for gpu in held:
            with torch.cuda.device(gpu.index):
                a, b = mats[gpu.index]
                _ = a @ b
        time.sleep(sleep_sec)


def sleep_loop(sleep_sec: float) -> None:
    while True:
        time.sleep(sleep_sec)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available.", file=sys.stderr)
        return 1
    if args.mem_gb is None and not (0 < args.mem_frac < 1):
        print("--mem-frac must be between 0 and 1.", file=sys.stderr)
        return 1

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<all>")
    print(f"CUDA_VISIBLE_DEVICES={visible}", flush=True)

    stop = False

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True
        print(f"Received signal {signum}; exiting.", flush=True)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    held: list[HeldGpu] = []
    for index in selected_gpus(args.gpus):
        name = torch.cuda.get_device_name(index)
        if args.require_name and args.require_name not in name:
            print(f"GPU {index}: skip {name!r}; does not contain {args.require_name!r}", flush=True)
            continue
        held.append(reserve_memory(index, args.mem_gb, args.mem_frac, args.chunk_gb))

    if not held:
        print("No GPUs were occupied.", file=sys.stderr)
        return 2

    print("Holding GPU memory. Press Ctrl-C or send SIGTERM to stop.", flush=True)
    try:
        if args.burn:
            burn_loop(held, args.burn_size, args.sleep_sec)
        else:
            while not stop:
                time.sleep(args.sleep_sec)
    finally:
        held.clear()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
