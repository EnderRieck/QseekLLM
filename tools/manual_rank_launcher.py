from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch heterogeneous distributed ranks on one node.")
    parser.add_argument("--rank-start", type=int, required=True)
    parser.add_argument("--local-world-size", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command to launch")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen[bytes]] = []

    def stop(signum: int, _frame: object) -> None:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 10
        for proc in procs:
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if proc.poll() is None:
                proc.kill()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    for local_rank in range(args.local_world_size):
        rank = args.rank_start + local_rank
        env = os.environ.copy()
        env.update(
            {
                "RANK": str(rank),
                "WORLD_SIZE": str(args.world_size),
                "LOCAL_RANK": str(local_rank),
                "LOCAL_WORLD_SIZE": str(args.local_world_size),
                "MASTER_ADDR": args.master_addr,
                "MASTER_PORT": str(args.master_port),
                "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            }
        )
        log_path = log_dir / f"rank{rank}.log"
        log_file = log_path.open("wb")
        proc = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        log_file.close()
        procs.append(proc)
        print(f"launched rank={rank} local_rank={local_rank} pid={proc.pid} log={log_path}", flush=True)

    failed = False
    for proc in procs:
        returncode = proc.wait()
        if returncode != 0:
            failed = True
            print(f"rank process pid={proc.pid} exited with {returncode}", file=sys.stderr, flush=True)

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
