#!/usr/bin/env python3
"""错题重放SFT:把教师池 sft_ready.jsonl 的一段消费成 verl SFT messages-parquet。

组件③(检查点循环)的数据入口。每次重放SFT阶段调用:从 .sft_consumed_offset 起
取 up to --max 条,写成与 train_sft_s3r1_16k.parquet 同 schema 的 parquet
(messages / data_source / ability / extra_info),并推进 offset。

用法:
  python3 RL/build_replay_sft_parquet.py --pool-dir /data/zilu/fastrl/wrongpool_v3exp \
      --out /data/zilu/data_unified_v2/parquet/replay_sft_cycleK.parquet --max 4000

退出码:0=产出有效 parquet;3=池中无新数据(orchestrator 据此跳过本轮SFT)。
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def read_offset(p):
    try:
        return int(open(p).read().strip())
    except Exception:
        return 0


def write_offset(p, n):
    tmp = p + ".tmp"
    open(tmp, "w").write(str(n))
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default=os.environ.get("WRONGPOOL_DIR", "/data/zilu/fastrl/wrongpool_v3exp"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=4000, help="本轮最多消费多少条教师样本")
    ap.add_argument("--min", type=int, default=64, help="少于该数则跳过(返回3)")
    ap.add_argument("--no-advance", action="store_true", help="不推进 offset(dry-run)")
    args = ap.parse_args()

    ready = os.path.join(args.pool_dir, "sft_ready.jsonl")
    off_f = os.path.join(args.pool_dir, ".sft_consumed_offset")
    if not os.path.exists(ready):
        print(f"[replay-sft] no sft_ready.jsonl at {ready}", flush=True)
        sys.exit(3)

    start = read_offset(off_f)
    with open(ready) as f:
        lines = f.readlines()
    chunk = lines[start:start + args.max]
    rows = []
    bad = 0
    for ln in chunk:
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
            msgs = o["messages"]
            assert isinstance(msgs, list) and len(msgs) >= 2
            rows.append({
                "messages": msgs,
                "data_source": o.get("data_source"),
                "ability": o.get("ability", "math"),
                "extra_info": o.get("extra_info") if isinstance(o.get("extra_info"), str)
                else json.dumps(o.get("extra_info", {}), ensure_ascii=False),
            })
        except Exception:
            bad += 1

    if len(rows) < args.min:
        print(f"[replay-sft] only {len(rows)} usable (start={start}, chunk={len(chunk)}, bad={bad}); "
              f"< min {args.min}; skip", flush=True)
        sys.exit(3)

    table = pa.Table.from_pylist(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pq.write_table(table, args.out)
    new_off = start + len(chunk)
    if not args.no_advance:
        write_offset(off_f, new_off)
    print(f"[replay-sft] wrote {len(rows)} rows -> {args.out} | "
          f"offset {start}->{new_off} | bad={bad}", flush=True)


if __name__ == "__main__":
    main()
