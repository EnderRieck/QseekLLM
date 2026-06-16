"""离线重导出已评完 step 的 heldout.md（用当前 write_eval_md 代码，探针全量展示）。

不碰 GPU、不影响正在跑的守护进程：只读 metrics.jsonl + 各 step 的 heldout.jsonl dump，重写 md。
用法: .venv/bin/python -m eval.regen_md [--dumps /data/zilu/fastrl/checkpoints/sft_foundation/eval_dumps]
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.async_eval import write_eval_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="/data/zilu/fastrl/checkpoints/sft_foundation/eval_dumps")
    args = ap.parse_args()
    with open(os.path.join(args.dumps, "metrics.jsonl"), encoding="utf-8") as f:
        ms = {m["step"]: m for m in map(json.loads, f)}
    for s, m in sorted(ms.items()):
        dump = os.path.join(args.dumps, f"step_{s}", "heldout.jsonl")
        if not os.path.exists(dump):
            print(f"step {s}: 无 dump, 跳过")
            continue
        with open(dump, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f]
        write_eval_md(recs, os.path.join(args.dumps, f"step_{s}", "heldout.md"), m)
        pm = [r for r in recs if r["source"] == "probe-math"]
        print(f"step {s}: md 重生成 ✓  probe-math {sum(r['correct'] for r in pm)}/{len(pm)} 对")


if __name__ == "__main__":
    main()
