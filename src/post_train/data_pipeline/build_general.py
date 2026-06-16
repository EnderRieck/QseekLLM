"""构建通用 SFT 池（③防退化），输出 train_general_sft.jsonl。

小源全量、大源(tulutalk/infinity)限量，按题面去重。与数学 SFT 池分文件，
训练时按比例混（防中文/语言退化），math:general 比例由 dataloader 控。
"""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter

from .general_adapters import GENERAL_SOURCES, iter_general
from .build import qhash, load_eval_hashes

# 默认每源上限（0=全量）。tulutalk/infinity 巨大，限量防通用压过数学。
DEFAULT_CAPS = {
    "no_robots": 0, "dolly": 0, "coig-cqia": 0, "dynamics": 0,
    "tulutalk": 150000, "infinity": 150000, "chinese-r1": 80000,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data/zilu/data_unified")
    ap.add_argument("--caps", default="", help="覆盖默认上限，如 tulutalk=100000,infinity=100000")
    ap.add_argument("--math-pool", default="", help="数学 SFT 池 jsonl 路径(默认 <out>/train_sft.jsonl)；其题面并入 seen,消同题双格式冲突")
    args = ap.parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    caps = dict(DEFAULT_CAPS)
    for kv in args.caps.split(","):
        if "=" in kv:
            k, v = kv.split("="); caps[k.strip()] = int(v)

    os.makedirs(args.out, exist_ok=True)
    # 2026-06-10 审计修复：①通用池同样做评测隔离(此前缺失,实测漏过 5 条评测题);
    # ②seen 以数学池题面打底(数学池优先)——消"同题在数学池教 think/boxed、在通用池教纯对话"的 7,256 条格式冲突。
    print("加载评测题面(隔离防泄漏)...", flush=True)
    eval_hashes = load_eval_hashes()
    print(f"  评测题面 {len(eval_hashes):,} 条", flush=True)
    seen = set()
    math_pool = args.math_pool or os.path.join(args.out, "train_sft.jsonl")
    if os.path.exists(math_pool):
        with open(math_pool, encoding="utf-8") as mf:
            for line in mf:
                o = json.loads(line)
                us = [m["content"] for m in o.get("prompt", []) if m.get("role") == "user"]
                if us:
                    seen.add(qhash(us[-1]))
        print(f"  数学池题面 {len(seen):,} 条已并入 seen", flush=True)
    else:
        print(f"  [warn] 数学池不存在: {math_pool}（跨池去重未生效）", flush=True)
    stats = Counter()
    dup = Counter()
    leak = Counter()
    with open(os.path.join(args.out, "train_general_sft.jsonl"), "w", encoding="utf-8") as f:
        for name in GENERAL_SOURCES:
            cap = caps.get(name, 0) or None
            kept = 0
            for rec in iter_general(name):
                h = qhash(rec.question)
                if h in eval_hashes:
                    leak[name] += 1
                    continue
                if h in seen:
                    dup[name] += 1
                    continue
                seen.add(h)
                f.write(json.dumps(rec.to_json_obj(), ensure_ascii=False) + "\n")
                stats[name] += 1
                kept += 1
                if cap and kept >= cap:
                    break
            print(f"  [{name}] kept={stats[name]} dup={dup[name]} leak={leak[name]}", flush=True)
    total = sum(stats.values())
    json.dump({"general_total": total,
               "per_source": {n: {"kept": stats[n], "dup": dup[n], "leak": leak[n]} for n in GENERAL_SOURCES}},
              open(os.path.join(args.out, "manifest_general.json"), "w"), ensure_ascii=False, indent=2)
    print("=" * 50, flush=True)
    print(f"通用 SFT 池: {total:,}  -> {args.out}/train_general_sft.jsonl", flush=True)


if __name__ == "__main__":
    main()
