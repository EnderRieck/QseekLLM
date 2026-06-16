#!/usr/bin/env python3
"""为 LLM 推演审计构造样本批次 (只读)。

每个 source 抽 K 条 (按难度铺开、按题面去重) 的完整样本，把 268 个 source 平均切成
N 批，每批写成一个易读的 .txt，供一个 subagent 通读评审。同时写 manifest.json。

用法:
  python scripts/qc/build_audit_batches.py [--data DIR] [--out DIR] [--per-source 4] [--batches 12]
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

DEFAULT_DATA = "/data/zilu/fastrl/Compute_Cot/data"
FILES = [  # broad/val 覆盖全部 268 source；stage 文件补充高频 source 的多样性
    "train/s5_broad", "val/val", "train/s1_arithmetic", "train/s2_fractions",
    "train/s3_algebra", "train/s4_equations", "test/id_test",
]


def collect(data: Path, per_source: int):
    buckets = defaultdict(list)       # source -> [sample, ...]
    seen_q = defaultdict(set)         # source -> {question}
    diff_count = defaultdict(lambda: defaultdict(int))
    for fname in FILES:
        p = data / f"{fname}.jsonl"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                src = o.get("source", "?")
                if len(buckets[src]) >= per_source:
                    continue
                q = o["messages"][0]["content"]
                if q in seen_q[src]:
                    continue
                d = str(o.get("metadata", {}).get("difficulty", "?"))
                # 尽量铺开难度: 同一难度最多取 ceil(per_source/3)+1
                if diff_count[src][d] > per_source // 3 + 1:
                    continue
                seen_q[src].add(q)
                diff_count[src][d] += 1
                buckets[src].append(o)
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default="/tmp/cc_audit")
    ap.add_argument("--per-source", type=int, default=4)
    ap.add_argument("--batches", type=int, default=12)
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    buckets = collect(data, args.per_source)
    sources = sorted(buckets)                      # 排序 → 同域相邻，批内主题集中
    B = args.batches
    batches = [sources[i::B] for i in range(B)]    # 轮转分配，使各域分散到不同批，规模均衡

    manifest = {}
    for bi, srcs in enumerate(batches):
        lines = []
        for src in srcs:
            for j, o in enumerate(buckets[src]):
                d = o.get("metadata", {}).get("difficulty", "?")
                lines.append(f"=== SOURCE: {src}  [sample {j+1}, difficulty={d}, answer={o.get('answer')!r}]")
                lines.append(f"Q: {o['messages'][0]['content']}")
                lines.append("A:")
                lines.append(o["messages"][1]["content"])
                lines.append("")
        (out / f"batch_{bi:02d}.txt").write_text("\n".join(lines), encoding="utf-8")
        manifest[f"batch_{bi:02d}"] = srcs

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    total_samples = sum(len(v) for v in buckets.values())
    print(f"sources={len(sources)}  samples={total_samples}  batches={B}  -> {out}")
    for bi, srcs in enumerate(batches):
        print(f"  batch_{bi:02d}: {len(srcs)} sources")


if __name__ == "__main__":
    main()
