#!/usr/bin/env python3
"""按 source / difficulty 抽取完整样本，供人工核查 <think> 推演质量。

(合并了早期的 _qc_sample.py 与 _qc_sample2.py。)

用法:
  # 抽某个 source 的前 5 条 (精确匹配)
  python scripts/qc/inspect_samples.py --source arithmetic.fraction_division -n 5

  # 子串匹配 + 限定难度, 跨所有 split 找
  python scripts/qc/inspect_samples.py --match long_division --difficulty hard -n 2

  # 指定单个文件
  python scripts/qc/inspect_samples.py --file train/s2_fractions --match decimal -n 3
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

DEFAULT_DATA = "/data/zilu/fastrl/Compute_Cot/data"
ALL_FILES = [
    "train/s1_arithmetic", "train/s2_fractions", "train/s3_algebra",
    "train/s4_equations", "train/s5_broad", "val/val",
    "test/id_test", "test/extrap_ood", "test/template_ood",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--file", default="", help="只搜某个 split (如 train/s2_fractions)，默认全部")
    ap.add_argument("--source", default="", help="source 精确匹配")
    ap.add_argument("--match", default="", help="source 子串匹配 (与 --source 二选一)")
    ap.add_argument("--difficulty", default="", help="限定 easy/medium/hard")
    ap.add_argument("-n", "--n", type=int, default=5, help="抽取条数")
    args = ap.parse_args()

    data = Path(args.data)
    files = [args.file] if args.file else ALL_FILES
    found = []
    for fname in files:
        if len(found) >= args.n:
            break
        with (data / f"{fname}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if len(found) >= args.n:
                    break
                o = json.loads(line)
                src = o.get("source", "")
                diff = str(o.get("metadata", {}).get("difficulty", ""))
                if args.source and src != args.source:
                    continue
                if args.match and args.match not in src:
                    continue
                if args.difficulty and diff != args.difficulty:
                    continue
                found.append((fname, o))

    if not found:
        print("没有匹配的样本。检查 --source/--match/--difficulty。")
        return
    for fname, o in found:
        print("=" * 100)
        print(f"[{fname}] {o['source']} | difficulty={o.get('metadata',{}).get('difficulty')} | answer={o.get('answer')!r}")
        print("-" * 100)
        print("Q:", o["messages"][0]["content"])
        print("A:")
        print(o["messages"][1]["content"])
        print()


if __name__ == "__main__":
    main()
