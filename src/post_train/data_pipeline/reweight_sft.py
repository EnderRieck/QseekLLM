"""为基础 SFT（阶段1-2）重新配权：易题为主、竞赛降权陪练、①算术地基为骨干。

按 source 上限重采（难题保留低比例，不删）。输出 messages-parquet 直接给 Verl SFT。
原则（hybrid_sft_rl_design.md §1.1/§3）：
- ①compute_cot worked 算术地基：**全留**（骨干，不稀释）。
- 易题(orca/gsm8k)：全留。
- 竞赛难题(numina/openr1/deepscaler)：**降权**到低比例（陪练）。
- metamath(GSM易+MATH难混)：限量，优先留 easy。
- 通用(防退化)：~22%，大源(tulutalk/infinity)限量。

用法: python -m data_pipeline.reweight_sft --out /data/zilu/data_unified/parquet/train_sft_foundation.parquet
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict, Counter

import pyarrow as pa
import pyarrow.parquet as pq

UNIFIED = "/data/zilu/data_unified_v2"   # 2026-06-10 起默认 v2 弹药库(审计修复版,见 docs/data_audit_report_20260610.md)

# F2 配方(training_plan v2 §4.2)。None=全留,0=剔除,正数=上限;easy_first=difficulty=easy 优先。
# 目标比例: compute_cot ~30% / 英文应用题易中 ~25% / 中文数学 ~15% / 竞赛陪练 ~5% / 通用 ~25%
FOUNDATION_CAPS = {
    # —— ①算术骨干 ——
    "compute_cot":          None,      # 全留(~394k)
    # —— 英文应用题(易中) ——
    "orca-math-200k":       None,      # 修复抽取后 ~157k 全留
    "gsm8k":                None,      # ~7.5k
    "metamathqa":           110000,    # easy(GSM段)优先
    "math-hendrycks":       None,      # MATH-train 7.5k,带官方 level(2026-06-10 新源)
    # —— 中文数学(新建,补 P0 缺口) ——
    "chinese-r1-math":      None,      # ~26k,R1 think 直接复用
    "calc-ape210k":         None,      # chain→中文 worked CoT ~153k
    # —— 竞赛陪练(少量见难题,全量留 S3) ——
    "numinamath-1.5":       60000,     # valid 双过滤后的 medium+hard
    "openr1-math-220k":     20000,     # R1 轨迹(超长部分被 8k 过滤再砍)
    "openthoughts3-math":   0,         # F2 剔除(70% 截断+中位 48k 字符,留 S3)
    "bespoke-stratos":      None,      # ~3.5k
    "deepscaler-preview":   None,      # ~5.7k
    # —— 通用(防退化;tulutalk 过滤器/dynamics 剔裸字母已在 adapter 生效) ——
    "tulutalk-annotated":   80000,
    "infinity-instruct":    80000,
    "chinese-deepseek-r1-distill": 80000,   # 通用段(数学 repos 已被数学池先占)
    "coig-cqia":            None,
    "dynamics-instruction-tuning": None,
    "dolly-15k":            8000,      # 审计:质量中等(短答/过时),降权
    "no_robots":            None,
    "infinity-math":        None,      # ~4k,显式入表(旧版靠默认全留,易误读)
}
EASY_FIRST = {"metamathqa"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{UNIFIED}/parquet/train_sft_foundation.parquet")
    ap.add_argument("--seed", type=int, default=20260610)
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rng = random.Random(args.seed)

    # 读两个 SFT 池，按 source 分桶（只保留有 gold 的）
    buckets = defaultdict(list)
    for path in [f"{UNIFIED}/train_sft.jsonl", f"{UNIFIED}/train_general_sft.jsonl"]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                if not o.get("gold_response"):
                    continue
                buckets[o["data_source"].split(":")[0]].append(o)

    # 按 cap 采样
    selected = []
    stats = Counter()
    for src, rows in buckets.items():
        cap = FOUNDATION_CAPS.get(src, None)
        if cap is None or len(rows) <= cap:
            chosen = rows
        elif src in EASY_FIRST:
            easy = [r for r in rows if r.get("extra_info", {}).get("difficulty") == "easy"]
            hard = [r for r in rows if r.get("extra_info", {}).get("difficulty") != "easy"]
            rng.shuffle(easy); rng.shuffle(hard)
            chosen = (easy + hard)[:cap]      # 优先 easy
        else:
            chosen = rng.sample(rows, cap)
        selected.extend(chosen)
        stats[src] = len(chosen)
    rng.shuffle(selected)

    # 写 messages-parquet
    def rows_iter():
        for o in selected:
            # 键序归一化:不同源的 prompt dict 键序不一,from_pylist 按键序推断 struct 字段序,批间不一致会写崩
            msgs = [{"role": m["role"], "content": m["content"]} for m in o["prompt"]]
            yield {"messages": msgs + [{"role": "assistant", "content": o["gold_response"]}],
                   "data_source": o["data_source"], "ability": o["ability"],
                   "extra_info": json.dumps(o.get("extra_info", {}), ensure_ascii=False)}

    writer = None
    buf = []
    n = 0
    for r in rows_iter():
        buf.append(r)
        if len(buf) >= 50000:
            t = pa.Table.from_pylist(buf)
            writer = writer or pq.ParquetWriter(args.out, t.schema)
            writer.write_table(t); n += len(buf); buf = []
    if buf:
        t = pa.Table.from_pylist(buf)
        writer = writer or pq.ParquetWriter(args.out, t.schema)
        writer.write_table(t); n += len(buf)
    if writer:
        writer.close()

    total = sum(stats.values())
    GENERAL = {"tulutalk-annotated", "infinity-instruct", "coig-cqia", "dynamics-instruction-tuning",
               "dolly-15k", "no_robots", "chinese-deepseek-r1-distill"}
    math_n = sum(v for k, v in stats.items() if k not in GENERAL)
    print(f"基础 SFT 重采: {total:,} 行  -> {args.out}")
    print(f"  数学 {math_n:,} ({100*math_n/total:.1f}%)  通用 {total-math_n:,} ({100*(total-math_n)/total:.1f}%)")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {k:28s} {v:>9,}  ({100*v/total:.1f}%)")


if __name__ == "__main__":
    main()
