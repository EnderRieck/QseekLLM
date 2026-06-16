"""heldout v2 = v1(能力分解版,3881 条)+ 高考/考研档观测层(S4 难题退火阶段配套)。

新增谱系(ability=gaokao,全部中文为主):
  ⑤a gaokao-cloze   : AGIEval 高考数学填空 118 题(外部基准,style math_verify)
  ⑤b gaokao-mathqa  : AGIEval 高考数学选择 351 题(外部基准,style mcq)
  ⑤c cnk12-heldout  : 清洗池 numina:cn_k12 留出 200(in-dist 吸收度信号,英文)
  ⑤d zhmath-heldout : 清洗池 EduChat-Math 120 + applied_math 40(中文教育题)
  ⑤e kaoyan-heldout : 清洗池 zhr1:kaoyan 留出 60(考研档,池中仅 338,余量进训练)
  ⑤f advmath-heldout: 清洗池 zhr1:Advanced-Math 留出 40(高数档 hard)

池内留出的题写入 exclude 文件(qhash),构建 S4 训练 parquet 时必须传
--exclude-hashes 隔离,防训练-评测重合。

用法:
  python -m eval.build_heldout_v2 \
    --v1 eval/heldout.jsonl --pool /data/zilu/data_unified_v2/train_sft_s4clean.jsonl \
    --out eval/heldout_v2.jsonl --exclude-out eval/heldout_v2_exclude.txt
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.build import qhash
from data_pipeline.format import make_prompt

BENCH = "/data/zilu/fastrl/data/benchmark"
# 池内留出配额: extra_info.source -> (heldout source 名, 条数)
POOL_SLICES = {
    "numina:cn_k12": ("cnk12-heldout", 200),
    "zhr1:EduChat-Math": ("zhmath-heldout", 120),
    "zhr1:applied_math": ("zhmath-heldout", 40),
    # zhr1:kaoyan 已弃用:考研全科混题(史/政/化/生/计算机/心理学),真数学仅~10%,
    #   作数学 bench 价值不大(2026-06-12 删,见 docs/s4_data_report §8)。
    "zhr1:Advanced-Math": ("advmath-heldout", 40),
}


def rec(prompt, gt, style, source, difficulty, ability="gaokao"):
    return {"prompt": prompt, "ground_truth": str(gt), "style": style,
            "source": source, "difficulty": str(difficulty), "ability": ability}


def load_agieval():
    from datasets import load_from_disk
    out = []
    ds = load_from_disk(f"{BENCH}/agieval-gaokao-mathcloze")["test"]
    for r in ds:
        q = r["query"].strip()
        q = q.removeprefix("问题：").removesuffix("答案：").strip()
        out.append(rec(make_prompt(q), r["answer"], "math_verify", "gaokao-cloze", "gaokao"))
    ds = load_from_disk(f"{BENCH}/agieval-gaokao-mathqa")["test"]
    for r in ds:
        q = r["query"].strip().removeprefix("问题：")
        q = q[: q.rfind("答案：")].strip() if "答案：" in q else q
        gold = "ABCD"[int(r["gold"][0])]
        out.append(rec(make_prompt(q), gold, "mcq", "gaokao-mathqa", "gaokao"))
    return out


def draw_pool_slices(pool_path, seed):
    rng = random.Random(seed)
    cand = defaultdict(list)
    with open(pool_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            src = (o.get("extra_info") or {}).get("source", "")
            if src in POOL_SLICES:
                cand[src].append(o)
    out, excl = [], []
    for src, (name, n) in POOL_SLICES.items():
        rows = cand[src]
        rng.shuffle(rows)
        for o in rows[:n]:
            rm = o["reward_model"]
            diff = (o.get("extra_info") or {}).get("difficulty", "")
            out.append(rec(o["prompt"], rm["ground_truth"], rm["style"], name, diff))
            us = [m["content"] for m in o["prompt"] if m["role"] == "user"]
            excl.append(qhash(us[-1]))
        print(f"  {name:18s} <- {src}: 抽 {min(n, len(rows))}/{len(rows)}")
    return out, excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="eval/heldout.jsonl")
    ap.add_argument("--pool", default="/data/zilu/data_unified_v2/train_sft_s4clean.jsonl")
    ap.add_argument("--out", default="eval/heldout_v2.jsonl")
    ap.add_argument("--exclude-out", default="eval/heldout_v2_exclude.txt")
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()

    v1 = [json.loads(l) for l in open(args.v1, encoding="utf-8")]
    print(f"v1 heldout: {len(v1)}")
    agi = load_agieval()
    print(f"AGIEval 高考层: {len(agi)}")
    pool_rows, excl = draw_pool_slices(args.pool, args.seed)
    print(f"池内留出层: {len(pool_rows)}")

    allrows = v1 + agi + pool_rows
    with open(args.out, "w", encoding="utf-8") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.exclude_out, "w", encoding="utf-8") as f:
        f.write("\n".join(excl) + "\n")
    print(f"-> {args.out}: {len(allrows)} 条")
    print(f"-> {args.exclude_out}: {len(excl)} 条排除 hash")
    print(Counter(r["source"] for r in agi + pool_rows))


if __name__ == "__main__":
    main()
