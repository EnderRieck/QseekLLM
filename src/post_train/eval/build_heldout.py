"""构建过程评测 held-out（**能力分解版**，2026-06-10 重构）。

设计转向：过程评测 = 按"本阶段在训能力"分解的【窄 / 对齐 / 低方差】信号，
而非广覆盖的 benchmark 拼盘。保留并强化与训练分布对齐的源，砍掉语言/格式错配的旁观项。

谱系（每条标 ability + 在文档里归到某能力轴）：
  ①算术执行力（核心, in-dist）   : compute_cot 每子源 N 条 → 同时作"动态课程"per-source 信号种子
  ②应用题理解/泛化（英文）        : gsm8k(本域) + SVAMP(结构扰动) + GSM-Plus(对抗扰动, 测模板过拟合)
  ③中文迁移（次要, 旁观）         : cmath（数值判分可靠）
  ④难度探针（观察, 为阶段3铺路）  : competition-math（gold 用平衡括号 extract_boxed, 已修旧贪婪 bug）

砍掉（语言/格式/难度错配, 旧版旁观噪声）: cmmlu / gaokao-mathqa / gaokao-mathcloze / bbh。
判分复用（reward.py 无需改）:
  数值应用题 → style "gsm8k"（精确 + 1e-6 容差 + sympy 回退）
  符号/竞赛   → style "math_verify"（sympy 等价）
  算术分源    → style "compute_cot"

用法: python -m eval.build_heldout --out eval/heldout.jsonl
对比基线: python -m eval.build_heldout --out eval/heldout.jsonl --cc-per-source 8 --gsm8k 500 --gsm-plus 800
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
from collections import defaultdict, Counter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.format import make_prompt, extract_gsm8k_answer, extract_boxed
from data_pipeline.build import qhash

BENCH = "/data/zilu/fastrl/data/benchmark"
COMPUTE_COT_TEST = "/data/zilu/QseekLLM/src/post_train/Compute_Cot/data/clean/test/id_test.jsonl"
GSM8K_TEST = "/data/zilu/math_sft_raw/gsm8k/main/test-*.parquet"
# 2026-06-10 审计 P0-2: heldout 构建时对训练池做 qhash 过滤(svamp 曾经 orca 泄漏 63/300)
DEFAULT_TRAIN_POOLS = ["/data/zilu/data_unified_v2/train_sft.jsonl",
                       "/data/zilu/data_unified_v2/train_rl.jsonl",
                       "/data/zilu/data_unified_v2/train_general_sft.jsonl"]


def _load_train_hashes(paths):
    h = set()
    for p in paths:
        if not os.path.exists(p):
            print(f"  [warn] 训练池不存在,跳过: {p}")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                us = [m["content"] for m in o.get("prompt", []) if m.get("role") == "user"]
                if us:
                    h.add(qhash(us[-1]))
    return h


def rec(q, gt, style, source, difficulty, ability="math"):
    return {"prompt": make_prompt(q), "ground_truth": str(gt), "style": style,
            "source": source, "difficulty": str(difficulty), "ability": ability}


def _isnum(s) -> bool:
    """gold 是否为可数值判分的纯数（含负/小数/分数斜杠由判分端处理，这里只滤明显不可判的）。"""
    if s is None:
        return False
    t = str(s).strip().replace(",", "")
    try:
        float(t)
        return True
    except ValueError:
        return False


def _arrow_rows(path):
    from datasets import load_from_disk
    # 多 config 目录（cmmlu/bbh 的父目录不是 dataset）→ 逐子目录
    if not (os.path.exists(os.path.join(path, "dataset_dict.json"))
            or os.path.exists(os.path.join(path, "dataset_info.json"))):
        rows = []
        for sub in sorted(os.listdir(path)):
            subp = os.path.join(path, sub)
            if os.path.isdir(subp):
                rows.extend(_arrow_rows(subp))
        return rows
    ds = load_from_disk(path)
    rows = []
    for sp in (ds.values() if hasattr(ds, "values") else [ds]):
        rows.extend(list(sp))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="eval/heldout.jsonl")
    ap.add_argument("--cc-per-source", type=int, default=8)   # ①compute_cot 每子源（旧版仅2→纯噪声）
    ap.add_argument("--gsm8k", type=int, default=500)         # ②本域英文应用题（旧版200方差太大）
    ap.add_argument("--gsm-plus", type=int, default=800)      # ②对抗扰动，按8类扰动均衡采
    ap.add_argument("--competition", type=int, default=200)   # ④难度探针
    ap.add_argument("--cmath", type=int, default=200)         # ③中文迁移
    ap.add_argument("--no-probe", action="store_true", help="不并入 build_probe 的随手探针(12条)")
    ap.add_argument("--train-pools", default="", help="逗号分隔训练池 jsonl;空=DEFAULT_TRAIN_POOLS(v2)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = []
    stat = {}

    # ① compute_cot test：算术分源（worked 能力核心观测 + 课程信号种子）
    by_src = defaultdict(list)
    with open(COMPUTE_COT_TEST, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line); by_src[o["source"]].append(o)
    n0 = len(out)
    for src, rows in by_src.items():
        for o in rng.sample(rows, min(args.cc_per_source, len(rows))):
            out.append(rec(o["messages"][0]["content"], str(o.get("answer", "")), "compute_cot",
                           f"compute_cot:{src}", o.get("metadata", {}).get("difficulty", "")))
    stat["compute_cot"] = (len(out) - n0, len(by_src))

    # ② gsm8k 英文本域应用题
    try:
        import pyarrow.parquet as pq
        rows = [(r["question"], extract_gsm8k_answer(str(r.get("answer", ""))))
                for fp in glob.glob(GSM8K_TEST) for r in pq.read_table(fp).to_pylist()]
        rows = [(q, a) for q, a in rows if a]
        n0 = len(out)
        for q, a in rng.sample(rows, min(args.gsm8k, len(rows))):
            out.append(rec(q, a, "gsm8k", "gsm8k", "easy"))
        stat["gsm8k"] = (len(out) - n0,)
    except Exception as e:
        print("  gsm8k warn:", str(e)[:80])

    # ② SVAMP 结构扰动应用题（gold 数值，全量 300）
    try:
        rows = [(f"{r['Body']} {r['Question']}".strip(), str(r["Answer"]).strip(), r.get("Type", ""))
                for r in _arrow_rows(f"{BENCH}/svamp")
                if r.get("Body") and r.get("Question") and _isnum(r.get("Answer"))]
        n0 = len(out)
        for q, a, typ in rows:
            out.append(rec(q, a, "gsm8k", f"svamp:{typ}", typ))
        stat["svamp"] = (len(out) - n0,)
    except Exception as e:
        print("  svamp warn:", str(e)[:80])

    # ② GSM-Plus 对抗扰动（8 类扰动均衡采；滤掉 critical-thinking 的不可判 None；测模板过拟合）
    try:
        by_pt = defaultdict(list)
        for r in _arrow_rows(f"{BENCH}/gsm-plus"):
            if r.get("question") and _isnum(r.get("answer")):
                by_pt[r.get("perturbation_type", "")].append((r["question"], str(r["answer"]).strip()))
        per = max(1, args.gsm_plus // max(1, len(by_pt)))
        n0 = len(out)
        for pt, rows in by_pt.items():
            for q, a in rng.sample(rows, min(per, len(rows))):
                out.append(rec(q, a, "gsm8k", f"gsm-plus:{pt}", pt))
        stat["gsm-plus"] = (len(out) - n0, len(by_pt))
    except Exception as e:
        print("  gsm-plus warn:", str(e)[:80])

    # ③ cmath 中文应用题（数值判分；旁观中文迁移）
    try:
        rows = [(r["question"], r["golden"], r.get("grade", "")) for r in _arrow_rows(f"{BENCH}/cmath")
                if r.get("question") and r.get("golden") is not None]
        n0 = len(out)
        for q, g, grade in rng.sample(rows, min(args.cmath, len(rows))):
            out.append(rec(q, g, "math_verify", "cmath", f"grade{grade}"))
        stat["cmath"] = (len(out) - n0,)
    except Exception as e:
        print("  cmath warn:", str(e)[:80])

    # ④ competition-math 难度探针。2026-06-10: 抽样源从全量 MATH(12.5k 含 train)改为 **math-500**
    # (官方 test 子集,带干净 answer 字段) —— MATH-train 已解放回训练,必须只从 test 侧采样。
    try:
        rows = []
        for r in _arrow_rows(f"{BENCH}/math-500"):
            a = str(r.get("answer", "") or "").strip() or extract_boxed(str(r.get("solution", "")))
            if r.get("problem") and a:
                rows.append((r["problem"], a, r.get("level", ""), r.get("subject", "")))
        n0 = len(out)
        for q, a, lvl, typ in rng.sample(rows, min(args.competition, len(rows))):
            out.append(rec(q, a, "math_verify", f"competition-math:{typ}", f"Level {lvl}"))
        stat["competition-math"] = (len(out) - n0,)
    except Exception as e:
        print("  competition-math warn:", str(e)[:80])

    # —— 训练池 qhash 过滤(P0-2):任何与训练题面逐字重合的评测题剔除 ——
    train_h = _load_train_hashes(args.train_pools.split(",") if args.train_pools else DEFAULT_TRAIN_POOLS)
    if train_h:
        before = Counter(r["source"].split(":")[0] for r in out)
        out = [r for r in out
               if qhash([m["content"] for m in r["prompt"] if m["role"] == "user"][-1]) not in train_h]
        removed = before - Counter(r["source"].split(":")[0] for r in out)
        print(f"  训练池过滤: 剔除 {sum(removed.values())} 条 {dict(removed)}  (训练题面 {len(train_h):,})")

    # 随手探针：6 通用对话(无gold,自由题) + 6 自编数学(带gold)，每份报告自带 🆓 区
    if not args.no_probe:
        from eval.build_probe import records as probe_records
        pr = probe_records()
        out.extend(pr)
        stat["probe"] = (len(pr),)

    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter(r["source"].split(":")[0] for r in out)
    print(f"held-out（能力分解版）: {len(out)} 条 -> {args.out}")
    print("  顶层来源:", dict(c.most_common(20)))
    print("  采样明细:", stat)


if __name__ == "__main__":
    main()
