"""S3 · 难度课程 SFT 切片(training_plan v2 §4.3,R1 轮)。

与 F2 的 reweight_sft 的本质区别:**按 token 预算配比**(不是条数)——
池内 token 极度偏斜(ot3 均长 46k 字符 vs ape210k 123),条数配比会把坡度冲掉。

逻辑:
  1. 难度归一:easy/medium/hard ∪ hendrycks Level 1-5 ∪ openr1 cc=N(correctness_count,
     越小越难:cc≥4→easy / cc=2-3→medium / cc=1→hard)。无标签者剔除(~9k,记数)。
  2. token 预算:每源抽样 ~300 条真实 tokenize 校准 chars/token,估算逐条 token。
  3. 桶内分配:难度桶 token 配额(默认 60:30:10)+ 通用防退化桶;
     单源 ≤ CAP(默认 40%) 防长轨迹源淹桶;floor:compute_cot ≥10%/桶、中文数学 ≥15%/桶
     (小比例贯穿防遗忘);通用桶内中文 ≥40%(CJK 字符占比判定)。
  4. 精确过滤:选中行真实 tokenize,> max-len 丢弃(打印按源统计),输出 messages-parquet
     (schema 与 F2 的 train_sft_foundation_8k.parquet 一致)+ manifest。

用法:
  python -m data_pipeline.reweight_s3 \
    --out /data/zilu/data_unified_v2/parquet/train_sft_s3r1_16k.parquet \
    --total-tokens 620000000 --ratio 60:30:10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

MATH_POOL = "/data/zilu/data_unified_v2/train_sft.jsonl"
GENERAL_POOL = "/data/zilu/data_unified_v2/train_general_sft.jsonl"
TOKENIZER = "/data/zilu/QseekLLM/src/llmtrain/qseek_digitsplit_base"

ZH_MATH_PREFIX = ("calc-ape210k", "chinese-r1-math")
CC_PREFIX = "compute_cot"
_CJK = re.compile(r"[一-鿿]")
_LEVEL = re.compile(r"Level (\d)")


def norm_bucket(diff: str) -> str | None:
    d = str(diff or "")
    if d in ("easy", "medium", "hard"):
        return d
    m = _LEVEL.match(d)
    if m:
        n = int(m.group(1))
        return "easy" if n <= 2 else "medium" if n == 3 else "hard"
    if d.startswith("cc="):
        try:
            n = int(d[3:])
        except ValueError:
            return None
        return "easy" if n >= 4 else "medium" if n >= 2 else "hard"
    return None


def row_text(o: dict) -> str:
    return "\n".join(m["content"] for m in o["prompt"]) + "\n" + (o.get("gold_response") or "")


def is_zh(o: dict) -> bool:
    t = row_text(o)[:400]
    return len(_CJK.findall(t)) >= 0.15 * max(len(t), 1)


def allocate(avail: dict[str, float], budget: float, cap_frac: float,
             floors: dict[str, float]) -> dict[str, float]:
    """桶内按源分配 token 配额:floor 保底 → 余量按可得量比例 → 单源 ≤ cap_frac*budget。"""
    cap = cap_frac * budget
    quota = {s: min(floors.get(s, 0.0) * budget, avail[s], cap) for s in avail}
    for _ in range(20):
        rem = budget - sum(quota.values())
        if rem <= budget * 0.005:
            break
        room = {s: min(avail[s], cap) - quota[s] for s in avail}
        room = {s: r for s, r in room.items() if r > 0}
        if not room:
            break
        w = sum(room.values())
        for s, r in room.items():
            quota[s] += min(r, rem * r / w)
    return {s: q for s, q in quota.items() if q > 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-tokens", type=int, default=620_000_000)
    ap.add_argument("--ratio", default="60:30:10", help="easy:medium:hard(token 口径)")
    ap.add_argument("--general-frac", type=float, default=0.15)
    ap.add_argument("--cap-frac", type=float, default=0.40, help="桶内单源 token 上限占比")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=20260612)
    ap.add_argument("--math-pool", default=MATH_POOL, help="数学池 jsonl(S4 用清洗后的池)")
    ap.add_argument("--general-pool", default=GENERAL_POOL)
    ap.add_argument("--exclude-hashes", default="", help="qhash 清单文件(每行一个),命中题面跳过(heldout 隔离)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    math_pool, general_pool = args.math_pool, args.general_pool
    excl = set()
    if args.exclude_hashes:
        from .build import qhash
        with open(args.exclude_hashes, encoding="utf-8") as f:
            excl = {ln.strip() for ln in f if ln.strip()}
        print(f"[excl] 载入排除题面 hash: {len(excl)}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

    e, m, h = (float(x) for x in args.ratio.split(":"))
    tot = e + m + h
    math_budget = args.total_tokens * (1 - args.general_frac)
    budgets = {"easy": math_budget * e / tot, "medium": math_budget * m / tot,
               "hard": math_budget * h / tot, "general": args.total_tokens * args.general_frac}

    # ---------- pass A: 扫描池,记录 (行号, 源, 桶, 字符数, zh) + 每源校准样本 ----------
    print("[A] 扫描池 + 校准 chars/token ...", flush=True)
    index = defaultdict(lambda: defaultdict(list))  # bucket -> source -> [(lineno, chars)]
    calib_texts = defaultdict(list)                 # source -> sample texts
    nolabel = Counter()
    for pool, is_math in [(math_pool, True), (general_pool, False)]:
        with open(pool, encoding="utf-8") as f:
            for ln, line in enumerate(f):
                o = json.loads(line)
                if not o.get("gold_response"):
                    continue
                if excl:
                    from .build import qhash
                    us = [mm["content"] for mm in o.get("prompt", []) if mm.get("role") == "user"]
                    if us and qhash(us[-1]) in excl:
                        continue
                src = o["data_source"].split(":")[0]
                if is_math:
                    b = norm_bucket((o.get("extra_info") or {}).get("difficulty"))
                    if b is None:
                        nolabel[src] += 1
                        continue
                else:
                    b = "general"
                    src = ("zh:" if is_zh(o) else "en:") + src
                t = row_text(o)
                index[b][src].append((ln, len(t)))
                cs = calib_texts[src]
                if len(cs) < 300:
                    cs.append(t)
                elif rng.random() < 0.01:
                    cs[rng.randrange(300)] = t
    print(f"  无难度标签剔除: {dict(nolabel)}", flush=True)

    cpt = {}  # chars per token
    for src, texts in calib_texts.items():
        ids = tok(texts, add_special_tokens=True).input_ids
        cpt[src] = sum(len(t) for t in texts) / max(sum(len(i) for i in ids), 1)

    # ---------- 桶内配额 + 抽样 ----------
    selected = {}  # lineno_key -> (bucket, src);  key = (pool_id, lineno)
    plan = defaultdict(dict)
    for b, srcs in index.items():
        avail = {s: sum(c for _, c in v) / cpt[s] for s, v in srcs.items()}
        floors = {}
        if b != "general":
            floors = {s: 0.10 for s in avail if s.startswith(CC_PREFIX)}
            zh_srcs = [s for s in avail if s.startswith(ZH_MATH_PREFIX)]
            for s in zh_srcs:
                floors[s] = max(floors.get(s, 0), 0.15 / len(zh_srcs))
        else:
            zh_av = sum(v for s, v in avail.items() if s.startswith("zh:"))
            zh_floor_total = min(0.40, zh_av / budgets[b] if budgets[b] else 0)
            zh_srcs = [s for s in avail if s.startswith("zh:")]
            for s in zh_srcs:
                floors[s] = zh_floor_total * (avail[s] / zh_av) if zh_av else 0
        quota = allocate(avail, budgets[b], args.cap_frac, floors)
        pool_id = 0 if b != "general" else 1
        for s, q in quota.items():
            items = srcs[s][:]
            rng.shuffle(items)
            got = 0.0
            for ln, chars in items:
                if got >= q:
                    break
                est = chars / cpt[s]
                if est > args.max_len * 1.15:   # 显超长的不选,省得精确段白算
                    continue
                selected[(pool_id, ln)] = (b, s)
                got += est
            plan[b][s] = (got, q)

    print("\n[B] 配额计划 (估算 token / 目标):")
    for b in ["easy", "medium", "hard", "general"]:
        bt = sum(g for g, _ in plan[b].values())
        print(f"  == {b}: {bt/1e6:.1f}M (目标 {budgets[b]/1e6:.1f}M)")
        for s, (g, q) in sorted(plan[b].items(), key=lambda kv: -kv[1][0]):
            print(f"     {s:38s} {g/1e6:7.1f}M / 配额 {q/1e6:7.1f}M")

    # ---------- pass B: 重读选中行,精确 tokenize,过滤,写 parquet ----------
    print("\n[C] 精确 tokenize + 写 parquet ...", flush=True)
    writer = None
    stats = defaultdict(lambda: [0, 0])   # (bucket,src) -> [rows, tokens]
    dropped = Counter()
    buf_rows, buf_meta = [], []

    def flush_buf():
        nonlocal writer, buf_rows, buf_meta
        if not buf_rows:
            return
        texts = ["\n".join(mm["content"] for mm in r["messages"]) for r in buf_rows]
        lens = [len(i) for i in tok(texts, add_special_tokens=True).input_ids]
        keep = []
        for r, (b, s), L in zip(buf_rows, buf_meta, lens):
            if L > args.max_len:
                dropped[s] += 1
                continue
            st = stats[(b, s)]
            st[0] += 1
            st[1] += L
            keep.append(r)
        if keep:
            t = pa.Table.from_pylist(keep)
            writer = writer or pq.ParquetWriter(args.out, t.schema)
            writer.write_table(t)
        buf_rows, buf_meta = [], []

    for pool_id, pool in [(0, math_pool), (1, general_pool)]:
        with open(pool, encoding="utf-8") as f:
            for ln, line in enumerate(f):
                key = (pool_id, ln)
                if key not in selected:
                    continue
                b, s = selected[key]
                o = json.loads(line)
                messages = list(o["prompt"]) + [{"role": "assistant", "content": o["gold_response"]}]
                buf_rows.append({"messages": messages, "data_source": o["data_source"],
                                 "ability": o["ability"],
                                 "extra_info": json.dumps({**(o.get("extra_info") or {}), "s3_bucket": b},
                                                          ensure_ascii=False)})
                buf_meta.append((b, s))
                if len(buf_rows) >= 2048:
                    flush_buf()
    flush_buf()
    if writer:
        writer.close()

    # ---------- 汇总 ----------
    print(f"\n==== S3-R1 切片完成 -> {args.out}")
    grand_r = grand_t = 0
    table = defaultdict(lambda: [0, 0])
    for (b, s), (r_, t_) in stats.items():
        table[b][0] += r_
        table[b][1] += t_
        grand_r += r_
        grand_t += t_
    print(f"总计 {grand_r:,} 条 / {grand_t/1e9:.3f}B token;超长丢弃 {sum(dropped.values()):,}")
    print("\n双口径分布(条数% / token%):")
    for b in ["easy", "medium", "hard", "general"]:
        r_, t_ = table[b]
        print(f"  {b:8s} {r_:8,} 条 ({r_/grand_r:5.1%})   {t_/1e6:7.1f}M tok ({t_/grand_t:5.1%})")
    print("\n桶内源明细(条 / M tok):")
    for b in ["easy", "medium", "hard", "general"]:
        for (bb, s), (r_, t_) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
            if bb == b:
                print(f"  [{b}] {s:38s} {r_:7,}  {t_/1e6:7.1f}M")
    if dropped:
        print("\n超长丢弃按源:", dict(dropped.most_common()))
    manifest = {"out": args.out, "seed": args.seed, "ratio": args.ratio,
                "total_tokens_target": args.total_tokens, "actual_rows": grand_r,
                "actual_tokens": grand_t, "dropped_overlen": dict(dropped),
                "buckets": {b: {"rows": table[b][0], "tokens": table[b][1]} for b in table},
                "detail": {f"{b}/{s}": {"rows": r_, "tokens": t_} for (b, s), (r_, t_) in stats.items()}}
    with open(args.out.replace(".parquet", ".manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
