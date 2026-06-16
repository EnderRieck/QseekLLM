"""构建统一的 SFT 池 + RL 池（全局题面去重 + 评测隔离）。

依据 docs/hybrid_sft_rl_design.md §3/§4/§5：
- 全部数学源 → 统一格式（data_pipeline.adapters）。
- 按题面**全局去重**（first-wins，Compute_Cot 等优质源优先）。
- 与评测 benchmark（gsm8k-test / cmath / competition-math / math-beyond）**题面隔离**，防泄漏。
- 输出 train_sft.jsonl（use∈{sft,both}，带 gold_response）与 train_rl.jsonl（use∈{rl,both}，带 ground_truth）。

用法:
  python -m data_pipeline.build --out data_unified --cap 0   # cap=0 不限量(全量)
  python -m data_pipeline.build --out /tmp/du_test --cap 500  # 小规模冒烟
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import re
from collections import Counter

from .adapters import SOURCES, iter_source

# 优先级：Compute_Cot(worked) 与高质量带解的源优先占用题面（去重 first-wins）
SOURCE_ORDER = [
    "compute_cot", "gsm8k", "math-hendrycks", "chinese-r1-math", "calc-ape210k", "metamathqa", "orca-math-200k",
    "bespoke-stratos", "openthoughts3-math", "openr1-math-220k", "numinamath-1.5",
    "deepscaler", "infinity-math", "big-math", "dapo",
]

# 评测集（题面要从训练集排除）。2026-06-10 审计修订:
# - competition-math(全量 MATH 12.5k) → math-hendrycks 仅 test split(5k):解放 MATH-train 增强样本 ~12.9万
# - +svamp(两种题面拼法):修复经 orca 泄漏 63/300 的事故
# - +Compute_Cot val/test:堵 train 切分穿透 113 条
EVAL_PATHS = [
    ("/data/zilu/math_sft_raw/gsm8k/main/test-*.parquet", "parquet", "question"),
    ("/data/zilu/fastrl/data/benchmark/cmath", "arrow", "question"),
    ("/data/zilu/fastrl/data/benchmark/math-hendrycks", "arrow_test", "problem"),
    ("/data/zilu/fastrl/data/benchmark/math-500", "arrow", "problem"),
    ("/data/zilu/fastrl/data/benchmark/svamp", "arrow", "question_concat"),
    ("/data/zilu/fastrl/data/benchmark/svamp", "arrow", "Question"),
    ("/data/zilu/fastrl/data/benchmark/agieval-gaokao-mathcloze", "arrow", "query"),
    ("/data/zilu/fastrl/data/benchmark/agieval-gaokao-mathqa", "arrow", "query"),
    ("/data/zilu/math_sft_raw/math-beyond/**/*.parquet", "parquet", "problem"),
    ("/data/zilu/QseekLLM/src/post_train/Compute_Cot/data/clean/val.jsonl", "cc_jsonl", None),
    ("/data/zilu/QseekLLM/src/post_train/Compute_Cot/data/clean/test/id_test.jsonl", "cc_jsonl", None),
]


def qhash(q: str) -> str:
    # 归一化：去空白/标点差异后 hash，提升跨源去重命中
    norm = re.sub(r"\s+", "", q).lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def load_eval_hashes() -> set:
    import pyarrow.parquet as pq
    hashes = set()
    for path, fmt, field in EVAL_PATHS:
        try:
            if fmt == "parquet":
                for f in glob.glob(path, recursive=True):
                    t = pq.read_table(f, columns=[field]) if field in pq.read_schema(f).names else pq.read_table(f)
                    for v in t.column(field).to_pylist():
                        if v:
                            hashes.add(qhash(str(v)))
            elif fmt == "cc_jsonl":   # Compute_Cot 格式: messages[0]=user 题面
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        o = json.loads(line)
                        if o.get("messages"):
                            hashes.add(qhash(o["messages"][0]["content"]))
            else:  # arrow save_to_disk; "arrow_test"=仅 test split(可含多级子目录,如 math-hendrycks 的 7 学科)
                from datasets import load_from_disk
                roots = [path]
                if not (os.path.exists(os.path.join(path, "dataset_dict.json"))
                        or os.path.exists(os.path.join(path, "dataset_info.json"))):
                    roots = [os.path.join(path, d) for d in sorted(os.listdir(path))
                             if os.path.isdir(os.path.join(path, d))]
                for root in roots:
                    ds = load_from_disk(root)
                    if fmt == "arrow_test":
                        splits = [ds["test"]] if hasattr(ds, "keys") and "test" in ds.keys() else [ds]
                    else:
                        splits = ds.values() if hasattr(ds, "values") else [ds]
                    for sp in splits:
                        if field in sp.column_names:
                            for v in sp[field]:
                                if v:
                                    hashes.add(qhash(str(v)))
        except Exception as e:
            print(f"  [eval load warn] {path}: {str(e)[:80]}")
    return hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_unified")
    ap.add_argument("--cap", type=int, default=0, help="每源上限，0=不限")
    ap.add_argument("--sources", default="", help="逗号分隔，只跑这些源")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    out = args.out
    os.makedirs(out, exist_ok=True)

    print("加载评测题面（隔离防泄漏）...", flush=True)
    eval_hashes = load_eval_hashes()
    print(f"  评测题面 {len(eval_hashes):,} 条，将从训练集排除", flush=True)

    names = [s.strip() for s in args.sources.split(",") if s.strip()] or [n for n in SOURCE_ORDER if n in SOURCES]
    seen = set()
    leaked = 0
    sft_f = open(os.path.join(out, "train_sft.jsonl"), "w", encoding="utf-8")
    rl_f = open(os.path.join(out, "train_rl.jsonl"), "w", encoding="utf-8")
    stats = {"sft": Counter(), "rl": Counter(), "dup": Counter(), "skip_leak": Counter()}

    for name in names:
        cap = args.cap or None
        kept = 0
        for rec in iter_source(name, limit=None):
            h = qhash(rec.question)
            if h in eval_hashes:
                leaked += 1
                stats["skip_leak"][name] += 1
                continue
            if h in seen:
                stats["dup"][name] += 1
                continue
            seen.add(h)
            obj = rec.to_json_obj()
            if rec.use in ("sft", "both") and rec.gold_response:
                sft_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                stats["sft"][name] += 1
            if rec.use in ("rl", "both") and rec.ground_truth:
                rl_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                stats["rl"][name] += 1
            kept += 1
            if cap and kept >= cap:
                break
        print(f"  [{name}] kept={kept}  sft={stats['sft'][name]} rl={stats['rl'][name]} "
              f"dup={stats['dup'][name]} leak={stats['skip_leak'][name]}", flush=True)

    sft_f.close()
    rl_f.close()
    total_sft = sum(stats["sft"].values())
    total_rl = sum(stats["rl"].values())
    manifest = {"sft_total": total_sft, "rl_total": total_rl, "eval_excluded": leaked,
                "per_source": {n: {"sft": stats["sft"][n], "rl": stats["rl"][n],
                                   "dup": stats["dup"][n], "leak": stats["skip_leak"][n]} for n in names}}
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), ensure_ascii=False, indent=2)
    print("=" * 60, flush=True)
    print(f"SFT 池: {total_sft:,}   RL 池: {total_rl:,}   评测排除: {leaked:,}   全局去重命中: {sum(stats['dup'].values()):,}", flush=True)
    print(f"输出 -> {out}/{{train_sft,train_rl}}.jsonl + manifest.json", flush=True)


if __name__ == "__main__":
    main()
