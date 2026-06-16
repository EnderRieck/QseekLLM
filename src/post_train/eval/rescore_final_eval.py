"""离线重判终评结果(不重新生成)。

背景(2026-06-12):
  1. 旧版 score_bench 落盘时丢弃了 Pass@k 采样全文(只存对错布尔),但 .tmp_{bench}/
     分片里原文还在 —— 本脚本把它们合并回 {bench}.jsonl 的 "gens" 字段。
  2. data_pipeline.reward 修复了 compute_cot 多解判分 bug(math_verify 对
     "x=-4 or x=26" 只解析最后一个解)—— 本脚本用当前判分器全量重判,
     重写 {bench}.jsonl / summary.json / summary.md。

用法(终评进程结束后再跑):
  .venv/bin/python -m eval.rescore_final_eval --out-dir <ckpt>/final_eval
"""

import argparse
import glob
import json
import os

from eval.final_eval import score_bench, write_summary_md


def _load_tmp_gens(tmp_dir: str) -> dict:
    """读分片 -> {idx: {"gen_greedy":..., "gens":[...]}}。"""
    gens = {}
    for p in sorted(glob.glob(os.path.join(tmp_dir, "shard_*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                gens[o["idx"]] = o
    return gens


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="<ckpt>/final_eval 目录")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    out = args.out_dir.rstrip("/")

    summary_path = os.path.join(out, "summary.json")
    ckpt = out
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            ckpt = json.load(f).get("ckpt", out)

    all_metrics = {}
    for bench in ["cc-reserved", "gsm8k", "math500", "gsmplus", "cmath", "svamp"]:
        path = os.path.join(out, f"{bench}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            records = [json.loads(l) for l in f]
        tmp_gens = _load_tmp_gens(os.path.join(out, f".tmp_{bench}"))
        n = max(r["idx"] for r in records) + 1
        rows = [{"meta": {}, "question": "", "gold": "", "style": "math_verify"}] * n
        gens = {}
        for r in records:
            i = r["idx"]
            rows[i] = {"meta": r["meta"], "question": r["question"],
                       "gold": r["gold"], "style": r["style"]}
            gens[i] = {"gen_greedy": r["gen_greedy"],
                       "gens": r.get("gens") or tmp_gens.get(i, {}).get("gens", [])}
        new_records, metrics = score_bench(bench, rows, gens, args.k)
        with open(path, "w", encoding="utf-8") as f:
            for r in new_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_metrics[bench] = metrics
        print(f"[{bench}] n={metrics['n']} pass@1={metrics['pass@1']}"
              f" pass@{args.k}={metrics[f'pass@{args.k}']} fmt={metrics['format_rate']}", flush=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"ckpt": ckpt, "k": args.k, "metrics": all_metrics}, f,
                  ensure_ascii=False, indent=2)
    write_summary_md(os.path.join(out, "summary.md"), ckpt, all_metrics, args.k)
    print(f"重判完成 -> {out}/{{summary.md,summary.json}}", flush=True)


if __name__ == "__main__":
    main()
