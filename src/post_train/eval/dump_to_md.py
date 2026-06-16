"""把 async_eval 的 heldout.jsonl dump 转成人类可读的 Markdown，方便翻看模型输出。

用法:
  # 直接给 dump 文件
  python -m eval.dump_to_md <ckpt>/eval_dumps/step_500/heldout.jsonl
  # 或给 step（自动找 <ckpt-dir>/eval_dumps/step_N/heldout.jsonl）
  python -m eval.dump_to_md --ckpt-dir <ckpt-dir> --step 500

常用过滤:
  --per-source 15     每个 source 最多输出几条(默认15, 0=全部)
  --source gsm8k      只导某类(子串匹配, 如 gsm8k / cmath / compute_cot / cmmlu)
  --only-wrong        只看答错的    --only-correct 只看答对的
  --out a.md          输出路径(默认 dump 同目录 heldout.md)
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", nargs="?", default="", help="heldout.jsonl 路径")
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--step", type=int, default=-1)
    ap.add_argument("--out", default="")
    ap.add_argument("--per-source", type=int, default=15, help="每 source 最多导出条数(0=全部)")
    ap.add_argument("--source", default="", help="只导含此子串的 source")
    ap.add_argument("--only-wrong", action="store_true")
    ap.add_argument("--only-correct", action="store_true")
    args = ap.parse_args()

    dump = args.dump
    if not dump:
        if not args.ckpt_dir or args.step < 0:
            ap.error("需给 dump 路径，或 --ckpt-dir + --step")
        dump = os.path.join(args.ckpt_dir, "eval_dumps", f"step_{args.step}", "heldout.jsonl")
    if not os.path.exists(dump):
        ap.error(f"找不到 {dump}")

    rows = [json.loads(l) for l in open(dump, encoding="utf-8")]
    if args.source:
        rows = [r for r in rows if args.source in r["source"]]
    if args.only_wrong:
        rows = [r for r in rows if not r["correct"]]
    if args.only_correct:
        rows = [r for r in rows if r["correct"]]

    # 按 source 分组
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"].split(":")[0]].append(r)

    out = args.out or os.path.join(os.path.dirname(dump), "heldout.md")
    step_name = os.path.basename(os.path.dirname(dump))
    n = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    n_fmt = sum(r["has_format"] for r in rows)

    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# 评测输出 · {step_name}\n\n")
        f.write(f"- dump: `{dump}`\n")
        f.write(f"- 条数(过滤后): **{n}**  正确 {n_correct} ({n_correct/max(n,1):.1%})  "
                f"含格式 {n_fmt} ({n_fmt/max(n,1):.1%})\n")
        if args.source or args.only_wrong or args.only_correct or args.per_source:
            flt = []
            if args.source: flt.append(f"source~`{args.source}`")
            if args.only_wrong: flt.append("仅错题")
            if args.only_correct: flt.append("仅对题")
            if args.per_source: flt.append(f"每source≤{args.per_source}")
            f.write(f"- 过滤: {', '.join(flt)}\n")
        f.write("\n## 各 source 准确率\n\n| source | 正确/总数 | acc |\n|---|---|---|\n")
        for s in sorted(by_src, key=lambda k: -len(by_src[k])):
            sub = by_src[s]
            c = sum(r["correct"] for r in sub)
            f.write(f"| {s} | {c}/{len(sub)} | {c/len(sub):.1%} |\n")
        f.write("\n---\n")

        for s in sorted(by_src, key=lambda k: -len(by_src[k])):
            sub = by_src[s]
            shown = sub if args.per_source == 0 else sub[:args.per_source]
            f.write(f"\n## {s}  ({len(sub)} 条" +
                    (f"，展示前 {len(shown)}" if len(shown) < len(sub) else "") + ")\n\n")
            for r in shown:
                mark = "✅" if r["correct"] else "❌"
                fmt = "有格式" if r["has_format"] else "无格式"
                f.write(f"### {mark} `{r['source']}`  ·  gold=`{r['gold']}`  ·  {fmt}  ·  "
                        f"len={len(r['generation'])}\n\n")
                f.write("**问题**\n\n")
                f.write("```text\n" + r["prompt"].strip() + "\n```\n\n")
                f.write("**模型输出**\n\n")
                f.write("```text\n" + r["generation"].strip() + "\n```\n\n")
                f.write("---\n")

    print(f"已写出 {out}（{n} 条，过滤后）")


if __name__ == "__main__":
    main()
