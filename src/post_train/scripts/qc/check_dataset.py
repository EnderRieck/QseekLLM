#!/usr/bin/env python3
"""对单个 jsonl 数据文件做一次性体检 (只读)：格式 / 答案一致 / 推演数值自洽 / 重复 / 难度分布。

把 quality_report(格式·一致·去重) 与 audit_reasoning(步级数值核算) 合到单文件上，
用于"用修好的生成器产一批新数据 → 立刻验收"。

用法:
  python scripts/qc/check_dataset.py <file.jsonl> [--examples 3]
"""
from __future__ import annotations
import argparse, json, re, sys, hashlib
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_reasoning import check_think  # 复用已验证的步级等式核算

FMT = re.compile(r"^<think>\n.*\n</think>\n#### \\boxed\{(.*)\}$", re.DOTALL)
BOX = re.compile(r"\\boxed\{(.*)\}\s*$", re.DOTALL)
NUMBERED = re.compile(r"(?m)^\s*\d+\.\s")
DIRTY = ["+-", "+ -", "--"]


def pct(a, b):
    return f"{100*a/b:.2f}%" if b else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--cap-per-source", type=int, default=400)
    args = ap.parse_args()

    n = vt = fmt = boxeq = ainthink = dirty = etrace = eans = numbered = dup = 0
    diff = Counter(); src = Counter(); uhashes = set()
    seen = defaultdict(int)
    bad_reason_src = Counter(); reason_examples = defaultdict(list)

    with open(args.file, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line); n += 1
            s = o.get("source", "?"); src[s] += 1
            asst = o["messages"][1]["content"]; user = o["messages"][0]["content"]
            ans = str(o.get("answer", "")).strip()
            diff[str(o.get("metadata", {}).get("difficulty", "?"))] += 1
            if o.get("verified") is True: vt += 1
            if FMT.match(asst): fmt += 1
            bm = BOX.search(asst)
            if bm and bm.group(1).strip() == ans: boxeq += 1
            think = asst.split("</think>")[0]
            if ans and ans in think: ainthink += 1
            if any(d in asst for d in DIRTY): dirty += 1
            if not o.get("trace"): etrace += 1
            if not ans: eans += 1
            if NUMBERED.search(think): numbered += 1
            h = hashlib.md5(user.encode()).digest()
            if h in uhashes: dup += 1
            uhashes.add(h)
            # 步级数值核算 (按 source 限量)
            if seen[s] < args.cap_per_source:
                seen[s] += 1
                _, bad, bads = check_think(think)
                if bad:
                    bad_reason_src[s] += 1
                    if len(reason_examples[s]) < args.examples:
                        reason_examples[s].append((user, bads[0]))

    print(f"file: {args.file}")
    print(f"samples         : {n:,}   sources: {len(src)}")
    print(f"verified=true   : {vt:,} ({pct(vt,n)})")
    print(f"format OK       : {fmt:,} ({pct(fmt,n)})")
    print(f"boxed==answer   : {boxeq:,} ({pct(boxeq,n)})")
    print(f"answer in think : {ainthink:,} ({pct(ainthink,n)})")
    print(f"dirty fragment  : {dirty}   empty_trace: {etrace}   empty_answer: {eans}   numbered: {numbered}")
    print(f"unique user     : {len(uhashes):,}  -> dup {dup:,} ({pct(dup,n)})")
    print(f"difficulty      : {dict(diff)}")
    flagged = bad_reason_src.most_common()
    print(f"\n步级数值审计: {len(flagged)} / {len(src)} 个 source 含不自洽算式")
    for s, c in flagged:
        print(f"  {c}/{seen[s]}  {s}")
        for q, bad in reason_examples[s]:
            print(f"      Q: {q}")
            print(f"      ✗ {bad}")

    problems = (n - vt) + (n - fmt) + (n - boxeq) + dirty + etrace + eans + numbered + len(flagged)
    print(f"\n{'✅ PASS — 未发现问题' if problems == 0 else '⚠️  存在问题, 见上'}")
    sys.exit(0 if problems == 0 else 1)


if __name__ == "__main__":
    main()
