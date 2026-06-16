#!/usr/bin/env python3
"""Compute_Cot 数据集质量体检 (只读，流式扫描全部 split)。

检查项:
  - 格式合规: <think>\\n...\\n</think>\\n#### \\boxed{...}
  - 一致性: boxed == answer 字段; answer 是否出现在 <think> (无跳步守卫)
  - 校验位: verified=true 占比; 空 trace / 空 answer / 编号列表 / 脏片段
  - 分布: 各 source 数量、difficulty 分布
  - 去重: split 内部 user 题面重复率
  - 泄漏: train ∩ (val/test) 的 user 题面完全重合

用法:
  python scripts/qc/quality_report.py [--data DIR]
默认 DATA = /data/zilu/fastrl/Compute_Cot/data
"""
from __future__ import annotations
import argparse, json, re, hashlib
from collections import Counter
from pathlib import Path

DEFAULT_DATA = "/data/zilu/fastrl/Compute_Cot/data"
SPLITS = [
    "train/s1_arithmetic", "train/s2_fractions", "train/s3_algebra",
    "train/s4_equations", "train/s5_broad", "val/val",
    "test/id_test", "test/extrap_ood", "test/template_ood",
]
FMT = re.compile(r"^<think>\n.*\n</think>\n#### \\boxed\{(.*)\}$", re.DOTALL)
BOX = re.compile(r"\\boxed\{(.*)\}\s*$", re.DOTALL)
NUMBERED = re.compile(r"(?m)^\s*\d+\.\s")
DIRTY = ["+-", "+ -", "--"]


def uhash(s: str) -> bytes:
    return hashlib.md5(s.encode("utf-8")).digest()


def pct(a: int, b: int) -> str:
    return f"{100*a/b:.2f}%" if b else "-"


def scan(path: Path) -> dict:
    s = Counter(); diff = Counter(); uhashes = set()
    n = vt = fmt = boxeq = ainthink = dirty = etrace = eans = numbered = dup = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line); n += 1
            msgs = o.get("messages", [])
            user = msgs[0]["content"] if msgs else ""
            asst = msgs[1]["content"] if len(msgs) > 1 else ""
            ans = str(o.get("answer", "")).strip()
            s[o.get("source", "?")] += 1
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
            h = uhash(user)
            if h in uhashes: dup += 1
            uhashes.add(h)
    return dict(n=n, vt=vt, fmt=fmt, boxeq=boxeq, ainthink=ainthink, dirty=dirty,
                etrace=etrace, eans=eans, numbered=numbered, n_src=len(s),
                uniq=len(uhashes), dup=dup, diff=dict(diff),
                top=s.most_common(3), rare=s.most_common()[-1] if s else None,
                hashes=uhashes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    args = ap.parse_args()
    data = Path(args.data)

    res = {name: scan(data / f"{name}.jsonl") for name in SPLITS}

    print("=" * 100, "\nPER-SPLIT QUALITY\n", "=" * 100, sep="")
    for name, r in res.items():
        print(f"\n### {name}  (n={r['n']:,})")
        print(f"  verified=true : {r['vt']:,} ({pct(r['vt'],r['n'])})   format OK: {r['fmt']:,} ({pct(r['fmt'],r['n'])})")
        print(f"  boxed==answer : {r['boxeq']:,} ({pct(r['boxeq'],r['n'])})   answer in think: {r['ainthink']:,} ({pct(r['ainthink'],r['n'])})")
        print(f"  dirty/empty   : dirty={r['dirty']}  empty_trace={r['etrace']}  empty_answer={r['eans']}  numbered={r['numbered']}")
        print(f"  #sources={r['n_src']}  difficulty={r['diff']}")
        print(f"  unique user   : {r['uniq']:,} -> intra-dup {r['dup']:,} ({pct(r['dup'],r['n'])})")
        print(f"  top src={r['top']}  rarest={r['rare']}")

    print("\n" + "=" * 100, "\nCROSS-SPLIT LEAKAGE (shared user-prompt hashes)\n", "=" * 100, sep="")
    train = set().union(*[r["hashes"] for k, r in res.items() if k.startswith("train/")])
    print(f"train union unique prompts: {len(train):,}")
    for tgt in ["val/val", "test/id_test", "test/extrap_ood", "test/template_ood"]:
        h = res[tgt]["hashes"]; inter = len(train & h)
        print(f"  train ∩ {tgt:18s}: {inter:,} / {len(h):,}  ({pct(inter,len(h))} of {tgt} leaked)")


if __name__ == "__main__":
    main()
