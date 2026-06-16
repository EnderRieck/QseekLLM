#!/usr/bin/env python3
"""确定性推演审计 (只读)：核算每条 <think> 中的纯数字等式是否成立。

验证器只查最终答案，查不到推演中途的错误算式 (如 `168÷140=12`)。本脚本把
<think> 里形如 `数字表达式 = 数字` 的等式抽出来，用 Python 实算左边并与右边比对，
按 source 汇总"含错误算式的样本比例"，把答案对、过程错的 source 揪出来。

策略 (保守，尽量不误报):
  - 只检查纯数字等式 (含字母/变量的段落跳过 → 不碰代数恒等式)。
  - LHS 必须含运算符 (+ - * /)，否则不算"计算"。
  - 仅当相对误差 > 1e-2 才判错 → 容忍合法的小数舍入。
  - 跳过含 < > ≤ ≥ ≈ 的比较/近似行。

用法:
  python scripts/qc/audit_reasoning.py [--data DIR] [--cap-per-source N] [--examples K]
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

DEFAULT_DATA = "/data/zilu/fastrl/Compute_Cot/data"
FILES = [
    "train/s1_arithmetic", "train/s2_fractions", "train/s3_algebra",
    "train/s4_equations", "train/s5_broad", "val/val",
    "test/id_test", "test/extrap_ood", "test/template_ood",
]

NORM = str.maketrans({"×": "*", "✕": "*", "⋅": "*", "·": "*", "÷": "/",
                      "−": "-", "–": "-", "—": "-", "^": "*"})  # ^ 先占位, 下面换 **
SAFE = re.compile(r"^[\d.+\-*/()\s]+$")
HAS_OP = re.compile(r"\d\s*[+*/]\s*[\d(]|[\d)]\s*-\s*[\d(]")  # 真含二元运算
SKIP_LINE = re.compile(r"[<>≤≥≈√π]|remainder|sqrt|sin|cos|tan|\blog|\bln\b|\[\[")  # 非纯算术记号跳过
# 一段文字尾部 / 首部的数字表达式 (天然在字母处断开)
TAIL = re.compile(r"[\d.()+\-*/\s]*$")
HEAD = re.compile(r"^[\d.()+\-*/\s]*")


def eval_num(expr: str):
    expr = expr.strip()
    if not expr or not SAFE.match(expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 (受 SAFE 白名单约束)
    except Exception:
        return None


def _has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


def check_think(think: str):
    """按 '=' 切链，比较相邻数字段。返回 (比较次数, 不一致数, [示例])."""
    total = bad = 0
    bads = []
    for raw in think.split("\n"):
        # 幂: a^b -> a**b。先临时标记，避免与乘号混淆
        line = raw.replace("^", "@POW@").translate(NORM).replace("@POW@", "**")
        if SKIP_LINE.search(line) or "=" not in line:
            continue
        parts = line.split("=")
        for i in range(len(parts) - 1):
            tail = TAIL.search(parts[i]).group(0).strip()
            head = HEAD.match(parts[i + 1]).group(0).strip()
            if not (_has_digit(tail) and _has_digit(head)):
                continue
            if not HAS_OP.search(tail):
                continue  # 左侧必须是含运算符的算式 (排除下标/裸数字/标签)
            if re.match(r"^[+*/]|^-\s", tail):
                continue  # 左侧以悬空运算符开头 = 含变量项被截断 (代数 FP)，跳过
            lv, rv = eval_num(tail), eval_num(head)
            if lv is None or rv is None:
                continue
            total += 1
            if abs(lv - rv) > 1e-2 * max(1.0, abs(rv)) and abs(lv - rv) > 1e-9:
                bad += 1
                bads.append(f"{tail} = {head}  (左={lv}, 右={rv})")
    return total, bad, bads


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--cap-per-source", type=int, default=400, help="每 source 最多核查多少条")
    ap.add_argument("--examples", type=int, default=2, help="每个问题 source 打印几条错误示例")
    args = ap.parse_args()
    data = Path(args.data)

    seen = defaultdict(int)
    bad_samples = defaultdict(int)
    n_eq = defaultdict(int)
    n_bad_eq = defaultdict(int)
    examples = defaultdict(list)

    for fname in FILES:
        with (data / f"{fname}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                src = o.get("source", "?")
                if seen[src] >= args.cap_per_source:
                    continue
                seen[src] += 1
                think = o["messages"][1]["content"].split("</think>")[0]
                total, bad, bads = check_think(think)
                n_eq[src] += total
                n_bad_eq[src] += bad
                if bad:
                    bad_samples[src] += 1
                    if len(examples[src]) < args.examples:
                        examples[src].append((o["messages"][0]["content"], bads[0]))

    rows = []
    for src in seen:
        rate = bad_samples[src] / seen[src] if seen[src] else 0
        rows.append((rate, src, seen[src], bad_samples[src], n_eq[src], n_bad_eq[src]))
    rows.sort(reverse=True)

    print("=" * 100)
    print(f"REASONING NUMERIC AUDIT  ({len(seen)} sources, cap {args.cap_per_source}/src)")
    print("=" * 100)
    flagged = [r for r in rows if r[0] > 0]
    print(f"\n含错误算式的 source: {len(flagged)} / {len(seen)}\n")
    print(f"{'bad%':>7}  {'bad/seen':>10}  {'badEq/eq':>12}  source")
    print("-" * 100)
    for rate, src, seen_n, bsamp, eq, beq in flagged:
        print(f"{100*rate:6.1f}%  {bsamp:>4}/{seen_n:<5}  {beq:>5}/{eq:<6}  {src}")

    print("\n" + "=" * 100)
    print("错误示例 (每个问题 source 取前若干条)")
    print("=" * 100)
    for rate, src, *_ in flagged:
        print(f"\n### {src}  ({100*rate:.1f}% bad)")
        for q, bad in examples[src]:
            print(f"  Q: {q}")
            print(f"     ✗ {bad}")


if __name__ == "__main__":
    main()
