"""Relaxed cc-reserved rescore for Qwen base outputs.

The standard verifier is intentionally strict for our trained models because
they are supervised to emit the exact Compute_Cot answer format. Qwen base often
solves correctly but writes equivalent final answers as inequalities or
comma-separated roots. This script keeps the original dump untouched and writes
an additional relaxed ability estimate for the external base baseline.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from data_pipeline.format import extract_boxed
from data_pipeline.reward import compute_reward, verify_answer


NUM = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"


def _pred_text(text: str) -> str:
    boxes = _extract_all_boxed(text or "")
    if len(boxes) >= 2:
        return ", ".join(boxes)
    pred = extract_boxed(text or "")
    if pred is not None:
        return pred
    hs = re.findall(r"####\s*(.+?)\s*$", text or "", flags=re.MULTILINE)
    return hs[-1].strip() if hs else (text or "").strip().splitlines()[-1] if (text or "").strip() else ""


def _extract_all_boxed(text: str) -> list[str]:
    out = []
    key = "\\boxed{"
    pos = 0
    while True:
        idx = text.find(key, pos)
        if idx < 0:
            break
        i = idx + len(key)
        depth = 1
        buf = []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            i += 1
        if depth == 0:
            val = "".join(buf).strip()
            if val:
                out.append(val)
            pos = i + 1
        else:
            break
    return out


def _clean(s: str) -> str:
    s = str(s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "")
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\(?:quad|qquad|;|,|!| )", " ", s)
    s = s.replace("−", "-").replace("–", "-")
    s = s.replace("∞", "infty").replace("+\\infty", "+infty").replace("-\\infty", "-infty")
    return re.sub(r"\s+", " ", s).strip()


def _norm_atom(s: str) -> str:
    s = _clean(s).strip()
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)
    s = s.strip(" $")
    s = re.sub(r"\s+", "", s)
    return s


def _split_values(s: str) -> list[str]:
    s = _clean(s)
    s = re.sub(r"\b(?:or|and)\b", ",", s, flags=re.I)
    s = s.replace("\\text{or}", ",").replace("\\text{and}", ",")
    vals = [_norm_atom(x) for x in re.split(r"[,;]", s) if _norm_atom(x)]
    return vals


def _same_multisolution(pred: str, gold: str) -> bool:
    if not re.search(r"\s+(?:or|and)\s+", gold):
        return False
    gp = [_norm_atom(x) for x in re.split(r"\s+(?:or|and)\s+", _clean(gold)) if _norm_atom(x)]
    pp = _split_values(pred)
    if len(pp) != len(gp):
        return False
    return sorted(pp) == sorted(gp)


def _parse_interval_gold(gold: str):
    g = _clean(gold).replace(" ", "")
    if "infty" in g and "∪" in g:
        nums = re.findall(NUM, g)
        if len(nums) == 2:
            lo, hi = sorted(float(x) for x in nums)
            return ("outside", lo, hi)
    if g.startswith("(") and g.endswith(")") and "," in g and "infty" not in g:
        parts = g.strip("()").split(",")
        if len(parts) == 2 and re.fullmatch(NUM, parts[0]) and re.fullmatch(NUM, parts[1]):
            return ("inside", float(parts[0]), float(parts[1]))
    return None


def _same_interval_or_ineq(pred: str, gold: str) -> bool:
    interval = _parse_interval_gold(gold)
    if interval is None:
        return False
    kind, a, b = interval
    p = _clean(pred)

    # Interval notation in the prediction, e.g. (-1, 13) or (-infty,-30) U (8,infty).
    pint = _parse_interval_gold(p)
    if pint is not None:
        return pint[0] == kind and abs(pint[1] - a) < 1e-9 and abs(pint[2] - b) < 1e-9

    if kind == "inside":
        # -1 < x < 13, or x > -1 and x < 13.
        m = re.search(rf"({NUM})\s*<\s*x\s*<\s*({NUM})", p)
        if m and abs(float(m.group(1)) - a) < 1e-9 and abs(float(m.group(2)) - b) < 1e-9:
            return True
        return (
            bool(re.search(rf"x\s*>\s*{re.escape(str(int(a) if a.is_integer() else a))}\b", p))
            and bool(re.search(rf"x\s*<\s*{re.escape(str(int(b) if b.is_integer() else b))}\b", p))
        )

    # outside: x < a or x > b, order-insensitive.
    aa = str(int(a) if a.is_integer() else a)
    bb = str(int(b) if b.is_integer() else b)
    return bool(re.search(rf"x\s*<\s*{re.escape(aa)}\b", p)) and bool(
        re.search(rf"x\s*>\s*{re.escape(bb)}\b", p)
    )


def relaxed_correct(text: str, gold: str, style: str) -> bool:
    if compute_reward(text, gold, style)["correct"]:
        return True
    pred = _pred_text(text)
    return _same_multisolution(pred, gold) or _same_interval_or_ineq(pred, gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    n = p1 = pk = fmt = 0
    by_source = defaultdict(lambda: [0, 0])
    for line in open(args.jsonl, encoding="utf-8"):
        o = json.loads(line)
        n += 1
        gold = str(o["gold"])
        style = o.get("style", "compute_cot")
        c1 = relaxed_correct(o.get("gen_greedy", ""), gold, style)
        ck = c1 or any(relaxed_correct(t, gold, style) for t in o.get("gens", []))
        p1 += bool(c1)
        pk += bool(ck)
        fmt += bool(o.get("has_format"))
        src = o.get("meta", {}).get("source", "")
        by_source[src][0] += bool(c1)
        by_source[src][1] += 1
    metrics = {
        "n": n,
        "pass@1_relaxed": round(p1 / n, 4) if n else 0.0,
        f"pass@{args.k}_relaxed": round(pk / n, 4) if n else 0.0,
        "format_rate_original": round(fmt / n, 4) if n else 0.0,
        "breakdown_source_relaxed": {
            src: {"acc": round(c / t, 4), "n": t} for src, (c, t) in sorted(by_source.items())
        },
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
