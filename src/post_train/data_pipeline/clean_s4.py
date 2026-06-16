"""S4(最终阶段难题退火 SFT)前置清洗 —— 只去"错",不去"杂"。

背景(2026-06-12 审计,docs/reward_verifier_fix_20260612.md):
numinamath/cn_k12 等源 ~23-28% 样本 gold 损坏(LaTeX 截断/丢空格/证明题无意义标签),
SFT 会把坏标签当监督信号。本脚本对统一 SFT 池做三条**机械**过滤,不做风格改写:

  R1 proof   : 题面是证明题(Prove/证明/Show that)——答案监督无意义;
  R2 garbage : ground_truth 损坏(未知 LaTeX 命令/花括号不配平/≥3 连续字母的
               变量糊/超长),探测器在已知坏样本上 8/8 召回、26 例合法答案零误杀;
  R3 inconsistent: gold_response 的最终 boxed 与 ground_truth 经修复后的
               verifier(reward.verify_answer)判不一致——标签自相矛盾。

用法:
  python -m data_pipeline.clean_s4 \
    --pool /data/zilu/data_unified_v2/train_sft.jsonl \
    --out  /data/zilu/data_unified_v2/train_sft_s4clean.jsonl
产出: out + out.stats.json(按源×规则的丢弃计数)
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter
from multiprocessing import Pool

KNOWN_CMDS = set("""frac dfrac tfrac sqrt pi text mathrm mbox textbf boxed left right cdot times div pm mp
leq geq neq le ge ne infty circ degree angle triangle overline underline bar hat vec dot ddot prime
sin cos tan cot sec csc arcsin arccos arctan sinh cosh tanh log ln lg exp lim sum prod int oint
binom dbinom tbinom choose mathbb mathcal mathfrak mathsf mathit
Delta delta alpha beta gamma Gamma theta Theta lambda Lambda mu nu xi Xi rho sigma Sigma tau
phi Phi varphi chi psi Psi omega Omega epsilon varepsilon eta zeta iota kappa
cup cap subset subseteq supset supseteq in notin emptyset varnothing setminus mid nmid equiv pmod mod bmod
gcd lcm min max arg deg det dim ker operatorname langle rangle lfloor rfloor lceil rceil
ldots cdots dots vdots ddots quad qquad approx sim simeq cong propto perp parallel because therefore
forall exists neg lor land oplus otimes star ast bullet diamond bigcup bigcap bigoplus bigotimes
hline begin end pmatrix bmatrix vmatrix matrix array cases aligned align gather split over atop
stackrel underset overset xrightarrow rightarrow leftarrow Rightarrow Leftarrow leftrightarrow
Leftrightarrow mapsto to uparrow downarrow not displaystyle textstyle rm bf it sf tt cal
top bot wedge vee partial nabla hbar ell Re Im aleph surd flat natural sharp S P
amalg coprod uplus sqcup sqcap models vdash dashv asymp bowtie smile frown
lhd rhd unlhd unrhd triangleleft triangleright bigtriangleup bigtriangledown""".split())

_PROOF = re.compile(r"\b[Pp]rove\b|\b[Ss]how that\b|证明|求证")
_ENV = re.compile(r"\\(?:begin|end)\s*\{[a-zA-Z*]+\}")
_TEXT = re.compile(r"\\(?:text|mathrm|mbox|textbf|operatorname)\s*\{([^{}]*)\}")
_WORDS = re.compile(r"[A-Za-z]+(?: [A-Za-z]+){0,2}")
_CMD = re.compile(r"\\([a-zA-Z]+)")
_RUN3 = re.compile(r"[a-zA-Z]{3,}")


def garbage_gold(gt: str) -> bool:
    s = str(gt).strip()
    if not s or len(s) > 80:
        return True
    core = _ENV.sub(" ", s)
    core = _TEXT.sub(r"\1", core)
    w = core.strip()
    if _WORDS.fullmatch(w) and len(w) <= 14:   # blue / even / no solution
        return False
    if core.count("{") != core.count("}"):     # 截断损坏
        return True
    bad = False

    def eat(m):
        nonlocal bad
        if m.group(1) not in KNOWN_CMDS:
            bad = True
        return " "

    t = _CMD.sub(eat, core)
    if bad:
        return True
    return bool(_RUN3.search(t))               # 多字母变量糊/丢空格


def _user_q(row) -> str:
    for m in row.get("prompt", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


# R2 豁免:compute_cot 是自家合成源,gold 为明文 DSL("sqrt(53)"/"center (7,2), radius 6"),
# 不存在 LaTeX 截断/丢空格的损坏模式,探测器会大面积误杀(冒烟实测 86%+)
_TRUSTED_R2 = ("compute_cot",)
# 无 ground_truth 但保留:R1 蒸馏长推理源,本来就没有独立 gold,质量靠上游验证
_NOGT_OK = ("openthoughts3-math", "bespoke-stratos")


def check_row(line: str):
    """返回 (data_source, verdict);verdict ∈ keep/keep_nogt/proof/garbage/inconsistent/no_gt"""
    from data_pipeline.reward import verify_answer
    row = json.loads(line)
    src = row.get("data_source", "?")
    rm = row.get("reward_model") or {}
    gt, style = rm.get("ground_truth"), rm.get("style", "math_verify")
    if _PROOF.search(_user_q(row)):
        return src, "proof"
    if gt is None or str(gt) == "":
        return src, ("keep_nogt" if src.startswith(_NOGT_OK) else "no_gt")
    if not src.startswith(_TRUSTED_R2) and garbage_gold(gt):
        return src, "garbage"
    gr = row.get("gold_response")
    if gr and not verify_answer(gr, gt, style):   # RL 池无 gold_response,只做 R1/R2
        return src, "inconsistent"
    return src, "keep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/data/zilu/data_unified_v2/train_sft.jsonl")
    ap.add_argument("--out", default="/data/zilu/data_unified_v2/train_sft_s4clean.jsonl")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    stats = Counter()
    n_in = n_out = 0
    with open(args.pool, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout, \
            Pool(args.workers) as pool:
        for line, (src, verdict) in zip(_relines(args.pool), pool.imap(check_row, fin, chunksize=256)):
            n_in += 1
            stats[f"{src}|{verdict}"] += 1
            if verdict in ("keep", "keep_nogt"):
                fout.write(line)
                n_out += 1
            if n_in % 200000 == 0:
                print(f"  {n_in} 行,保留 {n_out}")
    per_src = {}
    for k, v in sorted(stats.items()):
        s, verd = k.split("|")
        per_src.setdefault(s, {})[verd] = v
    meta = {"pool": args.pool, "out": args.out, "rows_in": n_in, "rows_out": n_out,
            "dropped": n_in - n_out, "per_source": per_src}
    with open(args.out + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"完成: {n_in} -> {n_out} (丢 {n_in-n_out}, {100*(n_in-n_out)/max(n_in,1):.1f}%)")
    print("stats ->", args.out + ".stats.json")


def _relines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line


if __name__ == "__main__":
    main()
