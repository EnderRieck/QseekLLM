"""各数据源加载器 + "原始行 → 统一 Record" 适配器。

每个 source 在 SOURCES 里登记：path / 加载方式 / 适配函数 / 默认用途与验证器。
适配函数返回 Record 或 None(跳过脏样本)。
"""
from __future__ import annotations
import glob
import json
import os
import re
from typing import Iterator, Optional

from .format import (
    Record, make_prompt, wrap_think_boxed, extract_boxed,
    extract_gsm8k_answer, extract_metamath_answer,
)

DAPO_PREFIX = re.compile(
    r"^Solve the following math problem step by step\..*?answer to the problem\.\s*", re.DOTALL)

FASTRL = "/data/zilu/fastrl/data/train"
MATH_RAW = "/data/zilu/math_sft_raw"
COMPUTE_COT = "/data/zilu/QseekLLM/src/post_train/Compute_Cot/data/clean"


# ----------------------------------------------------------------------------- loaders
def iter_jsonl(path: str) -> Iterator[dict]:
    for f in sorted(glob.glob(path)) if "*" in path else [path]:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def iter_json_array(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    yield from data


def iter_parquet(globpat: str) -> Iterator[dict]:
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(globpat, recursive=True)):
        t = pq.read_table(f)
        for batch in t.to_batches(max_chunksize=2048):
            for row in batch.to_pylist():
                yield row


def iter_arrow(path: str) -> Iterator[dict]:
    from datasets import load_from_disk
    ds = load_from_disk(path)
    splits = ds.values() if hasattr(ds, "values") else [ds]
    for sp in splits:
        for row in sp:
            yield row


def iter_arrow_train(path: str) -> Iterator[dict]:
    """只取 train split（隔离 test/val，防泄漏）。"""
    from datasets import load_from_disk
    ds = load_from_disk(path)
    sp = ds["train"] if hasattr(ds, "keys") and "train" in ds.keys() else ds
    for row in sp:
        yield row


# ----------------------------------------------------------------------------- adapters
_NUM_PAT = r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?"   # 整数/小数/分数(整体捕获)
_LEAD_RE = re.compile(
    r"(?:\bis\b|\bare\b|\bwas\b|\bwere\b|\bbe\b|=|:)\s*"
    r"(?:approximately\s*|about\s*|around\s*)?[\$€£]?\s*(" + _NUM_PAT + r")\s*%?",
    re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def _num_from_text(text: str) -> Optional[str]:
    """从解答抽最终答案（2026-06-10 重写，修复旧版"末200字符取第一个="抓中间结果的 bug，
    旧版实测 56% 抽错——见 docs/data_audit_report_20260610.md §1.1）。

    策略（审计验证 ~90-95% 准确）：
    1. 有 \\boxed{} 直接用；
    2. 显式 "answer is/答案是" 引导(全文找最后一次)；
    3. 末个含数字的句子里，取最后一个 is/are/=/: 引导的数（分数整体捕获）；
    4. 末含数句只有一个数 → 用它；多个数且无引导词 → 低置信，返回 None(调用方跳过该行)。
    """
    t = text.strip()
    b = extract_boxed(t)
    if b:
        return b
    m = None
    for m in re.finditer(r"(?:answer is|答案是)[:\s]*\$?(" + _NUM_PAT + r")", t, re.IGNORECASE):
        pass
    if m:
        return m.group(1).rstrip(".").replace(",", "")
    sents = [s for s in _SENT_SPLIT.split(t) if s and re.search(r"\d", s)]
    if not sents:
        return None
    last = sents[-1].replace(",", "")
    led = _LEAD_RE.findall(last)
    if led:
        return led[-1].rstrip(".")
    nums = re.findall(_NUM_PAT, last)
    if len(nums) == 1:
        return nums[0].rstrip(".")
    return None                       # 多数字无引导词：低置信，丢弃


def adapt_compute_cot(r: dict) -> Optional[Record]:
    msgs = r.get("messages", [])
    if len(msgs) < 2:
        return None
    q = msgs[0]["content"]
    gold = msgs[1]["content"]            # 已是 <think>…</think>\n#### \boxed{}
    ans = str(r.get("answer", "")).strip()
    src = r.get("source", "compute_cot")
    return Record(prompt=make_prompt(q), data_source=f"compute_cot:{src}", ability="math",
                  use="both", reward_style="compute_cot", ground_truth=ans, gold_response=gold,
                  difficulty=str(r.get("metadata", {}).get("difficulty", "")), source=src, question=q)


def adapt_orca(r: dict) -> Optional[Record]:
    q, sol = r.get("question", ""), str(r.get("answer", ""))
    if not q or not sol:
        return None
    ans = _num_from_text(sol)
    if not ans:                          # 抽不到答案 → 跳过，不产生空 boxed
        return None
    return Record(prompt=make_prompt(q), data_source="orca-math-200k", ability="math",
                  use="both", reward_style="math_verify", ground_truth=ans,
                  gold_response=wrap_think_boxed(sol, ans), difficulty="easy",
                  source="orca_math", question=q)


def adapt_metamath(r: dict) -> Optional[Record]:
    q, resp = r.get("query", ""), str(r.get("response", ""))
    if not q or not resp:
        return None
    ans = extract_metamath_answer(resp)
    if not ans:
        return None
    typ = r.get("type", "")
    diff = "easy" if typ.startswith("GSM") else "hard"
    return Record(prompt=make_prompt(q), data_source="metamathqa", ability="math",
                  use="both", reward_style="math_verify", ground_truth=ans,
                  gold_response=wrap_think_boxed(resp, ans), difficulty=diff,
                  source=f"metamath:{typ}", question=q)


_NUMINA_EASY_SRC = {"orca_math", "synthetic_math", "cn_k12", "metamath", "gsm8k"}
_BAD_ANS = ("", "proof", "notfound", "none", "null")


def adapt_numina(r: dict) -> Optional[Record]:
    """NuminaMath-1.5（2026-06-10 重写）：
    - 按 problem_is_valid/solution_is_valid 双 Yes 过滤(挡掉 ~4.1% 残题/未解出,含波兰语残题);
    - 剔 answer∈{proof,notfound,空}(旧版产生 1.8万 \\boxed{proof} 占位毒样本);
    - source 字段纠错(旧版误用 problem_type),难度按 source 分层。"""
    if str(r.get("problem_is_valid", "")).strip() != "Yes":
        return None
    if str(r.get("solution_is_valid", "")).strip() != "Yes":
        return None
    q = r.get("problem", "")
    ans = str(r.get("answer", "") or "").strip()
    if not q or ans.lower() in _BAD_ANS:
        return None
    sol = str(r.get("solution", "") or "").strip()
    gold = wrap_think_boxed(sol, ans) if sol else ""
    src = str(r.get("source", "") or "numina")
    diff = "medium" if src in _NUMINA_EASY_SRC else "hard"
    return Record(prompt=make_prompt(q), data_source="numinamath-1.5", ability="math",
                  use="both" if gold else "rl", reward_style="math_verify", ground_truth=ans,
                  gold_response=gold, difficulty=diff, source=f"numina:{src}", question=q)


def adapt_openr1(r: dict) -> Optional[Record]:
    """OpenR1-Math-220k（2026-06-10 重写）：改用 R1 generations + correctness 标注。
    旧版错把人写 solution(52.6% 无 boxed)当 CoT,没用该集的核心价值——
    现取 correctness_math_verify=True 中最短的 R1 轨迹做 gold,correctness_count 当难度代理。"""
    q = r.get("problem", "")
    ans = str(r.get("answer", "") or "").strip()
    if not q or ans.lower() in _BAD_ANS:
        return None
    gens = list(r.get("generations") or [])
    corr = list(r.get("correctness_math_verify") or [])
    cand = [g for g, c in zip(gens, corr) if c and g]
    gold = wrap_think_boxed(min(cand, key=len), ans) if cand else ""
    src = str(r.get("source", "") or "openr1")
    return Record(prompt=make_prompt(q), data_source="openr1-math-220k", ability="math",
                  use="both" if gold else "rl", reward_style="math_verify", ground_truth=ans,
                  gold_response=gold, difficulty=f"cc={r.get('correctness_count', '')}",
                  source=f"openr1:{src}", question=q)


def _adapt_problem_solution_answer(src_name: str, src_tag_field: Optional[str] = None,
                                   diff_field: Optional[str] = None, difficulty="hard"):
    def fn(r: dict) -> Optional[Record]:
        q = r.get("problem", "")
        sol = str(r.get("solution", "") or "")
        ans = str(r.get("answer", "") or "").strip()
        if not q:
            return None
        if ans.lower() in ("", "proof"):       # 非数值答案 → 仅 SFT
            use, style, gt = "sft", "none", ""
        else:
            use, style, gt = "both", "math_verify", ans
        gold = wrap_think_boxed(sol, ans) if sol else ""
        if not gold and not gt:
            return None
        src = r.get(src_tag_field, "") if src_tag_field else ""
        return Record(prompt=make_prompt(q), data_source=src_name, ability="math",
                      use=use if gold else "rl", reward_style=style, ground_truth=gt,
                      gold_response=gold, difficulty=str(r.get(diff_field, difficulty)) if diff_field else difficulty,
                      source=f"{src_name}:{src}" if src else src_name, question=q)
    return fn


def adapt_gsm8k(r: dict) -> Optional[Record]:
    q, ans_field = r.get("question", ""), str(r.get("answer", ""))
    ans = extract_gsm8k_answer(ans_field)
    if not q or not ans:
        return None
    return Record(prompt=make_prompt(q), data_source="gsm8k", ability="math", use="both",
                  reward_style="gsm8k", ground_truth=ans, gold_response=wrap_think_boxed(ans_field, ans),
                  difficulty="easy", source="gsm8k", question=q)


def adapt_math_hendrycks(r: dict) -> Optional[Record]:
    """官方 split 版 Hendrycks MATH 的 **train** 段(7.5k,带 worked solution)。
    2026-06-10 新增:test 段(5k)在 EVAL_PATHS 隔离,train 段是干净的中段 SFT 源,difficulty=官方 level。"""
    q, sol = r.get("problem", ""), str(r.get("solution", "") or "")
    ans = extract_boxed(sol)
    if not q or not sol or not ans:
        return None
    return Record(prompt=make_prompt(q), data_source="math-hendrycks", ability="math",
                  use="both", reward_style="math_verify", ground_truth=ans,
                  gold_response=wrap_think_boxed(sol, ans),
                  difficulty=str(r.get("level", "")), source=f"hendrycks:{r.get('type', '')}", question=q)


def iter_hendrycks_train(path: str) -> Iterator[dict]:
    """math-hendrycks 是 7 个学科子目录,各取 train split。"""
    from datasets import load_from_disk
    for sub in sorted(os.listdir(path)):
        p = os.path.join(path, sub)
        if os.path.isdir(p):
            ds = load_from_disk(p)
            yield from ds["train"]


def adapt_big_math(r: dict) -> Optional[Record]:
    q, ans = r.get("problem", ""), str(r.get("answer", "")).strip()
    if not q or not ans:
        return None
    sr = r.get("llama8b_solve_rate", None)
    return Record(prompt=make_prompt(q), data_source="big-math-rl-verified", ability="math",
                  use="rl", reward_style="math_verify", ground_truth=ans, gold_response="",
                  difficulty=f"solve_rate={sr}" if sr is not None else "", source=r.get("source", "big_math"),
                  question=q)


def adapt_infinity_math(r: dict) -> Optional[Record]:
    """infinity-instruct 里**含 \\boxed{} 的对话** → 数学题（有可验证答案，SFT+RL）。"""
    conv = r.get("conversations") or []
    if len(conv) < 2:
        return None
    q = conv[0].get("value", "")
    sol = conv[-1].get("value", "")
    ans = extract_boxed(sol)
    if not q or not ans:
        return None
    ans = re.sub(r"\.?\s*The answer is:.*$", "", ans, flags=re.DOTALL).strip()  # 去尾巴
    if not ans:
        return None
    return Record(prompt=make_prompt(q), data_source="infinity-math", ability="math", use="both",
                  reward_style="math_verify", ground_truth=ans, gold_response=wrap_think_boxed(sol, ans),
                  difficulty="", source=f"infinity:{r.get('langdetect', '')}", question=q)


def adapt_openthoughts3_math(r: dict) -> Optional[Record]:
    """OpenThoughts3：仅 domain=math 且含 \\boxed 的 → 数学(强模型长推理, SFT+RL)。难/长 → 阶段2-3。"""
    if r.get("domain") != "math":
        return None
    conv = r.get("conversations") or []
    if len(conv) < 2:
        return None
    q, sol = conv[0].get("value", ""), conv[-1].get("value", "")
    ans = extract_boxed(sol)
    if not q or not ans:
        return None
    # 蒸馏未验证答案 → 仅 SFT（学长推理风格，不进 RL 当奖励）
    return Record(prompt=make_prompt(q), data_source="openthoughts3-math", ability="math", use="sft",
                  reward_style="none", ground_truth="", gold_response=wrap_think_boxed(sol, ans),
                  difficulty="hard", source=f"ot3:{r.get('source', '')}", question=q)


_BESPOKE_PREFIX = re.compile(r"^Return your final response within \\boxed\{\}\.\s*")


def adapt_bespoke(r: dict) -> Optional[Record]:
    """Bespoke-Stratos：R1 风格长推理，题目要求 boxed。SFT+RL，难 → 阶段2-3。"""
    conv = r.get("conversations") or []
    if len(conv) < 2:
        return None
    q = _BESPOKE_PREFIX.sub("", conv[0].get("value", ""))
    sol = conv[-1].get("value", "")
    ans = extract_boxed(sol)
    if not q or not ans:
        return None
    # 蒸馏(curated)→ 仅 SFT（R1 长推理风格）
    return Record(prompt=make_prompt(q), data_source="bespoke-stratos", ability="math", use="sft",
                  reward_style="none", ground_truth="", gold_response=wrap_think_boxed(sol, ans),
                  difficulty="hard", source="bespoke", question=q)


_GADGET = re.compile(r"</?gadget[^>]*>|</?output>")

# ---------------- 中文数学池（2026-06-10 审计 P1：组建中文数学 SFT，详见 data_audit_report §五-P1.5）
_CHAIN_STEP = re.compile(r'<gadget id="calculator">(.*?)</gadget>\s*<output>(.*?)</output>', re.S)
_OP_ZH = [("*", " × "), ("/", " ÷ "), ("+", " + "), ("-", " - ")]


def _eval_expr(expr: str) -> Optional[float]:
    e = expr.replace("%", "/100").replace("^", "**").replace(" ", "")
    if not re.fullmatch(r"[-+*/().\d]+", e.replace("**", "")):
        return None
    try:
        return float(eval(e, {"__builtins__": {}}))
    except Exception:
        return None


def _parse_out(s: str) -> Optional[float]:
    s = s.strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)(?:/(\d+(?:\.\d+)?))?", s.replace(",", ""))
    if not m:
        return None
    v = float(m.group(1))
    return v / float(m.group(2)) if m.group(2) else v


def adapt_calc_ape(r: dict) -> Optional[Record]:
    """APE210k（2026-06-10 重写）：chain 计算链 → **中文 worked CoT**(SFT+RL)。
    旧版只当 RL 题面用,浪费了 95% 自带分步链的最大中文数学资产。
    每步用 Python 复算校验,链不可验/退化题(x=常数)按旧逻辑降级或丢弃。"""
    q = r.get("question_chinese") or r.get("question", "")
    res = r.get("result_float", None)
    if not q or res is None:
        return None
    res = float(res)
    ans = str(int(res)) if res.is_integer() else format(res, ".6g")   # 洗 float 噪声(0.100000365→0.1)
    eq = str(r.get("equation", "") or "")
    steps = _CHAIN_STEP.findall(str(r.get("chain", "") or ""))
    # 退化题(无运算无步骤,多为"比较/判断"类,答案 0/1 无意义) → 丢弃
    if not steps and not re.search(r"\d\s*[-+*/]", eq):
        return None
    gold = ""
    if steps:
        ok = True
        lines = []
        for i, (expr, out) in enumerate(steps, 1):
            ev, ov = _eval_expr(expr), _parse_out(out)
            if ev is None or ov is None or abs(ev - ov) > max(1e-6, abs(ov) * 1e-6):
                ok = False
                break
            disp = re.sub(r"\s+", "", expr).replace("**", "^")
            for a, b in _OP_ZH:
                disp = disp.replace(a, b)
            fr = re.search(r"-?\d+(?:\.\d+)?/\d+(?:\.\d+)?", out)   # output 取干净分数,丢 "around …" 噪声
            shown = fr.group(0) if fr else (str(int(ov)) if float(ov).is_integer() else str(round(ov, 6)))
            lines.append(f"第{i}步：{disp} = {shown}。")
        # 末步结果须与最终答案吻合,否则不当 SFT
        if ok and lines and abs(_parse_out(steps[-1][1]) - res) <= max(1e-6, abs(res) * 1e-6):
            # 列式行仅在 equation 能独立验算到最终答案时展示(个别行 equation 字段损坏,与 chain 矛盾)
            ev_eq = _eval_expr(eq[2:]) if eq.startswith("x=") else None
            head = f"列式：{eq}。\n" if (ev_eq is not None and abs(ev_eq - res) <= max(1e-6, abs(res) * 1e-6)
                                        and not re.fullmatch(r"x=\d+(?:\.\d+)?", eq)) else ""
            gold = "<think>\n" + head + "\n".join(lines) + f"\n所以答案是 {ans}。\n</think>\n#### \\boxed{{{ans}}}"
    return Record(prompt=make_prompt(q), data_source="calc-ape210k", ability="math",
                  use="both" if gold else "rl", reward_style="math_verify", ground_truth=ans,
                  gold_response=gold, difficulty="easy", source="ape210k", question=q)


_ZHR1_MATH_REPOS = {
    "EduChat-Math": "medium", "gavinluo/applied_math": "medium",
    "Haijian/Advanced-Math": "hard", "exam/kaoyan": "hard", "stem_zh/phy": "medium",
    # 'meta-math/GSM8K_zh' 故意排除：gsm8k 的中文翻译(含 test 段),跨语言泄漏 qhash 截不住
}


def adapt_chinese_r1_math(r: dict) -> Optional[Record]:
    """chinese-r1 数学子集（2026-06-10 新增）：score≥8(验证通过) + reasoning_content 包 think。
    此前整库在通用池只训 content,R1 think 轨迹全浪费——这是最优现成中文数学 CoT(~3万)。"""
    repo = str(r.get("repo_name", ""))
    if repo not in _ZHR1_MATH_REPOS:
        return None
    try:
        if float(r.get("score") or 0) < 8:
            return None
    except (ValueError, TypeError):
        return None
    q = r.get("input", "")
    think = str(r.get("reasoning_content", "") or "").strip()
    content = str(r.get("content", "") or "")
    ans = extract_boxed(content)
    if not q or not think or not ans:
        return None
    return Record(prompt=make_prompt(q), data_source="chinese-r1-math", ability="math",
                  use="both", reward_style="math_verify", ground_truth=ans,
                  gold_response=f"<think>\n{think}\n</think>\n#### \\boxed{{{ans}}}",
                  difficulty=_ZHR1_MATH_REPOS[repo], source=f"zhr1:{repo.split('/')[-1]}", question=q)


def adapt_dapo(r: dict) -> Optional[Record]:
    prompt = r.get("prompt")
    rm = r.get("reward_model") or {}
    gt = str(rm.get("ground_truth", "")).strip()
    if not prompt or not gt:
        return None
    # prompt 已是 chat list；取 user 题面做去重 key
    q = ""
    for m in (prompt if isinstance(prompt, list) else []):
        if m.get("role") == "user":
            q = m.get("content", "")
    q = DAPO_PREFIX.sub("", q).strip()   # 剥掉 "Solve the following... answer to the problem." 指令前缀
    return Record(prompt=make_prompt(q) if q else list(prompt), data_source="dapo-math-17k-dedup",
                  ability="math", use="rl", reward_style="math_verify", ground_truth=gt, gold_response="",
                  difficulty="", source="dapo", question=q or json.dumps(prompt, ensure_ascii=False)[:200])


# name -> (loader_fn, path, adapter_fn)
SOURCES = {
    # ① 自产 worked 算术（SFT 锚 + 可 RL）
    "compute_cot":     (iter_jsonl,     f"{COMPUTE_COT}/train.jsonl",                      adapt_compute_cot),
    # ② 各类数学题（SFT 主力，部分可 RL）
    "orca-math-200k":  (iter_arrow,     f"{FASTRL}/orca-math-200k",                        adapt_orca),
    "metamathqa":      (iter_json_array, f"{MATH_RAW}/metamathqa/MetaMathQA-395K.json",    adapt_metamath),
    "numinamath-1.5":  (iter_parquet,   f"{MATH_RAW}/numinamath-1.5/**/*.parquet",          adapt_numina),
    "openr1-math-220k":(iter_arrow,     f"{FASTRL}/openr1-math-220k/all",                   adapt_openr1),
    "deepscaler":      (iter_json_array, f"{MATH_RAW}/deepscaler-preview/deepscaler.json",
                        _adapt_problem_solution_answer("deepscaler-preview")),
    "gsm8k":           (iter_parquet,   f"{MATH_RAW}/gsm8k/main/train*.parquet",           adapt_gsm8k),
    "math-hendrycks":  (iter_hendrycks_train, "/data/zilu/fastrl/data/benchmark/math-hendrycks", adapt_math_hendrycks),
    # ② infinity 里含 boxed 的数学对话（之前被当通用浪费了，现捞进数学）
    "infinity-math":   (iter_parquet,   "/data/zilu/general_sft_raw/infinity-instruct/**/*.parquet", adapt_infinity_math),
    # ② 新增高质量源：OpenThoughts3(强模型长推理,难) / Bespoke-Stratos(R1,难) / APE210k(中文易)
    "openthoughts3-math": (iter_parquet, "/data/zilu/math_sft_raw/openthoughts3-1.2m/**/*.parquet",  adapt_openthoughts3_math),
    "bespoke-stratos": (iter_parquet,   "/data/zilu/math_sft_raw/bespoke-stratos-17k/**/*.parquet", adapt_bespoke),
    "calc-ape210k":    (iter_arrow_train, "/data/zilu/fastrl/data/train/calc-ape210k/original-splits", adapt_calc_ape),
    # ② 中文数学(2026-06-10 新增): chinese-r1 数学 repos,R1 think 直接复用
    "chinese-r1-math": (iter_arrow_train, f"{FASTRL}/chinese-deepseek-r1-distill",          adapt_chinese_r1_math),
    # ④ RL 池（仅可验证答案）
    "big-math":        (iter_parquet,   f"{MATH_RAW}/big-math-rl-verified/**/*.parquet",   adapt_big_math),
    "dapo":            (iter_parquet,   f"{MATH_RAW}/dapo-math-17k-dedup/**/*.parquet",     adapt_dapo),
}


def iter_source(name: str, limit: Optional[int] = None) -> Iterator[Record]:
    loader, path, adapt = SOURCES[name]
    n = 0
    for raw in loader(path):
        rec = adapt(raw)
        if rec is None:
            continue
        yield rec
        n += 1
        if limit and n >= limit:
            return
