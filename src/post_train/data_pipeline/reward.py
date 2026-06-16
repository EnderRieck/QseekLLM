"""统一奖励/答案验证函数（hybrid_sft_rl_design.md §5）。

输入模型输出 + ground_truth + style，路由到对应验证器，返回 reward。
RL 阶段用作奖励；过程评测也用它算 per-source 准确率。

reward = 答案正确(主, 1/0) + format 奖励 + 有界 thinking 长度奖励 - 循环重复惩罚。
"""
from __future__ import annotations
import re
from typing import Optional

_BOXED = re.compile(r"\\boxed\{(.*)\}", re.DOTALL)
_HASH = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_THINK = re.compile(r"<think>.*</think>", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_REASONING_TOKEN = re.compile(r"\\[A-Za-z]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\s]")

# math_verify 库（sympy 等价判定）懒加载
_mv_parse = _mv_verify = None


def _load_mv():
    global _mv_parse, _mv_verify
    if _mv_parse is None:
        from math_verify import parse, verify
        _mv_parse, _mv_verify = parse, verify
    return _mv_parse, _mv_verify


def extract_pred(text: str) -> Optional[str]:
    """从模型输出抽最终答案：优先 \\boxed{}（平衡括号），其次 '#### x'。"""
    from data_pipeline.format import extract_boxed
    b = extract_boxed(text)
    if b is not None:
        return b
    hs = _HASH.findall(text)
    if hs:
        return hs[-1].strip().replace(",", "")
    return None


def _exact(pred: str, gold: str) -> bool:
    np = re.sub(r"\s+", "", pred).rstrip(".")
    ng = re.sub(r"\s+", "", str(gold)).rstrip(".")
    return np == ng


_MCQ = re.compile(r"\b([A-D])\b")
# 无格式输出只认明确作答声明,不扫尾部裸字母:数学正文里 sinA cosB / 点A 之类
# 会撞上 \b[A-D]\b 造成假阳(2026-06-12 gaokao-mathqa 审计:截断输出因公式里的
# 'A' 被判对)。声明取最后一处(模型可能中途改口)。
_MCQ_DECL = re.compile(
    r"(?:故选|答案[是为应选]?|正确(?:答案|选项)[是为]?|[Tt]he answer is|[Aa]nswer\s*[:：]|ANSWER\s*[:：])"
    r"\s*[:：.]?\s*[\(（]?([A-D])\b")
_MCQ_LASTLINE = re.compile(r"^[\(（]?([A-D])[\)）]?\s*[.。．]?$")


def _to_letter(s: str) -> str:
    """把 gold（可能是 'A'/'(A)'/索引 0-3/'[3]'）归一成字母 A-D。"""
    s = str(s).strip().strip("[]() ").strip()
    if s and s[0].upper() in "ABCD":
        return s[0].upper()
    try:
        return "ABCD"[int(float(s))]
    except (ValueError, IndexError):
        return s.upper()[:1]


# 多解答案的连接词(仅 compute_cot 风格的 gold 用这种模板,如 "x=-4 or x=26"、"(-7, 0) and (7, 0)")
_MULTI_SEP = re.compile(r"\s+(?:or|and)\s+")


def _split_tuple(s: str):
    """"(a, b)" 形式拆分量;不是该形式返回 None。"""
    s = s.strip()
    if s.startswith("(") and s.endswith(")") and "(" not in s[1:-1]:
        parts = [p.strip() for p in s[1:-1].split(",")]
        if len(parts) >= 2:
            return parts
    return None


_LEFT_RIGHT = re.compile(r"\\left\s*|\\right\s*")
_BASE_NUM = re.compile(r"^(\d+)_\{?(\d+)\}?$")


def _mv_parse_wrapped(s: str):
    """包上 $...$ 再交给 math_verify.parse。
    背景:parse 对不在数学环境里的裸字符串不走 LaTeX 解析,退化成"抽第一个
    数字"('6+9i'→6,'8 \\pi'→8,'4\\sqrt{5}'→4),于是首数字相同的错解会被
    判对(终评 math500 上 31/92 的采样救回是这么来的)。"""
    parse, _ = _load_mv()
    s = str(s).strip()
    if "$" not in s and "\\boxed" not in s:
        s = f"${s}$"
    return parse(s)


def _verify_single(pred: str, gt: str) -> bool:
    """单值判定:精确 → 数值容差 → 坐标元组逐分量 → sympy 等价。"""
    # \left( a, b \right) 与 (a, b) 同形,去掉定界符再走元组/等价判定
    pred = _LEFT_RIGHT.sub("", str(pred))
    gt = _LEFT_RIGHT.sub("", str(gt))
    if _exact(pred, gt):
        return True
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except (ValueError, TypeError):
        pass
    # 进制下标 "40_9"/"40_{10}":math_verify 解析时丢下标,只剩 40==40;须数字+底数都一致
    bg, bp = _BASE_NUM.match(gt), _BASE_NUM.match(pred)
    if bg or bp:
        return bool(bg and bp) and bg.groups() == bp.groups()
    # 坐标点 "(a, b)":math_verify 会当成开区间,(8,0)/(7,0) 同为空区间误判相等,须逐分量比
    tg, tp = _split_tuple(gt), _split_tuple(pred)
    if tg is not None and tp is not None:
        return len(tg) == len(tp) and all(_verify_single(p, g) for p, g in zip(tp, tg))
    try:
        _, verify = _load_mv()
        return bool(verify(_mv_parse_wrapped(gt), _mv_parse_wrapped(pred)))
    except Exception:
        return False


def _verify_multi(pred: str, gt: str) -> bool:
    """多解判定:按 or/and 拆解集,逐项配对(顺序无关),解的个数必须一致。
    背景:math_verify 对 "x=-4 or x=26" 只解析出最后一个表达式,会把错解判对。"""
    parts_g = [p.strip() for p in _MULTI_SEP.split(gt) if p.strip()]
    parts_p = [p.strip() for p in _MULTI_SEP.split(pred) if p.strip()]
    if len(parts_p) != len(parts_g):
        return False
    used = [False] * len(parts_p)
    for g_ in parts_g:
        hit = False
        for i, p_ in enumerate(parts_p):
            if not used[i] and _verify_single(p_, g_):
                used[i] = hit = True
                break
        if not hit:
            return False
    return True


def verify_answer(pred_text: str, ground_truth: str, style: str = "math_verify") -> bool:
    """判断模型输出是否答对。"""
    gt = str(ground_truth)
    if style == "mcq":
        # 选择题：boxed/#### 内抽字母;无格式时只认明确作答声明或末行单独字母,
        # 不扫尾部裸字母(见 _MCQ_DECL 注释)
        pred = extract_pred(pred_text)
        if pred:
            m = _MCQ.search(pred)
            return bool(m) and m.group(1) == _to_letter(gt)
        decls = _MCQ_DECL.findall(pred_text)
        if decls:
            return decls[-1] == _to_letter(gt)
        lines = [l.strip() for l in pred_text.strip().splitlines() if l.strip()]
        m = _MCQ_LASTLINE.match(lines[-1]) if lines else None
        return bool(m) and m.group(1) == _to_letter(gt)
    if style == "exact_match":
        pred = extract_pred(pred_text) or pred_text.strip().splitlines()[-1] if pred_text.strip() else ""
        return _exact(str(pred), gt) or re.sub(r"\s+", "", str(pred)).lower() == re.sub(r"\s+", "", gt).lower()
    pred = extract_pred(pred_text)
    if pred is None or ground_truth is None or ground_truth == "":
        return False
    if style == "compute_cot" and _MULTI_SEP.search(gt):
        return _verify_multi(pred, gt)
    if style in ("gsm8k", "compute_cot"):
        if _exact(pred, gt):
            return True
        # 数值容差（compute_cot/gsm8k 多为精确数值/表达式）
        try:
            return abs(float(pred) - float(gt)) < 1e-6
        except (ValueError, TypeError):
            pass
        # 回退 sympy 等价
    # math_verify / math_dapo / 其余 → sympy 符号等价（处理 LaTeX/分数/区间/集合）
    # 统一走 _verify_single:精确 → 数值容差 → 元组逐分量 → $...$ 包裹后 sympy 等价
    return _verify_single(pred, gt)


def extract_think(text: str) -> str:
    """抽取第一段 <think>...</think> 内容；没有则返回空串。"""
    m = _THINK_BLOCK.search(text or "")
    return m.group(1).strip() if m else ""


def _count_reasoning_tokens(text: str) -> int:
    """轻量 token 近似:英文词/数字/LaTeX 命令/中文字符/符号。"""
    return len(_REASONING_TOKEN.findall(text or ""))


def _think_length_bonus(think_text: str, max_bonus: float, min_tokens: int, target_tokens: int) -> tuple[float, int]:
    """给非空思考过程一个有界长度 shaping，超过 target 不再继续涨。"""
    n_tok = _count_reasoning_tokens(think_text)
    if max_bonus <= 0 or n_tok <= min_tokens or target_tokens <= min_tokens:
        return 0.0, n_tok
    progress = min(1.0, (n_tok - min_tokens) / float(target_tokens - min_tokens))
    return max_bonus * progress, n_tok


def _norm_repeat_text(text: str) -> list[str]:
    return [t.lower() for t in _REASONING_TOKEN.findall(text or "") if t.strip()]


def _repeat_penalty(text: str, max_penalty: float) -> float:
    """惩罚明显循环:重复长 ngram 或重复整行。"""
    if max_penalty <= 0:
        return 0.0

    penalty = 0.0

    lines = [re.sub(r"\s+", " ", x.strip().lower()) for x in (text or "").splitlines()]
    lines = [x for x in lines if len(x) >= 20]
    if lines:
        counts = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
        repeated_extra = sum(c - 1 for c in counts.values() if c >= 3)
        penalty += min(0.2, repeated_extra * 0.04)

    toks = _norm_repeat_text(text)
    n = 12
    if len(toks) >= n * 3:
        counts = {}
        for i in range(len(toks) - n + 1):
            gram = tuple(toks[i:i + n])
            counts[gram] = counts.get(gram, 0) + 1
        max_count = max(counts.values(), default=1)
        repeated_extra = sum(c - 1 for c in counts.values() if c >= 3)
        total = max(1, len(toks) - n + 1)
        repeat_frac = repeated_extra / total
        if max_count >= 3 or repeat_frac >= 0.04:
            penalty += min(0.3, max(0.0, max_count - 2) * 0.06 + min(0.18, repeat_frac * 1.8))

    return min(max_penalty, penalty)


def compute_reward(response_text: str, ground_truth: str, style: str = "math_verify",
                   format_bonus: float = 0.1,
                   think_len_max_bonus: float = 0.2,
                   think_len_min_tokens: int = 64,
                   think_len_target_tokens: int = 512,
                   repeat_max_penalty: float = 0.4) -> dict:
    """返回 reward 与可聚合分项。

    主信号仍是 correct；format/length/repetition 都是小幅 shaping。
    """
    correct = verify_answer(response_text, ground_truth, style)
    has_think = bool(_THINK.search(response_text))
    has_box = bool(_BOXED.search(response_text)) or bool(_HASH.search(response_text))
    has_format = has_think and has_box
    think_text = extract_think(response_text)
    think_len_bonus, think_len_tokens = _think_length_bonus(
        think_text if has_format else "",
        think_len_max_bonus,
        think_len_min_tokens,
        think_len_target_tokens,
    )
    repeat_penalty = _repeat_penalty(think_text or response_text, repeat_max_penalty)
    reward = (
        (1.0 if correct else 0.0)
        + (format_bonus if has_format else 0.0)
        + think_len_bonus
        - repeat_penalty
    )
    return {
        "reward": reward,
        "correct": correct,
        "has_format": has_format,
        "think_len_tokens": think_len_tokens,
        "think_len_bonus": think_len_bonus,
        "repeat_penalty": repeat_penalty,
    }


def _fmt_case_result(r: dict) -> str:
    return (
        f"correct={r['correct']} reward={r['reward']:.2f} "
        f"len={r['think_len_tokens']} len_bonus={r['think_len_bonus']:.2f} "
        f"repeat_penalty={r['repeat_penalty']:.2f}"
    )



if __name__ == "__main__":
    # 自测（从真文件跑，math_verify 多进程才正常）
    cases = [
        ("<think>...\n</think>\n#### \\boxed{72}", "72", "gsm8k", True),
        ("\\boxed{71}", "72", "math_verify", False),
        ("answer is \\boxed{1/2}", "0.5", "math_verify", True),
        ("the result is \\boxed{\\frac{1}{2}}", "1/2", "math_verify", True),
        ("\\boxed{(-\\infty, 2) \\cup (3, \\infty)}", "(-\\infty,2)\\cup(3,\\infty)", "math_verify", True),
        ("<think>step</think>\n#### \\boxed{x=5}", "x=5", "compute_cot", True),
        # 多解:全对才算对(防 math_verify 只看最后一个解)
        ("#### \\boxed{x=-6 or x=26}", "x=-4 or x=26", "compute_cot", False),
        ("#### \\boxed{x=26 or x=-4}", "x=-4 or x=26", "compute_cot", True),
        ("#### \\boxed{(-7, 0) and (8, 0)}", "(-7, 0) and (7, 0)", "compute_cot", False),
        ("#### \\boxed{(7, 0) and (-7, 0)}", "(-7, 0) and (7, 0)", "compute_cot", True),
        ("#### \\boxed{x=26}", "x=-4 or x=26", "compute_cot", False),
        # 回归:裸字符串退化成"抽第一个数字"导致的误判(2026-06-12,math500 终评审计)
        ("#### \\boxed{8}", "8 \\pi", "math_verify", False),
        ("#### \\boxed{4\\sqrt{5}}", "4", "math_verify", False),
        ("#### \\boxed{6-3i}", "6+9i", "math_verify", False),
        ("#### \\boxed{2\\sqrt{2}}", "2", "math_verify", False),
        ("#### \\boxed{1+2i}", "1+274i", "math_verify", False),
        ("#### \\boxed{40_{10}}", "40_9", "math_verify", False),
        ("#### \\boxed{2k-2}", "2k+2", "math_verify", False),
        ("#### \\boxed{(\\frac{3}{2}, \\pi-3)}", "\\left( 3, \\frac{\\pi}{2} \\right)", "math_verify", False),
        # 这些宽松判定应保持为对
        ("#### \\boxed{x=2}", "2", "math_verify", True),
        ("#### \\boxed{10\\%}", "10", "math_verify", True),
        ("#### \\boxed{8\\pi}", "8 \\pi", "math_verify", True),
        ("#### \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}", "\\left( 3, \\frac{\\pi}{2} \\right)", "math_verify", True),
        # mcq:无格式时不扫尾部裸字母(公式 sinA cosB 假阳回归),只认声明/末行单独字母
        ("推导 sin(A - B) = sinA cosB - cosA sinB。所以，sin(2x/4)cos(π/4) - cos(2x", "A", "mcq", False),
        ("#### \\boxed{B}", "B", "mcq", True),
        ("#### \\boxed{B}", "A", "mcq", False),
        ("分析过程很长……故选C", "C", "mcq", True),
        ("我先猜A,不对,正确答案是D", "D", "mcq", True),
        ("The answer is B.", "B", "mcq", True),
        ("一通分析之后\n(C)", "C", "mcq", True),
        ("分析到一半提到选项D但没有作答", "D", "mcq", False),
    ]
    for resp, gt, style, exp in cases:
        r = compute_reward(resp, gt, style)
        ok = "✓" if r["correct"] == exp else "✗ EXPECTED " + str(exp)
        print(f"{ok}  style={style:12s} gt={gt:30s} -> {_fmt_case_result(r)}")
