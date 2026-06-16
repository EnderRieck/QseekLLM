"""统一训练样本格式 + 文本包装/抽取辅助。

设计依据 docs/hybrid_sft_rl_design.md §4：全部数据 → 单一对话模板，
输出统一为 `<think>\n逐步推演\n</think>\n#### \\boxed{answer}`。
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

# 统一系统提示：引导"先 <think> 逐步、再 #### \boxed{}"。中英各一版，按题面语言选。
SYS_EN = (
    "Solve the problem step by step. Put your full step-by-step reasoning between "
    "<think> and </think>, expanding every multi-digit calculation; then give the final "
    "answer on a new line as #### \\boxed{ANSWER}."
)
SYS_ZH = (
    "请一步一步解题。把完整的逐步推演写在 <think> 和 </think> 之间，多位数运算要逐步展开；"
    "然后另起一行用 #### \\boxed{答案} 给出最终答案。"
)

THINK_BOXED_RE = re.compile(r"^<think>\n.*\n</think>\n#### \\boxed\{(.*)\}$", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{(.*)\}\s*$", re.DOTALL)


@dataclass
class Record:
    """统一训练记录。SFT 用 gold_response；RL 用 reward_model.ground_truth；可二者皆有。"""
    prompt: list           # [{"role":"system",...},{"role":"user",...}]
    data_source: str       # 来源标识（含细粒度 source，用于验证器路由 + 监控分桶）
    ability: str           # "math" | "general"
    use: str               # "sft" | "rl" | "both"
    reward_style: str = "" # 验证器选择：compute_cot|gsm8k|math_verify|math_dapo|none
    ground_truth: str = "" # 可验证答案（RL 必需）
    gold_response: str = ""# 金标准解答（统一格式，SFT 必需）
    difficulty: str = ""   # easy|medium|hard|<数值/分层>
    source: str = ""       # 原始细分来源
    question: str = ""      # 题面纯文本（去重用）

    def to_json_obj(self) -> dict:
        return {
            "prompt": self.prompt,
            "data_source": self.data_source,
            "ability": self.ability,
            "use": self.use,
            "reward_model": {"ground_truth": self.ground_truth, "style": self.reward_style},
            "gold_response": self.gold_response,
            "extra_info": {"difficulty": self.difficulty, "source": self.source},
        }


def detect_lang(text: str) -> str:
    """粗判中英：含较多 CJK 视为中文。"""
    cjk = sum(1 for c in text[:200] if "一" <= c <= "鿿")
    return "zh" if cjk >= 5 else "en"


def make_prompt(question: str) -> list:
    sys = SYS_ZH if detect_lang(question) == "zh" else SYS_EN
    return [{"role": "system", "content": sys}, {"role": "user", "content": question.strip()}]


def wrap_think_boxed(reasoning: str, answer: str) -> str:
    """把(推演, 答案)包成统一输出格式。reasoning 已含/不含 <think> 都能处理，并清洗脏标记。"""
    r = reasoning.strip()
    if THINK_BOXED_RE.match(r):
        return r
    # 去掉已有的 think 包裹
    r = re.sub(r"^<think>\s*", "", r)
    r = re.sub(r"\s*</think>.*$", "", r, flags=re.DOTALL)
    # 清洗：R1/蒸馏的特殊标记(<|begin_of_thought|>等)、calculator 标签、gsm8k 的 <<>>、内部 #### 行、尾部 boxed
    r = re.sub(r"<\|[^|>]*\|>", "", r)
    r = re.sub(r"</?(?:gadget|output|result)[^>]*>", "", r)
    r = re.sub(r"<<[^>]*>>", "", r)
    r = re.sub(r"(?m)^\s*#{2,}\s.*$", "", r)        # 去掉 '#### ...' 这类行
    r = re.sub(r"\n*#+\s*\\?boxed\{.*\}\s*$", "", r, flags=re.DOTALL)
    r = re.sub(r"\n{3,}", "\n\n", r).strip()
    return f"<think>\n{r}\n</think>\n#### \\boxed{{{answer}}}"


def extract_boxed(text: str) -> Optional[str]:
    """抽取最后一个 \\boxed{...} 的内容，**平衡括号匹配**（正确处理嵌套 \\text{}/\\frac{}{}）。"""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def extract_gsm8k_answer(answer_field: str) -> Optional[str]:
    """gsm8k answer 末行 '#### N'。"""
    m = re.search(r"####\s*(.+?)\s*$", answer_field.strip())
    return m.group(1).strip().replace(",", "") if m else None


def extract_metamath_answer(response: str) -> Optional[str]:
    """MetaMathQA response 末尾 'The answer is: X'。"""
    m = re.search(r"[Tt]he answer is:?\s*\$?(.+?)\$?\s*\.?\s*$", response.strip())
    if not m:
        return None
    a = m.group(1).strip().rstrip(".").strip("$").strip()
    # 去掉可能的尾随文字（如 "units apart"）
    a = re.split(r"\s+(?:units|个|的|是)\b", a)[0].strip()
    return a or None


def extract_hashtag_or_boxed(text: str) -> Optional[str]:
    return extract_boxed(text) or extract_gsm8k_answer(text)
