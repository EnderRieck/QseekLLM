"""verl 自定义奖励函数：直接复用我们评测同口径的判分器 data_pipeline.reward.compute_reward。

verl naive reward manager 调用契约（verl/workers/reward_manager/naive.py）：
    compute_score(data_source, solution_str, ground_truth, extra_info) -> float | dict
返回 dict 时取 dict["score"] 作主奖励，其余键自动进 reward_extra_info（会被记进 metrics，
便于在监控里分别看 correct / has_format / thinking 长度 / 重复惩罚）。

style 路由：verl 只传 data_source，不传 reward_model.style。我们按 data_source 前缀映射到
判分 style；smoke 数据全是 big-math（math_verify），未知前缀兜底 math_verify。
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.reward import compute_reward

# data_source 前缀 -> 判分 style
_STYLE_BY_SRC = {
    "gsm8k": "gsm8k",
    "compute_cot": "compute_cot",
}


def _style_of(data_source: str) -> str:
    pre = (data_source or "").split(":")[0]
    return _STYLE_BY_SRC.get(pre, "math_verify")


# RL shaping 配比(v2 诊断后下调,见 docs/rl_degradation_diagnosis_20260613.md)。
# v1 用 compute_reward 默认 format=0.1 / think_len=0.2,实测让模型先薅满便宜的 format+
# 长度分、牺牲解题(easy 99.6%->73%、mid 36%->26%)。v2 砍 shaping 让 correct 主导;
# 经 env 注入便于 meta.env 追溯,且不改 compute_reward 共享默认(final_eval/SFT 自检仍用旧值)。
_FORMAT_BONUS = float(os.environ.get("REWARD_FORMAT_BONUS", "0.05"))
_THINK_LEN_MAX_BONUS = float(os.environ.get("REWARD_THINK_LEN_MAX_BONUS", "0.0"))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    style = _style_of(data_source)
    r = compute_reward(
        solution_str, ground_truth, style=style,
        format_bonus=_FORMAT_BONUS,
        think_len_max_bonus=_THINK_LEN_MAX_BONUS,
    )
    # 主奖励 = correct + format/length shaping - repetition penalty；额外暴露分项供监控聚合
    return {
        "score": r["reward"],
        "correct": float(r["correct"]),
        "has_format": float(r["has_format"]),
        "think_len_tokens": float(r["think_len_tokens"]),
        "think_len_bonus": float(r["think_len_bonus"]),
        "repeat_penalty": float(r["repeat_penalty"]),
    }
