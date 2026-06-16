from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "post_train"))

from data_pipeline.reward import compute_reward  # noqa: E402
from RL.reward_verl import compute_score  # noqa: E402


def test_thinking_length_bonus_is_bounded():
    short = "<think>short</think>\n#### \\boxed{42}"
    long_think = " ".join(f"step {i} reason carefully" for i in range(150))
    long = f"<think>{long_think}</think>\n#### \\boxed{{42}}"

    short_reward = compute_reward(short, "42", style="exact_match")
    long_reward = compute_reward(long, "42", style="exact_match")

    assert short_reward["think_len_bonus"] == 0.0
    assert 0.0 < long_reward["think_len_bonus"] <= 0.2
    assert long_reward["reward"] > short_reward["reward"]


def test_repetition_penalty_catches_looping_think():
    clean_think = " ".join(f"step {i} introduces a new equation" for i in range(90))
    repeated_think = " ".join(["repeat this exact reasoning phrase again"] * 40)
    clean = f"<think>{clean_think}</think>\n#### \\boxed{{42}}"
    repeated = f"<think>{repeated_think}</think>\n#### \\boxed{{42}}"

    clean_reward = compute_reward(clean, "42", style="exact_match")
    repeated_reward = compute_reward(repeated, "42", style="exact_match")

    assert clean_reward["repeat_penalty"] == 0.0
    assert repeated_reward["repeat_penalty"] > 0.0
    assert repeated_reward["reward"] < clean_reward["reward"]


def test_verl_reward_exposes_shaping_metrics():
    think = " ".join(f"step {i} reason carefully" for i in range(150))
    result = compute_score("compute_cot:test", f"<think>{think}</think>\n#### \\boxed{{42}}", "42")

    assert result["score"] > 1.0
    assert result["correct"] == 1.0
    assert result["has_format"] == 1.0
    assert result["think_len_tokens"] > 0.0
    assert result["think_len_bonus"] >= 0.0
    assert "repeat_penalty" in result
