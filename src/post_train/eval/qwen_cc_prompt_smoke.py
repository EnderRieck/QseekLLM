"""Prompt smoke for Qwen base on cc-reserved.

This is intentionally small: compare default/strict/few-shot prompt wording on
the same cc-reserved slice before running the full external baseline.
"""
from __future__ import annotations

import json
import os

from data_pipeline.format import SYS_EN
from eval.final_eval import _load_cc_reserved, generate_all, score_bench


HF_DIR = "/data/zilu/fastrl/checkpoints/external/qwen3_1_7b_base"
OUT_DIR = "/data/zilu/fastrl/checkpoints/external/qwen3_1_7b_base/final_eval_cc_smoke_prompts"


def _variants() -> dict[str, str]:
    strict_sys = (
        "Solve the math problem step by step. End with exactly one final line in the form "
        "#### \\boxed{ANSWER}. Use the same variable names as the problem. "
        'If there are multiple solutions, write them with "or", for example '
        "#### \\boxed{x=-3 or x=5}. Do not use comma-separated lists for multiple solutions."
    )
    fewshot_sys = (
        "Solve the math problem step by step. End with exactly one final line in the form "
        "#### \\boxed{ANSWER}. Use the same variable names as the problem. If there are "
        'multiple solutions, write them with "or" and include the variable name.\n\n'
        "Example:\n"
        "Problem: Solve |x - (2)| = 5.\n"
        "Solution: |x-2|=5 gives x-2=5 or x-2=-5, so x=7 or x=-3.\n"
        "#### \\boxed{x=-3 or x=7}\n\n"
        "Now solve the user problem."
    )
    return {"default": SYS_EN, "strict": strict_sys, "fewshot": fewshot_sys}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows_raw = _load_cc_reserved(20, 20260610)[:64]
    summary = {}
    for name, sys_prompt in _variants().items():
        rows = [
            {
                **r,
                "prompt": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": r["question"].strip()},
                ],
            }
            for r in rows_raw
        ]
        print(f"=== variant {name} ===", flush=True)
        gens = generate_all(
            rows,
            HF_DIR,
            [4],
            1,
            0.8,
            0.95,
            1024,
            32,
            0.75,
            os.path.join(OUT_DIR, f".tmp_{name}"),
        )
        records, metrics = score_bench("cc-reserved", rows, gens, 1)
        with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        boxed = sum("\\boxed" in r["gen_greedy"] for r in records) / max(1, len(records))
        hash_line = sum("####" in r["gen_greedy"] for r in records) / max(1, len(records))
        summary[name] = {**metrics, "boxed_rate": round(boxed, 4), "hash_rate": round(hash_line, 4)}
        print(name, summary[name], flush=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"done -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
