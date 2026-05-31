#!/usr/bin/env python
"""Quick preview of completed eval results."""
import json
from pathlib import Path

base_dir = Path("runs/stage1_general_300m_v2_wsd_30b")
results = []

for eval_dir in sorted(base_dir.glob("eval_*B")):
    results_file = eval_dir / "results.json"
    if not results_file.exists():
        continue

    tokens_str = eval_dir.name.replace("eval_", "").replace("B", "")
    tokens_b = int(tokens_str)

    with results_file.open() as f:
        data = json.load(f)

    results.append({
        "tokens_b": tokens_b,
        "chinese_ppl": data["results"]["chinese_wplc"]["perplexity,none"],
        "chinese_acc": data["results"]["chinese_wplc"]["acc,none"] * 100,
        "lambada_ppl": data["results"]["lambada_openai"]["perplexity,none"],
        "lambada_acc": data["results"]["lambada_openai"]["acc,none"] * 100,
    })

print(f"Completed: {len(results)}/15")
print(f"\n{'Tokens':<10} {'CN PPL':<12} {'CN Acc%':<10} {'EN PPL':<12} {'EN Acc%':<10}")
print("-" * 60)
for r in results:
    print(f"{r['tokens_b']:>3}B      {r['chinese_ppl']:>10.2f}  {r['chinese_acc']:>8.2f}  {r['lambada_ppl']:>10.2f}  {r['lambada_acc']:>8.2f}")
