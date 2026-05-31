#!/usr/bin/env python
"""Extract eval results and plot learning curves."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def extract_results(base_dir: Path) -> dict:
    """Extract results from all eval_*B subdirectories."""
    results = []
    for eval_dir in sorted(base_dir.glob("eval_*B")):
        results_file = eval_dir / "results.json"
        if not results_file.exists():
            print(f"Skipping {eval_dir.name}: no results.json", file=sys.stderr)
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

    return results


def plot_curves(results: list, output_path: Path):
    """Plot 4-panel learning curves."""
    if not results:
        print("No results to plot", file=sys.stderr)
        return

    tokens = [r["tokens_b"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("300M v2 WSD 30B Training - Eval Curves", fontsize=16, fontweight="bold")

    # Chinese WPLC Perplexity
    ax = axes[0, 0]
    ax.plot(tokens, [r["chinese_ppl"] for r in results], "o-", linewidth=2, markersize=6, color="#2E86AB")
    ax.set_xlabel("Training Tokens (B)", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("Chinese WPLC - Perplexity (lower is better)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Chinese WPLC Accuracy
    ax = axes[0, 1]
    ax.plot(tokens, [r["chinese_acc"] for r in results], "o-", linewidth=2, markersize=6, color="#A23B72")
    ax.set_xlabel("Training Tokens (B)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Chinese WPLC - Accuracy (higher is better)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="Random baseline (~20%)")
    ax.legend()

    # Lambada Perplexity
    ax = axes[1, 0]
    ax.plot(tokens, [r["lambada_ppl"] for r in results], "o-", linewidth=2, markersize=6, color="#F18F01")
    ax.set_xlabel("Training Tokens (B)", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("Lambada OpenAI - Perplexity (lower is better)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Lambada Accuracy
    ax = axes[1, 1]
    ax.plot(tokens, [r["lambada_acc"] for r in results], "o-", linewidth=2, markersize=6, color="#C73E1D")
    ax.set_xlabel("Training Tokens (B)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Lambada OpenAI - Accuracy (higher is better)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=25, color="gray", linestyle="--", alpha=0.5, label="Random baseline (~25%)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main():
    base_dir = Path("runs/stage1_general_300m_v2_wsd_30b")
    results = extract_results(base_dir)

    if not results:
        print("No eval results found", file=sys.stderr)
        return 1

    print(f"Found {len(results)} eval results:")
    print(f"{'Tokens':<10} {'CN PPL':<12} {'CN Acc%':<10} {'EN PPL':<12} {'EN Acc%':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['tokens_b']:>3}B      {r['chinese_ppl']:>10.2f}  {r['chinese_acc']:>8.2f}  {r['lambada_ppl']:>10.2f}  {r['lambada_acc']:>8.2f}")

    output_path = base_dir / "eval_curves.png"
    plot_curves(results, output_path)

    # Also save as TSV
    tsv_path = base_dir / "eval_results.tsv"
    with tsv_path.open("w") as f:
        f.write("tokens_b\tchinese_ppl\tchinese_acc\tlambada_ppl\tlambada_acc\n")
        for r in results:
            f.write(f"{r['tokens_b']}\t{r['chinese_ppl']:.4f}\t{r['chinese_acc']:.4f}\t{r['lambada_ppl']:.4f}\t{r['lambada_acc']:.4f}\n")
    print(f"Saved TSV to {tsv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
