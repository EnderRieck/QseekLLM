#!/usr/bin/env python
"""Plot learning curves from TSV."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_curves_from_tsv(tsv_path: Path, output_path: Path):
    """Plot 4-panel learning curves from TSV."""
    df = pd.read_csv(tsv_path, sep="\t")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("300M v2 WSD 30B Training - Evaluation Curves", fontsize=16, fontweight="bold")

    # Chinese WPLC Perplexity
    ax = axes[0, 0]
    ax.plot(df["tokens_b"], df["chinese_ppl"], "o-", linewidth=2.5, markersize=7, color="#2E86AB", label="Chinese WPLC")
    ax.set_xlabel("Training Tokens (B)", fontsize=13)
    ax.set_ylabel("Perplexity", fontsize=13)
    ax.set_title("Chinese WPLC - Perplexity", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_yscale("log")
    ax.legend(fontsize=11)

    # Chinese WPLC Accuracy
    ax = axes[0, 1]
    ax.plot(df["tokens_b"], df["chinese_acc"], "o-", linewidth=2.5, markersize=7, color="#A23B72", label="Chinese WPLC")
    ax.set_xlabel("Training Tokens (B)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Chinese WPLC - Accuracy", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(y=20, color="gray", linestyle=":", alpha=0.6, linewidth=1.5, label="Random (~20%)")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(df["chinese_acc"]) * 1.2)

    # Lambada Perplexity
    ax = axes[1, 0]
    ax.plot(df["tokens_b"], df["lambada_ppl"], "o-", linewidth=2.5, markersize=7, color="#F18F01", label="Lambada OpenAI")
    ax.set_xlabel("Training Tokens (B)", fontsize=13)
    ax.set_ylabel("Perplexity", fontsize=13)
    ax.set_title("Lambada OpenAI - Perplexity", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_yscale("log")
    ax.legend(fontsize=11)

    # Lambada Accuracy
    ax = axes[1, 1]
    ax.plot(df["tokens_b"], df["lambada_acc"], "o-", linewidth=2.5, markersize=7, color="#C73E1D", label="Lambada OpenAI")
    ax.set_xlabel("Training Tokens (B)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Lambada OpenAI - Accuracy", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(y=25, color="gray", linestyle=":", alpha=0.6, linewidth=1.5, label="Random (~25%)")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(df["lambada_acc"]) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved plot to {output_path}")


def main():
    base_dir = Path("runs/stage1_general_300m_v2_wsd_30b")
    tsv_path = base_dir / "eval_results.tsv"
    output_path = base_dir / "eval_curves.png"

    if not tsv_path.exists():
        print(f"❌ TSV not found: {tsv_path}", file=sys.stderr)
        return 1

    plot_curves_from_tsv(tsv_path, output_path)

    # Print summary
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"\n📊 Summary (n={len(df)} checkpoints):")
    print(f"{'Tokens':<10} {'CN PPL':<12} {'CN Acc%':<10} {'EN PPL':<12} {'EN Acc%':<10}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{int(row['tokens_b']):>3}B      {row['chinese_ppl']:>10.2f}  {row['chinese_acc']:>8.2f}  {row['lambada_ppl']:>10.2f}  {row['lambada_acc']:>8.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
