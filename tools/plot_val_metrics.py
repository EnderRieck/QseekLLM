#!/usr/bin/env python3
"""Plot val_metrics.jsonl curves: per-source + overall CE/PPL vs consumed tokens.

Usage: python tools/plot_val_metrics.py --run-dir runs/stage1_general_300m_6b
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load(run_dir: Path) -> dict:
    path = run_dir / "val_metrics.jsonl"
    by_source: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            src = r["source"]
            by_source[src].append((r["consumed_tokens"], r["mean_ce"], r["ppl"]))
    return {k: sorted(v) for k, v in by_source.items()}


def plot(by_source: dict, out_dir: Path, title_suffix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    overall = by_source.pop("__overall__", None)
    sources = sorted(by_source.keys())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for src in sources:
        xs = [t / 1e9 for t, _, _ in by_source[src]]
        ces = [ce for _, ce, _ in by_source[src]]
        axes[0].plot(xs, ces, marker=".", label=src, alpha=0.75)
    if overall is not None:
        xs = [t / 1e9 for t, _, _ in overall]
        ces = [ce for _, ce, _ in overall]
        axes[0].plot(xs, ces, marker="o", color="black", linewidth=2.5, label="__overall__")
    axes[0].set_xlabel("Consumed tokens (B)")
    axes[0].set_ylabel("Validation CE")
    axes[0].set_title(f"Validation CE per source{title_suffix}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")

    for src in sources:
        xs = [t / 1e9 for t, _, _ in by_source[src]]
        ppls = [ppl for _, _, ppl in by_source[src]]
        axes[1].plot(xs, ppls, marker=".", label=src, alpha=0.75)
    if overall is not None:
        xs = [t / 1e9 for t, _, _ in overall]
        ppls = [ppl for _, _, ppl in overall]
        axes[1].plot(xs, ppls, marker="o", color="black", linewidth=2.5, label="__overall__")
    axes[1].set_xlabel("Consumed tokens (B)")
    axes[1].set_ylabel("Validation PPL (log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Validation PPL per source{title_suffix}")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=8, ncol=2, loc="upper right")

    fig.tight_layout()
    out_path = out_dir / "val_curves.png"
    fig.savefig(out_path, dpi=120)
    print(f"saved: {out_path}")

    if overall is not None:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        xs = [t / 1e9 for t, _, _ in overall]
        ces = [ce for _, ce, _ in overall]
        ppls = [ppl for _, _, ppl in overall]
        ax2.plot(xs, ces, marker="o", color="C0", label="overall CE")
        ax2.set_xlabel("Consumed tokens (B)")
        ax2.set_ylabel("CE", color="C0")
        ax2.tick_params(axis="y", labelcolor="C0")
        ax2.grid(True, alpha=0.3)
        ax2b = ax2.twinx()
        ax2b.plot(xs, ppls, marker="s", color="C3", label="overall PPL")
        ax2b.set_yscale("log")
        ax2b.set_ylabel("PPL (log)", color="C3")
        ax2b.tick_params(axis="y", labelcolor="C3")
        ax2.set_title(f"Overall validation curve{title_suffix}")
        out_path2 = out_dir / "val_overall.png"
        fig2.tight_layout()
        fig2.savefig(out_path2, dpi=120)
        print(f"saved: {out_path2}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    out_dir = args.out_dir or (args.run_dir / "plots")
    by_source = load(args.run_dir)
    plot(by_source, out_dir, title_suffix=f" ({args.title})" if args.title else "")


if __name__ == "__main__":
    main()
