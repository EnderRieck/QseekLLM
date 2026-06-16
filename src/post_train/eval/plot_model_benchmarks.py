#!/usr/bin/env python3
"""Plot final benchmark comparison for Qwen base, Qwen instruct, and ours.

Outputs are report-ready figures under docs/benchmark/.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("/data/zilu/QseekLLM/src/post_train/docs/benchmark")
OUT.mkdir(parents=True, exist_ok=True)

BENCHES = ["svamp", "gsm8k", "gsmplus", "cmath", "math500", "cc-reserved"]
BENCH_LABEL = {
    "svamp": "SVAMP",
    "gsm8k": "GSM8K",
    "gsmplus": "GSM-Plus",
    "cmath": "CMATH",
    "math500": "MATH500",
    "cc-reserved": "CC-reserved",
}
MODELS = [
    {
        "key": "qwen_base",
        "label": "Qwen3-1.7B Base",
        "color": "#4C78A8",
        "summary": "/data/zilu/fastrl/checkpoints/external/qwen3_1_7b_base/final_eval/summary.json",
        "overrides": [
            "/data/zilu/fastrl/checkpoints/external/qwen3_1_7b_base/final_eval_gsmplus/summary.json",
            "/data/zilu/fastrl/checkpoints/external/qwen3_1_7b_base/final_eval_cc_reserved_default/summary.json",
        ],
    },
    {
        "key": "qwen_instruct",
        "label": "Qwen3-1.7B Instruct",
        "color": "#F58518",
        "summary": "/data/zilu/fastrl/checkpoints/external/qwen3_1_7b/final_eval/summary.json",
        "overrides": [
            "/data/zilu/fastrl/checkpoints/external/qwen3_1_7b/final_eval_gsmplus/summary.json",
        ],
        # The cc Pass@1-only run was written by final_eval.py with k=1, whose
        # summary leaves pass@1 null. Recompute it from the per-sample dump.
        "cc_p1_dump": "/data/zilu/fastrl/checkpoints/external/qwen3_1_7b/final_eval_cc_reserved_p1_default/cc-reserved.jsonl",
        "cc_format_summary": "/data/zilu/fastrl/checkpoints/external/qwen3_1_7b/final_eval_cc_reserved_p1_default/summary.json",
    },
    {
        "key": "ours_s4",
        "label": "Ours (S4-1140)",
        "color": "#54A24B",
        "summary": "/data/zilu/fastrl/checkpoints/sft_s4_anneal/global_step_1140_HFFIX/final_eval/summary.json",
        "overrides": [],
    },
]

RL_MODEL = {
    "key": "ours_rl_v2",
    "label": "Ours RL (v2-gs300)",
    "color": "#B279A2",
    "summary": "/data/zilu/fastrl/v3eval_hf/v2_gs300_hf/final_eval/summary.json",
    "overrides": [],
}

ALL_MODELS = [
    *MODELS,
    {
        "key": "ours_rl_v2",
        "label": "Ours RL (v2-gs300)",
        "color": "#B279A2",
        "summary": "/data/zilu/fastrl/v3eval_hf/v2_gs300_hf/final_eval/summary.json",
        "overrides": [],
    },
]


def load_summary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["metrics"]


def merge_metrics(model: dict) -> dict:
    metrics = dict(load_summary(model["summary"]))
    for path in model.get("overrides", []):
        metrics.update(load_summary(path))

    if model.get("cc_p1_dump"):
        n = correct = 0
        with open(model["cc_p1_dump"], encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                n += 1
                correct += bool(rec.get("correct@1"))
        fmt = load_summary(model["cc_format_summary"])["cc-reserved"].get("format_rate")
        metrics["cc-reserved"] = {
            "n": n,
            "pass@1": correct / n if n else None,
            "pass@8": None,
            "format_rate": fmt,
        }
    return metrics


def build_table() -> list[dict]:
    rows = []
    for model in ALL_MODELS:
        metrics = merge_metrics(model)
        for bench in BENCHES:
            m = metrics.get(bench, {})
            rows.append(
                {
                    "model": model["label"],
                    "model_key": model["key"],
                    "benchmark": bench,
                    "benchmark_label": BENCH_LABEL[bench],
                    "n": m.get("n"),
                    "pass1": m.get("pass@1"),
                    "pass8": m.get("pass@8"),
                    "format_rate": m.get("format_rate"),
                }
            )
    return rows


def write_csv(rows: list[dict]) -> Path:
    path = OUT / "model_benchmark_scores.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _value(rows: list[dict], model_key: str, bench: str, metric: str) -> float:
    for row in rows:
        if row["model_key"] == model_key and row["benchmark"] == bench:
            val = row[metric]
            return float("nan") if val is None else float(val) * 100
    return float("nan")


def plot_grouped(rows: list[dict], metric: str, title: str, outfile: str) -> Path:
    x = np.arange(len(BENCHES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for i, model in enumerate(MODELS):
        vals = [_value(rows, model["key"], b, metric) for b in BENCHES]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            [0 if math.isnan(v) else v for v in vals],
            width,
            label=model["label"],
            color=model["color"],
            alpha=0.92,
        )
        for bar, val in zip(bars, vals):
            cx = bar.get_x() + bar.get_width() / 2
            if math.isnan(val):
                ax.text(cx, 2.0, "n/a", ha="center", va="bottom", fontsize=8, color="#555")
                bar.set_alpha(0.15)
                bar.set_hatch("//")
                continue
            if val >= 7:
                ax.text(cx, val + 1.3, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABEL[b] for b in BENCHES], fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.38)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    path = OUT / outfile
    plt.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_combined(rows: list[dict]) -> Path:
    x = np.arange(len(BENCHES))
    width = 0.25
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 11.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.9]})
    for ax, metric, title in [
        (axes[0], "pass1", "Pass@1: greedy accuracy"),
        (axes[1], "pass8", "Pass@8: sampled potential (k=8, T=0.8)"),
    ]:
        for i, model in enumerate(MODELS):
            vals = [_value(rows, model["key"], b, metric) for b in BENCHES]
            bars = ax.bar(
                x + (i - 1) * width,
                [0 if math.isnan(v) else v for v in vals],
                width,
                label=model["label"],
                color=model["color"],
                alpha=0.92,
            )
            for bar, val in zip(bars, vals):
                if math.isnan(val):
                    bar.set_alpha(0.15)
                    bar.set_hatch("//")
                elif val >= 8:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 1.0,
                        f"{val:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_ylim(0, 102)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax.grid(axis="y", linestyle=":", alpha=0.38)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([BENCH_LABEL[b] for b in BENCHES], fontsize=10)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)

    ax = axes[2]
    rl_width = 0.34
    for i, (metric, label, color) in enumerate([
        ("pass1", "RL Pass@1", "#B279A2"),
        ("pass8", "RL Pass@8", "#D6A5C8"),
    ]):
        vals = [_value(rows, RL_MODEL["key"], b, metric) for b in BENCHES]
        bars = ax.bar(x + (i - 0.5) * rl_width, vals, rl_width, label=label, color=color, alpha=0.94)
        for bar, val in zip(bars, vals):
            if not math.isnan(val) and val >= 7:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 102)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ours RL (v2-gs300): final evaluation", fontsize=12, fontweight="bold", loc="left")
    ax.grid(axis="y", linestyle=":", alpha=0.38)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABEL[b] for b in BENCHES], fontsize=10)

    axes[2].text(
        0.0,
        -0.24,
        "Note: Qwen Instruct cc-reserved Pass@8 was not run; hatched/blank bar marks missing data. "
        "The third row shows RL separately, not as an additional baseline column.",
        transform=axes[2].transAxes,
        fontsize=9,
        color="#555",
    )
    fig.suptitle("Benchmark comparison across models", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    path = OUT / "model_benchmark_comparison.png"
    plt.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_md(rows: list[dict], paths: dict[str, Path]) -> Path:
    path = OUT / "model_benchmark_comparison.md"
    by_model = {m["key"]: m["label"] for m in ALL_MODELS}
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Model Benchmark Comparison\n\n")
        f.write("Main rows compare Qwen3-1.7B Base, Qwen3-1.7B Instruct, and Ours (S4-1140). "
                "The third row reports Ours RL (v2-gs300) separately.\n\n")
        f.write(f"- Combined figure: `{paths['combined'].name}`\n")
        f.write(f"- Pass@1 figure: `{paths['pass1'].name}`\n")
        f.write(f"- Pass@8 figure: `{paths['pass8'].name}`\n")
        f.write(f"- Raw data: `{paths['csv'].name}`\n\n")
        for metric, title in [("pass1", "Pass@1"), ("pass8", "Pass@8")]:
            f.write(f"## {title} (%)\n\n")
            f.write("| benchmark | " + " | ".join(by_model.values()) + " |\n")
            f.write("|---|" + "|".join(["---:"] * len(MODELS)) + "|\n")
            for bench in BENCHES:
                vals = []
                for model in ALL_MODELS:
                    val = _value(rows, model["key"], bench, metric)
                    vals.append("—" if math.isnan(val) else f"{val:.1f}")
                f.write(f"| {BENCH_LABEL[bench]} | " + " | ".join(vals) + " |\n")
            f.write("\n")
        f.write(
            "Note: Qwen Instruct cc-reserved Pass@8 was not run; its Pass@1 is recomputed "
            "from the per-sample dump. Ours RL uses the v2-gs300 final-eval checkpoint.\n"
        )
    return path


def main() -> None:
    rows = build_table()
    csv_path = write_csv(rows)
    p1 = plot_grouped(rows, "pass1", "Final benchmark comparison: Pass@1", "model_benchmark_pass1.png")
    p8 = plot_grouped(rows, "pass8", "Final benchmark comparison: Pass@8", "model_benchmark_pass8.png")
    combined = plot_combined(rows)
    md = write_md(rows, {"csv": csv_path, "pass1": p1, "pass8": p8, "combined": combined})
    print(f"[ok] {combined}")
    print(f"[ok] {p1}")
    print(f"[ok] {p8}")
    print(f"[ok] {csv_path}")
    print(f"[ok] {md}")


if __name__ == "__main__":
    main()
