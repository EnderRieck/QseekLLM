#!/usr/bin/env python3
"""Plot cc-reserved breakdown by source family."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("/data/zilu/QseekLLM/src/post_train/docs/benchmark")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "key": "qwen_base",
        "label": "Qwen Base",
        "summary": "/data/zilu/fastrl/checkpoints/external/qwen3_1_7b_base/final_eval_cc_reserved_default/summary.json",
    },
    {
        "key": "qwen_instruct",
        "label": "Qwen Instruct",
        "summary": "/data/zilu/fastrl/checkpoints/external/qwen3_1_7b/final_eval_cc_reserved_p1_default/summary.json",
    },
    {
        "key": "ours_s4",
        "label": "Ours S4",
        "summary": "/data/zilu/fastrl/checkpoints/sft_s4_anneal/global_step_1140_HFFIX/final_eval/summary.json",
    },
    {
        "key": "ours_rl_v2",
        "label": "Ours RL v2",
        "summary": "/data/zilu/fastrl/v3eval_hf/v2_gs300_hf/final_eval/summary.json",
    },
]


def load_cc(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["metrics"]["cc-reserved"]


def source_breakdown(metric: dict) -> dict:
    return metric.get("breakdown", {}).get("source", {})


def family_of(source: str) -> str:
    return source.split(".", 1)[0]


def aggregate_family(src: dict) -> dict[str, dict]:
    fam = {}
    for source, stat in src.items():
        name = family_of(source)
        fam.setdefault(name, {"correct": 0.0, "n": 0})
        fam[name]["correct"] += float(stat["acc"]) * int(stat["n"])
        fam[name]["n"] += int(stat["n"])
    return {k: {"acc": v["correct"] / v["n"], "n": v["n"]} for k, v in fam.items() if v["n"]}


def build_rows() -> list[dict]:
    by_model = {}
    for model in MODELS:
        by_model[model["key"]] = aggregate_family(source_breakdown(load_cc(model["summary"])))
    families = sorted(
        set().union(*(set(x) for x in by_model.values())),
        key=lambda f: (-max(by_model[m].get(f, {"n": 0})["n"] for m in by_model), f),
    )
    rows = []
    for fam in families:
        row = {"family": fam}
        row["n"] = max(by_model[m].get(fam, {"n": 0})["n"] for m in by_model)
        for model in MODELS:
            stat = by_model[model["key"]].get(fam)
            row[model["key"]] = None if stat is None else stat["acc"]
        rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> Path:
    path = OUT / "cc_reserved_family_breakdown.csv"
    fields = ["family", "n"] + [m["key"] for m in MODELS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_heatmap(rows: list[dict]) -> Path:
    rows = [r for r in rows if int(r["n"]) >= 40]
    labels = [r["family"].replace("_", " ") + f"  (n={int(r['n'])})" for r in rows]
    data = np.array([[float(r[m["key"]]) * 100 for m in MODELS] for r in rows])

    fig_h = max(8.0, 0.34 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([m["label"] for m in MODELS], fontsize=10)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("CC-reserved breakdown by source family (Pass@1)", fontsize=13, fontweight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val < 35 or val > 78 else "#222"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Pass@1 (%)")
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.tight_layout()
    path = OUT / "cc_reserved_family_heatmap.png"
    plt.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_delta(rows: list[dict]) -> Path:
    rows = [r for r in rows if int(r["n"]) >= 40]
    rows = sorted(rows, key=lambda r: float(r["ours_rl_v2"]) - float(r["ours_s4"]))
    labels = [r["family"].replace("_", " ") for r in rows]
    deltas = [(float(r["ours_rl_v2"]) - float(r["ours_s4"])) * 100 for r in rows]
    colors = ["#D6616B" if d < 0 else "#54A24B" for d in deltas]

    fig_h = max(8.0, 0.28 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    y = np.arange(len(rows))
    ax.barh(y, deltas, color=colors, alpha=0.9)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("RL v2 - S4 Pass@1 (percentage points)")
    ax.set_title("CC-reserved: RL change by source family", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    path = OUT / "cc_reserved_rl_delta.png"
    plt.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_md(rows: list[dict], heatmap: Path, delta: Path, csv_path: Path) -> Path:
    path = OUT / "cc_reserved_breakdown.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# CC-reserved Breakdown\n\n")
        f.write(f"- Family heatmap: `{heatmap.name}`\n")
        f.write(f"- RL delta plot: `{delta.name}`\n")
        f.write(f"- Raw family table: `{csv_path.name}`\n\n")
        f.write("| family | n | Qwen Base | Qwen Instruct | Ours S4 | Ours RL v2 |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            vals = [float(r[m["key"]]) * 100 for m in MODELS]
            f.write(
                f"| {r['family']} | {int(r['n'])} | "
                + " | ".join(f"{v:.1f}" for v in vals)
                + " |\n"
            )
    return path


def main() -> None:
    rows = build_rows()
    csv_path = write_csv(rows)
    heatmap = plot_heatmap(rows)
    delta = plot_delta(rows)
    md = write_md(rows, heatmap, delta, csv_path)
    print(f"[ok] {heatmap}")
    print(f"[ok] {delta}")
    print(f"[ok] {csv_path}")
    print(f"[ok] {md}")


if __name__ == "__main__":
    main()
