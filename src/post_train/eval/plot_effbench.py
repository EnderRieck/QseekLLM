#!/usr/bin/env python3
"""GRPO efficiency comparison plots.

Inputs (per variant):
  - {run}_gpu.csv     : nvidia-smi per-card util/mem samples (hardware truth)
  - {run}_phase.jsonl : marked_timer phase events (driver critical path)

Outputs (docs/effbench/):
  - {variant}_timeline.png : driver-phase Gantt + per-role (actor/ref/rollout) util
  - compare_concurrency.png: side-by-side "#roles busy at once" -> serial vs overlap
  - metrics.json           : sec/step, overlap, per-role util, sync count (training window)

Labels are English on purpose (matplotlib here has no CJK font). The analysis
write-up in docs/ is Chinese.
Run:  python3 eval/plot_effbench.py
"""
import csv
import json
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/data/zilu/QseekLLM/src/post_train/logs"
OUT = "/data/zilu/QseekLLM/src/post_train/docs/effbench"
os.makedirs(OUT, exist_ok=True)

VARIANTS = [
    dict(key="serial", label="(1) serial-sync GRPO",
         run="effbench_serial_rollout4", actor=1, ref=3, rollout=[4, 5, 6, 7], ref_colocated=False),
    dict(key="verlsplit_sync_ref", label="(2) verl fully-async split (sync ref on own card3)",
         run="effbench_verlsplit_refcard3_sync_ref_rollout4_steps5", actor=1, ref=3,
         rollout=[4, 5, 6, 7], ref_colocated=False),
    dict(key="ourasync", label="(3) our fully-async split (async ref on own card3)",
         run="effbench_ourasync_rollout4_steps5", actor=1, ref=3, rollout=[4, 5, 6, 7], ref_colocated=False),
]

LANE = {
    "gen": "rollout/gen", "generate_async": "rollout/gen",
    "ref": "ref", "ref_async_join": "ref", "compute_ref_log_prob": "ref",
    "update_actor": "actor-update", "update_actor_async": "actor-update",
    "param_sync": "weight-sync", "sync_rollout_weights": "weight-sync", "update_weights": "weight-sync",
    "reward": "reward", "old_log_prob": "old_logp",
}
LANES = ["rollout/gen", "ref", "actor-update", "weight-sync", "reward"]
LANE_COLOR = {"rollout/gen": "#1f77b4", "ref": "#2ca02c", "actor-update": "#d62728",
              "weight-sync": "#9467bd", "reward": "#ff7f0e"}
ROLE_COLOR = {"actor": "#d62728", "ref": "#2ca02c", "rollout": "#1f77b4"}
BUSY_THR = 25.0


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def to_ep(s):
    s = s.strip()
    base = time.mktime(datetime.strptime(s[:19], "%Y/%m/%d %H:%M:%S").timetuple())
    return base + (float("0" + s[19:]) if "." in s else 0.0)


def load_gpu(run):
    path = f"{LOG}/{run}_gpu.csv"
    if not os.path.exists(path):
        return None
    series = {}
    for r in csv.reader(open(path)):
        if not r or r[0].strip() == "timestamp":
            continue
        try:
            idx = int(r[1]); util = float(r[5]); t = to_ep(r[0])
        except Exception:
            continue
        series.setdefault(idx, ([], [])); series[idx][0].append(t); series[idx][1].append(util)
    return series


def load_phase(run):
    path = f"{LOG}/{run}_phase.jsonl"
    if not os.path.exists(path):
        return []
    evs = []
    for line in open(path):
        try:
            evs.append(json.loads(line))
        except Exception:
            pass
    spans, open_st = [], {}
    for e in evs:
        if e.get("ev") == "phase_start":
            open_st.setdefault(e["phase"], []).append(e["ts"])
        elif e.get("ev") == "phase_end" and open_st.get(e.get("phase")):
            spans.append((e["phase"], open_st[e["phase"]].pop(), e["ts"]))
    return spans


def role_series(series, v):
    def mean_cards(cards):
        cards = [c for c in cards if c in series]
        if not cards:
            return [], []
        xs = series[cards[0]][0]
        n = min(len(series[c][1]) for c in cards)
        ys = [sum(series[c][1][i] for c in cards) / len(cards) for i in range(n)]
        return xs[:n], ys
    actor = series.get(v["actor"], ([], []))
    rollout = mean_cards(v["rollout"])
    ref = series.get(v["ref"], ([], [])) if v["ref"] is not None else actor
    return actor, ref, rollout


def analyze(v):
    series = load_gpu(v["run"])
    if not series:
        return None
    spans = load_phase(v["run"])
    actor, ref, rollout = role_series(series, v)
    allt = [t for s in (actor, ref, rollout) for t in s[0]]
    if not allt:
        return None
    T0 = min(allt)
    steps = sorted([(s, e) for (p, s, e) in spans if p == "step" and e > s + 0.5])
    syncs = [(s, e) for (p, s, e) in spans if LANE.get(p) == "weight-sync"]
    # sec/step = steady-state cadence: median gap between consecutive step starts,
    # dropping the first interval (one-time warmup: queue / ref-service priming).
    # The raw step-span mean is misleading because the warmup step is 2x normal
    # and async warmup (ref-service priming) is even longer, masking steady-state gains.
    step_starts = sorted(s for (s, e) in steps)
    gaps = [step_starts[i + 1] - step_starts[i] for i in range(len(step_starts) - 1)]
    steady = gaps[1:] if len(gaps) > 1 else gaps
    sec_per_step = _median(steady) if steady else None
    # Steady-state window for plotting/statistics: skip the first measured step
    # when possible. Fully-async runs spend that first step priming queues, so
    # showing it as a normal timeline makes rollout look artificially long.
    if steps:
        plot_steps = steps[1:4] if len(steps) >= 4 else steps[:min(3, len(steps))]
        win0 = plot_steps[0][0] - 2.0
        win1 = plot_steps[-1][1] + 2.0
        train0, train1 = plot_steps[0][0], plot_steps[-1][1]
    else:
        win0, win1 = T0, max(allt)
        train0, train1 = T0, max(allt)
    return dict(v=v, series=series, actor=actor, ref=ref, rollout=rollout, spans=spans,
                T0=T0, steps=steps, syncs=syncs, sec_per_step=sec_per_step,
                win=(win0, win1), train=(train0, train1))


def _slice(s, t0, t1):
    return [(t, u) for t, u in zip(s[0], s[1]) if t0 <= t <= t1]


def mean_util(s, t0, t1):
    pts = _slice(s, t0, t1)
    return (sum(u for _, u in pts) / len(pts)) if pts else 0.0


def concurrency(a, t0, t1):
    """On a unified grid over [t0,t1]: #roles busy (0..3), overlap ratio, mean concurrency."""
    actor, ref, rollout = a["actor"], a["ref"], a["rollout"]
    grid = sorted(set(t for s in (actor, ref, rollout) for t in s[0] if t0 <= t <= t1))
    if not grid:
        return [], [], 0.0, 0.0
    def bset(s):
        return set(t for t, u in zip(s[0], s[1]) if u >= BUSY_THR)
    ba, br, bro = bset(actor), bset(ref), bset(rollout)
    conc, ov, robusy = [], 0, 0
    for t in grid:
        c = (t in ba) + (t in br) + (t in bro)
        conc.append(c)
        if t in bro:
            robusy += 1
            if (t in ba) or (t in br):
                ov += 1
    overlap = ov / robusy if robusy else 0.0
    meanc = sum(conc) / len(conc)
    return grid, conc, overlap, meanc


def plot_variant(a):
    v = a["v"]; T0 = a["T0"]; w0, w1 = a["win"]
    def m(t):
        return t - T0
    mw0, mw1 = m(w0 + T0 - T0), w1 - T0
    mw0 = w0 - T0
    fig, (axg, aa, ar, aro) = plt.subplots(
        4, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1, 1, 1]})
    lane_y = {ln: i for i, ln in enumerate(LANES)}
    for (p, s, e) in a["spans"]:
        ln = LANE.get(p)
        if ln not in lane_y:
            continue
        x0, x1 = m(s), m(e)
        if x1 < mw0 or x0 > mw1:
            continue
        axg.barh(lane_y[ln], max(x1 - x0, 0.08), left=x0, height=0.62,
                 color=LANE_COLOR.get(ln, "#999"), edgecolor="white", linewidth=0.4)
    axg.set_yticks(list(lane_y.values())); axg.set_yticklabels(LANES, fontsize=9)
    axg.set_ylim(-0.6, len(LANES) - 0.4); axg.set_xlim(mw0, mw1)
    sps = f"{a['sec_per_step']:.1f}s/step" if a["sec_per_step"] else "n/a"
    axg.set_title(f"{v['label']}   |   {sps}   |   steady-state driver phases",
                  fontsize=12, fontweight="bold")
    axg.grid(axis="x", ls=":", alpha=0.4)
    sync_x = [m(s) for s, e in a["syncs"] if mw0 <= m(s) <= mw1]
    for axp, role, s, note in [
        (aa, "actor", a["actor"], f"actor card{v['actor']} (A800)"),
        (ar, "ref", a["ref"], (f"ref card{v['ref']} (A4000)" if v["ref"] is not None
                               else f"ref ~ actor card{v['actor']} (COLOCATED, no own card)")),
        (aro, "rollout", a["rollout"], f"rollout card{v['rollout']} mean (A4000)")]:
        pts = [(m(t), u) for t, u in zip(s[0], s[1]) if mw0 <= m(t) <= mw1]
        if pts:
            axp.fill_between([p[0] for p in pts], [p[1] for p in pts], color=ROLE_COLOR[role], alpha=0.35)
            axp.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.1, color=ROLE_COLOR[role])
        axp.set_ylim(0, 105); axp.set_xlim(mw0, mw1); axp.set_ylabel("util %")
        for sx in sync_x:
            axp.axvline(sx, color="#9467bd", lw=0.9, alpha=0.5, zorder=0)
        mu = mean_util(s, *a["train"])
        axp.text(0.004, 0.82, f"{note}    mean(train)={mu:.0f}%",
                 transform=axp.transAxes, fontsize=10, color=ROLE_COLOR[role], fontweight="bold")
    aro.set_xlabel("seconds from start   (purple line = weight-sync)")
    plt.tight_layout()
    p = f"{OUT}/{v['key']}_timeline.png"
    plt.savefig(p, dpi=110); plt.close()
    return p


def _card_util(a, card, t0, t1):
    s = a["series"].get(card, ([], []))
    return mean_util(s, t0, t1)


def plot_compare(analyses):
    """Honest cross-variant view: per *physical card* utilisation (no role abstraction,
    so a colocated ref does not get double-counted) + sec/step. The card3 bar exposes
    'verl-native leaves card3 idle' vs 'ours keeps card3 busy with async ref-service'."""
    import numpy as np
    labels = [a["v"]["label"] for a in analyses]
    sps = [a["sec_per_step"] or 0 for a in analyses]
    # physical cards: card1 (actor / A800), card3 (ref / A4000), rollout mean (4-7)
    c1 = [_card_util(a, 1, *a["train"]) for a in analyses]
    c3 = [_card_util(a, 3, *a["train"]) for a in analyses]
    cro = [mean_util(a["rollout"], *a["train"]) for a in analyses]

    fig, (axu, axs) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(analyses)); w = 0.26
    b1 = axu.bar(x - w, c1, w, label="card1 actor (A800)", color="#d62728")
    b3 = axu.bar(x, c3, w, label="card3 ref (A4000)", color="#2ca02c")
    bo = axu.bar(x + w, cro, w, label="card4-7 rollout mean", color="#1f77b4")
    for bars in (b1, b3, bo):
        for r in bars:
            axu.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5,
                     f"{r.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    axu.set_xticks(x); axu.set_xticklabels([l.split(" ", 1)[-1] for l in labels], fontsize=9)
    axu.set_ylabel("mean GPU util % (training window)"); axu.set_ylim(0, 105)
    axu.legend(fontsize=9, loc="upper left")
    axu.set_title("Per-physical-card utilisation\n(note card3: verl-native=idle, ours=busy via async ref-service)",
                  fontsize=11, fontweight="bold")
    axu.grid(axis="y", ls=":", alpha=0.4)
    # sec/step bar
    base = sps[0] if sps and sps[0] else 1
    bar_cols = ["#999999", "#6699cc", "#88aa55", "#cc4444", "#cc8844"]
    bb = axs.bar(x, sps, 0.5, color=[bar_cols[i % len(bar_cols)] for i in range(len(sps))])
    for r, s in zip(bb, sps):
        axs.text(r.get_x() + r.get_width() / 2, r.get_height() + 1,
                 f"{s:.1f}s\n{base/s:.2f}x" if s else "n/a", ha="center", va="bottom", fontsize=9)
    short = [a["v"]["label"].split(")")[0] + ")" for a in analyses]
    axs.set_xticks(x); axs.set_xticklabels(short, fontsize=8)
    axs.set_ylabel("sec / step"); axs.set_ylim(0, max(sps) * 1.25)
    axs.set_title("Wall-clock per step\n(lower = faster)", fontsize=11, fontweight="bold")
    axs.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    p = f"{OUT}/compare_cards.png"
    plt.savefig(p, dpi=110); plt.close()
    return p


def main():
    analyses, metrics = [], {}
    for v in VARIANTS:
        a = analyze(v)
        if a is None:
            print(f"[skip] {v['key']}: missing data ({v['run']})")
            continue
        analyses.append(a)
        p = plot_variant(a)
        t0, t1 = a["train"]
        _, _, ov, meanc = concurrency(a, t0, t1)
        metrics[v["key"]] = dict(
            label=v["label"], sec_per_step=round(a["sec_per_step"], 1) if a["sec_per_step"] else None,
            n_steps=len(a["steps"]), n_syncs=len(a["syncs"]),
            overlap_ratio=round(ov, 3), mean_concurrency=round(meanc, 2),
            mean_util_actor=round(mean_util(a["actor"], t0, t1), 1),
            mean_util_ref=round(mean_util(a["ref"], t0, t1), 1),
            mean_util_rollout=round(mean_util(a["rollout"], t0, t1), 1),
            ref_colocated=v["ref_colocated"])
        print(f"[ok] {v['key']}: {p}")
    if analyses:
        print(f"[ok] compare: {plot_compare(analyses)}")
    json.dump(metrics, open(f"{OUT}/metrics.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
