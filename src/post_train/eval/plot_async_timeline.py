#!/usr/bin/env python3
"""叠加 A(events.jsonl 语义事件)+ B(gpu.csv 硬件util)成对齐时间轴。"""
import json, time, csv, sys
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

B = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/grpo_v2curric_s3r1_normadvF_kl005_noshaping_n16_t12_rollout4_steps5850"

# ---- A: events ----
ev = [json.loads(l) for l in open(f"{B}_events.jsonl")]
T0 = min(e["ts"] for e in ev)              # 公共原点
def m(ts): return (ts - T0) / 60.0          # -> 分钟

train, sync, val = [], [], []
st = {}
for e in ev:
    k = e["ev"]
    if k == "update_actor_start": st["tr"] = e["ts"]
    elif k == "update_actor_end" and "tr" in st: train.append((st.pop("tr"), e["ts"]))
    elif k == "param_sync_start": st["ps"] = e["ts"]
    elif k == "param_sync_end" and "ps" in st: sync.append((st.pop("ps"), e["ts"]))
    elif k == "validate_start": st["va"] = e["ts"]
    elif k == "validate_end" and "va" in st: val.append((st.pop("va"), e["ts"]))

# ---- B: gpu.csv -> per-card (minute, util) ----
def to_ep(s):
    s = s.strip()
    base = time.mktime(datetime.strptime(s[:19], "%Y/%m/%d %H:%M:%S").timetuple())
    return base + (float("0" + s[19:]) if "." in s else 0)
series = {1: ([], []), 3: ([], []), 4: ([], []), 5: ([], []), 6: ([], []), 7: ([], [])}
for r in csv.reader(open(f"{B}_gpu.csv")):
    if not r or r[0].strip() == "timestamp": continue
    try: idx = int(r[1]); util = float(r[5])
    except: continue
    if idx in series:
        series[idx][0].append(m(to_ep(r[0]))); series[idx][1].append(util)

def avg(cards):
    xs = series[cards[0]][0]
    ys = [sum(series[c][1][i] for c in cards) / len(cards) for i in range(len(xs))]
    return xs, ys
ax_actor = series[1]; ax_ref = series[3]; ax_roll = avg([4, 5, 6, 7])

# ===== 图1:全程总览 —— 每个角色一条独立子图 =====
def smooth(xs, ys, win=15):
    if len(ys) < win: return xs, ys
    out = []
    half = win // 2
    for i in range(len(ys)):
        a, b = max(0, i - half), min(len(ys), i + half + 1)
        out.append(sum(ys[a:b]) / (b - a))
    return xs, out

XMAX = max(ax_actor[0])
panels = [("actor / train (card1)", ax_actor, "#d62728", "busy~78%  ← bottleneck"),
          ("ref (card3)",           ax_ref,   "#2ca02c", "busy~73%"),
          ("rollout mean (card4-7)", ax_roll, "#1f77b4", "busy~44%  ← lots of idle")]
fig, axes = plt.subplots(4, 1, figsize=(16, 9), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1, 1, 0.5]})
for axp, (name, (xs, ys), col, note) in zip(axes[:3], panels):
    axp.plot(xs, ys, lw=0.3, color=col, alpha=0.35)               # 原始
    axp.plot(*smooth(xs, ys), lw=1.1, color=col)                  # 滑动平均
    for s, e in val:                                             # validate 橙带
        axp.axvspan(m(s), m(e), color="orange", alpha=0.22, zorder=0)
    axp.set_ylim(0, 105); axp.set_ylabel("util %")
    axp.text(0.004, 0.86, f"{name}   {note}", transform=axp.transAxes,
             fontsize=10, color=col, fontweight="bold")
# 第4栏:事件泳道(actor/train 区间 + param_sync tick + validate)
axe = axes[3]
for s, e in train: axe.barh(1, m(e) - m(s), left=m(s), height=0.55, color="#d62728")
for s, e in val:   axe.barh(0, m(e) - m(s), left=m(s), height=0.55, color="orange")
for s, e in sync:  axe.axvline(m(s), color="#9467bd", lw=0.25, alpha=0.5)
axe.set_yticks([0, 1]); axe.set_yticklabels(["validate", "train"])
axe.set_ylim(-0.5, 1.5); axe.set_xlim(0, XMAX)
axe.text(0.004, 0.78, "events.jsonl  (purple tick = param_sync x326)",
         transform=axe.transAxes, fontsize=9, color="#555")
axe.set_xlabel("minutes from start")
axes[0].set_title("Async-GRPO v2 full-run (13.7h) | per-role hardware util (gpu.csv) + semantic events")
plt.tight_layout(); plt.savefig("docs/async_timeline_full.png", dpi=110); plt.close()

# ===== 图2:局部放大(Gantt 泳道 + 三角色各一栏 util,取 ~3 个 param_sync 周期)=====
W0, W1 = 60, 66
# 第1栏 = 事件 Gantt;后3栏 = actor/ref/rollout 各自独立 util
fig, (axg, a_act, a_ref, a_roll) = plt.subplots(
    4, 1, figsize=(16, 8), sharex=True,
    gridspec_kw={"height_ratios": [0.9, 1, 1, 1]})
def bars(rows, y, color):
    for s, e in rows:
        a, b = m(s), m(e)
        if b < W0 or a > W1: continue
        axg.barh(y, b - a, left=a, height=0.6, color=color, edgecolor="none")
bars(train, 2, "#d62728"); bars(sync, 1, "#9467bd"); bars(val, 0, "orange")
axg.set_yticks([0, 1, 2])
axg.set_yticklabels(["validate", "param_sync", "actor/train"])
axg.set_ylim(-0.6, 2.6); axg.set_title(f"Zoom {W0}-{W1} min (~3 param_sync cycles) : semantic events (events.jsonl)")
axg.grid(axis="x", ls=":", alpha=0.4)
# param_sync 竖线投到每一栏,方便对齐周期
sync_x = [m(s) for s, e in sync if W0 <= m(s) <= W1]
for axp, (xs, ys), c, lb in [(a_act, ax_actor, "#d62728", "actor card1"),
                             (a_ref, ax_ref, "#2ca02c", "ref card3"),
                             (a_roll, ax_roll, "#1f77b4", "rollout card4-7")]:
    pts = [(x, y) for x, y in zip(xs, ys) if W0 <= x <= W1]
    if pts: axp.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.1, color=c)
    for sx in sync_x: axp.axvline(sx, color="#9467bd", lw=0.8, alpha=0.55, zorder=0)
    axp.set_ylim(0, 105); axp.set_xlim(W0, W1); axp.set_ylabel("util %")
    axp.grid(ls=":", alpha=0.4)
    axp.text(0.004, 0.84, lb, transform=axp.transAxes, fontsize=10,
             color=c, fontweight="bold")
a_act.set_title("hardware util (gpu.csv) — per role (purple line = param_sync)")
a_roll.set_xlabel("minutes from start")
plt.tight_layout(); plt.savefig("docs/async_timeline_zoom.png", dpi=110); plt.close()
print("saved docs/async_timeline_full.png , docs/async_timeline_zoom.png")
print(f"events: train={len(train)} sync={len(sync)} val={len(val)} | gpu samples/card={len(ax_actor[0])}")
