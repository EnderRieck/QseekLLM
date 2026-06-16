"""解析 GRPO smoke 的 tensorboard tfevents,汇总时间拆解 + reward 趋势。

CLAUDE.md 要求记录:采样生成 / 训练 fwd-bwd / 权重同步 等各部分耗时。
verl 的 timing_s/* 标量即各阶段墙钟:
  gen          = rollout 采样生成(vLLM)
  old_log_prob = 重算 rollout 的 logprob(actor 前向)
  ref          = ref 模型 logprob(KL 用)
  adv          = 优势/奖励计算(含我们的判分器)
  update_actor = actor 训练 fwd+bwd+optim
  step         = 整步墙钟
权重同步(FSDP→vLLM)在 v0.5.0 内含于 gen 阶段的 wake/update_weights,
  若单列则看 timing_s/ 下相关键(本脚本自动打印所有 timing_s/*)。

用法: python RL/parse_timing.py [tb_dir]
"""
import glob
import csv
import sys
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

NON_TIME_TIMING_SUFFIXES = (
    "/num_preempted",
    "/num_preempted/min",
    "/num_preempted/max",
    "/num_preempted/mean",
    "/prompt_length",
    "/response_length",
)


def mean(xs):
    return sum(xs) / len(xs)


def is_timing_seconds(tag):
    """verl 0.8 async also stores diagnostic scalars under timing_s/."""
    return tag.startswith("timing_s/") and not tag.endswith(NON_TIME_TIMING_SUFFIXES)


tb_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/tb_grpo_smoke"
gpu_csv = sys.argv[2] if len(sys.argv) > 2 else None
files = sorted(glob.glob(f"{tb_dir}/events.out.tfevents.*"))
if not files:
    print("无 tfevents:", tb_dir); sys.exit(1)

ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]

# 按 step 收集
by_step = defaultdict(dict)
for t in tags:
    for ev in ea.Scalars(t):
        by_step[ev.step][t] = ev.value
steps = sorted(by_step)
print(f"tb: {tb_dir}  标量 tag 数={len(tags)}  step 数={len(steps)} (steps {steps[:1]}..{steps[-1:]})\n")

# ---- 时间拆解(训练步,取均值,跳过 step0 验证暖机)----
timing_tags = [t for t in tags if is_timing_seconds(t)]
diag_tags = [t for t in tags if t.startswith("timing_s/") and not is_timing_seconds(t)]
train_steps = [s for s in steps if any(tt in by_step[s] for tt in timing_tags)]
print("=== 时间拆解 timing_s/*（训练步均值，秒）===")
agg = defaultdict(list)
for s in train_steps:
    for t in timing_tags:
        if t in by_step[s]:
            agg[t].append(by_step[s][t])
step_total = mean(agg["timing_s/step"]) if "timing_s/step" in agg else None
for t in sorted(agg, key=lambda x: -mean(agg[x])):
    v = mean(agg[t])
    name = t.replace("timing_s/", "")
    pct = f"{v / step_total * 100:5.1f}%" if step_total else "  -  "
    print(f"  {name:24s} {v:8.2f}s  占整步 {pct}  (n={len(agg[t])})")

if diag_tags:
    print("\n=== timing_s/* 诊断标量（非耗时，不参与占比）===")
    diag_agg = defaultdict(list)
    for s in train_steps:
        for t in diag_tags:
            if t in by_step[s]:
                diag_agg[t].append(by_step[s][t])
    for t in sorted(diag_agg):
        vals = diag_agg[t]
        name = t.replace("timing_s/", "")
        print(f"  {name:40s} mean={mean(vals):8.2f} min={min(vals):8.2f} max={max(vals):8.2f} (n={len(vals)})")

print("\n=== worker 内部显存 perf/*（训练步，GB）===")
perf_tags = [
    t for t in tags if (t.startswith("perf/") or "/perf/" in t) and ("memory" in t or "cpu_memory" in t)
]
perf_agg = defaultdict(list)
for s in train_steps:
    for t in perf_tags:
        if t in by_step[s]:
            perf_agg[t].append(by_step[s][t])
if perf_agg:
    for t in sorted(perf_agg):
        vals = perf_agg[t]
        print(f"  {t:36s} mean={mean(vals):8.2f} max={max(vals):8.2f} (n={len(vals)})")
else:
    print("  无 perf/*memory* 标量")

if gpu_csv:
    print("\n=== nvidia-smi 外部采样峰值 ===")
    gpu_rows = []
    try:
        with open(gpu_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned = {k.strip(): v.strip() for k, v in row.items() if k is not None}
                try:
                    cleaned["index"] = int(cleaned["index"])
                    cleaned["memory.used [MiB]"] = float(cleaned["memory.used [MiB]"])
                    cleaned["memory.total [MiB]"] = float(cleaned["memory.total [MiB]"])
                    cleaned["utilization.gpu [%]"] = float(cleaned["utilization.gpu [%]"])
                except (KeyError, ValueError):
                    continue
                gpu_rows.append(cleaned)
    except FileNotFoundError:
        gpu_rows = []

    if gpu_rows:
        by_gpu = defaultdict(list)
        for row in gpu_rows:
            by_gpu[row["index"]].append(row)
        for idx in sorted(by_gpu):
            rows = by_gpu[idx]
            name = rows[-1].get("name", "")
            max_mem = max(r["memory.used [MiB]"] for r in rows)
            total_mem = max(r["memory.total [MiB]"] for r in rows)
            max_util = max(r["utilization.gpu [%]"] for r in rows)
            mean_util = mean([r["utilization.gpu [%]"] for r in rows])
            print(
                f"  gpu{idx} {name:24s} max_mem={max_mem/1024:7.2f}/{total_mem/1024:7.2f}GB "
                f"max_util={max_util:5.1f}% mean_util={mean_util:5.1f}% samples={len(rows)}"
            )
    else:
        print(f"  无法解析: {gpu_csv}")

# ---- reward / correct 趋势 ----
def trend(tag_sub):
    cand = [t for t in tags if tag_sub in t and ("reward" in t or "score" in t or "correct" in t or "format" in t)]
    return cand

print("\n=== reward / correct 趋势(训练 rollout)===")
for key in ["critic/rewards/mean", "critic/score/mean", "reward", "actor/reward"]:
    hit = [t for t in tags if t == key or t.endswith(key)]
    for t in hit:
        vals = [(s, by_step[s][t]) for s in steps if t in by_step[s]]
        if vals:
            head = " ".join(f"{v:.3f}" for _, v in vals[:12])
            print(f"  {t}: {head}")

print("\n=== 验证指标(step0 基线 + 末次)===")
for t in tags:
    if t.startswith("val-"):
        vals = [(s, by_step[s][t]) for s in steps if t in by_step[s]]
        if vals:
            print(f"  {t}: " + " ".join(f"s{s}={v:.3f}" for s, v in vals))
