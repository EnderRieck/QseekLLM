"""Summarize synchronous GRPO rollout-card sweep into one markdown report."""

from __future__ import annotations

import csv
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LOG_DIR = Path("/data/zilu/QseekLLM/src/post_train/logs")
TIMING_PREFIX = "timing_s/"
NON_TIME_TIMING_SUFFIXES = (
    "/num_preempted",
    "/num_preempted/min",
    "/num_preempted/max",
    "/num_preempted/mean",
    "/prompt_length",
    "/response_length",
)
STAGE_ORDER = [
    "step",
    "gen",
    "old_log_prob",
    "ref",
    "adv",
    "update_actor",
    "update_weights",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def is_timing_seconds(tag: str) -> bool:
    return tag.startswith(TIMING_PREFIX) and not tag.endswith(NON_TIME_TIMING_SUFFIXES)


def read_meta(path: Path) -> dict[str, str]:
    meta = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def read_tb(tb_dir: Path) -> dict:
    files = sorted(glob.glob(str(tb_dir / "events.out.tfevents.*")))
    if not files:
        return {"missing": True, "tb_dir": str(tb_dir)}

    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]

    by_step: dict[int, dict[str, float]] = defaultdict(dict)
    for tag in tags:
        for event in ea.Scalars(tag):
            by_step[event.step][tag] = event.value

    timing_tags = [tag for tag in tags if is_timing_seconds(tag)]
    steps = sorted(by_step)
    train_steps = [step for step in steps if any(tag in by_step[step] for tag in timing_tags)]

    timing = defaultdict(list)
    per_step = []
    for step in train_steps:
        row = {"step_id": step}
        for tag in timing_tags:
            if tag in by_step[step]:
                name = tag.removeprefix(TIMING_PREFIX)
                value = by_step[step][tag]
                timing[name].append(value)
                row[name] = value
        per_step.append(row)

    perf_tags = [
        tag
        for tag in tags
        if (tag.startswith("perf/") or "/perf/" in tag) and ("memory" in tag or "cpu_memory" in tag)
    ]
    perf = defaultdict(list)
    for step in train_steps:
        for tag in perf_tags:
            if tag in by_step[step]:
                perf[tag].append(by_step[step][tag])

    reward = {}
    for tag in ["critic/rewards/mean", "critic/score/mean"]:
        vals = [by_step[step][tag] for step in steps if tag in by_step[step]]
        if vals:
            reward[tag] = vals

    return {
        "missing": False,
        "tb_dir": str(tb_dir),
        "tags": tags,
        "steps": steps,
        "train_steps": train_steps,
        "timing": timing,
        "per_step": per_step,
        "perf": perf,
        "reward": reward,
    }


def read_gpu_csv(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}

    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {key.strip(): value.strip() for key, value in row.items() if key is not None}
            try:
                cleaned["index"] = int(cleaned["index"])
                cleaned["memory.used [MiB]"] = float(cleaned["memory.used [MiB]"])
                cleaned["memory.total [MiB]"] = float(cleaned["memory.total [MiB]"])
                cleaned["utilization.gpu [%]"] = float(cleaned["utilization.gpu [%]"])
            except (KeyError, ValueError):
                continue
            rows.append(cleaned)

    by_gpu = defaultdict(list)
    for row in rows:
        by_gpu[row["index"]].append(row)

    summary = {}
    for idx, gpu_rows in by_gpu.items():
        summary[idx] = {
            "name": gpu_rows[-1].get("name", ""),
            "max_mem_gb": max(row["memory.used [MiB]"] for row in gpu_rows) / 1024,
            "total_mem_gb": max(row["memory.total [MiB]"] for row in gpu_rows) / 1024,
            "max_util": max(row["utilization.gpu [%]"] for row in gpu_rows),
            "mean_util": mean([row["utilization.gpu [%]"] for row in gpu_rows]),
            "samples": len(gpu_rows),
        }
    return summary


def read_weight_syncs(path: Path) -> list[float]:
    if not path.exists():
        return []
    pattern = re.compile(r"update_weights done, time cost: ([0-9.]+)s")
    return [float(match.group(1)) for match in pattern.finditer(path.read_text(errors="ignore"))]


def role_for_gpu(idx: int, meta: dict[str, str]) -> str:
    train = {int(x) for x in meta.get("train_card", "").split(",") if x}
    ref = {int(x) for x in meta.get("ref_cards", "").split(",") if x}
    rollout = {int(x) for x in meta.get("rollout_cards", "").split(",") if x}
    if idx in train:
        return "actor/update"
    if idx in ref:
        return "ref"
    if idx in rollout:
        return "rollout"
    return "idle/other"


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def timing_mean(run: dict, stage: str) -> float | None:
    vals = run.get("tb", {}).get("timing", {}).get(stage, [])
    return mean(vals) if vals else None


def build_report(run_group: str, runs: list[dict]) -> str:
    first_meta = next((run["meta"] for run in runs if run.get("meta")), {})
    train_card = first_meta.get("train_card", "1")
    ref_cards = first_meta.get("ref_cards", "?")
    ref_count = first_meta.get("ref_count", "?")
    rollout_pool = "4/5/6/7"
    logged_counts = "/".join(run["meta"].get("rollout_count", "?") for run in runs)

    lines = []
    lines.append(f"# 同步 GRPO 性能探索报告: {run_group}")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- 训练/update 卡: 物理 GPU{train_card}, NVIDIA A800-SXM4-80GB。")
    lines.append(f"- ref forward 卡: 物理 GPU{ref_cards}, NVIDIA RTX A4000；ref_count={ref_count}。")
    lines.append(f"- rollout 采样卡池: 物理 GPU{rollout_pool}, NVIDIA RTX A4000。")
    lines.append(f"- 本报告包含已产生日志的 rollout 卡数: {logged_counts}。")
    lines.append("- actor 全参训练: `actor_rollout_ref.model.lora_rank=0`。")
    lines.append("- rollout: vLLM standalone server, TP=1, 每张采样卡一个 replica。")
    lines.append("- old logprob: 使用 rollout/sample 侧 logprob bypass, 不额外做 actor old forward。")
    lines.append("- ref: 关闭 torch compile；实际 ref 卡数以各 run 的 meta 文件为准。")
    lines.append("- 每轮训练步数见各 run 的 meta 文件；默认脚本为 5 steps。")
    lines.append("")

    lines.append("## 关键文件")
    lines.append("")
    lines.append("- 入口: `src/post_train/RL/main_grpo_sync_split.py`")
    lines.append("- 单实验脚本: `src/post_train/RL/run_grpo_sync_a800_ref3_a4000_one.sh`")
    lines.append("- sweep 脚本: `src/post_train/RL/run_grpo_sync_a800_ref3_a4000_sweep.sh`")
    lines.append("- 单实验解析: `src/post_train/RL/parse_timing.py`")
    lines.append("- 报告汇总: `src/post_train/RL/summarize_grpo_sweep.py`")
    lines.append("")

    lines.append("## 阶段含义")
    lines.append("")
    lines.append("| 阶段 | 含义 |")
    lines.append("| --- | --- |")
    lines.append("| gen | rollout/vLLM 采样生成；本实验中也包含 rollout 侧 sampled-token logprob 返回。 |")
    lines.append("| old_log_prob | bypass 模式下复用 rollout_log_probs 写入 old_log_probs，预期接近 0。 |")
    lines.append("| ref | A4000 上 ref model forward，计算 KL 需要的 ref_log_prob。 |")
    lines.append("| adv | reward/advantage 计算与队列字段整理。 |")
    lines.append("| update_actor | A800 上 actor 训练 forward/backward/optimizer step。 |")
    lines.append("| update_weights | A800 actor 权重同步到 rollout vLLM replica。 |")
    lines.append("| step | 同步训练循环端到端墙钟。 |")
    lines.append("")

    lines.append("## 总览: 各 rollout 卡数平均耗时")
    lines.append("")
    lines.append("| rollout卡数 | rollout物理卡 | steps | step | gen | old_log_prob | ref | adv | update_actor | update_weights |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run in runs:
        tb = run["tb"]
        meta = run["meta"]
        lines.append(
            "| "
            + " | ".join(
                [
                    meta.get("rollout_count", "?"),
                    meta.get("rollout_cards", "?"),
                    str(len(tb.get("train_steps", []))) if not tb.get("missing") else "0",
                    fmt(timing_mean(run, "step")),
                    fmt(timing_mean(run, "gen")),
                    fmt(timing_mean(run, "old_log_prob")),
                    fmt(timing_mean(run, "ref")),
                    fmt(timing_mean(run, "adv")),
                    fmt(timing_mean(run, "update_actor")),
                    fmt(timing_mean(run, "update_weights")),
                ]
            )
            + " |"
        )
    lines.append("")

    baseline = next((run for run in runs if run["meta"].get("rollout_count") == "1"), None)
    if baseline and timing_mean(baseline, "gen"):
        base_gen = timing_mean(baseline, "gen")
        base_step = timing_mean(baseline, "step")
        lines.append("## 相对 1 卡 rollout 加速")
        lines.append("")
        lines.append("| rollout卡数 | gen speedup | step speedup |")
        lines.append("| ---: | ---: | ---: |")
        for run in runs:
            gen = timing_mean(run, "gen")
            step = timing_mean(run, "step")
            lines.append(
                f"| {run['meta'].get('rollout_count', '?')} | "
                f"{fmt(base_gen / gen if gen else None)}x | "
                f"{fmt(base_step / step if step else None)}x |"
            )
        lines.append("")

    for run in runs:
        meta = run["meta"]
        tb = run["tb"]
        lines.append(f"## rollout={meta.get('rollout_count', '?')} 详细结果")
        lines.append("")
        lines.append(f"- run_name: `{meta.get('run_name', run['run_name'])}`")
        lines.append(f"- CUDA_VISIBLE_DEVICES: `{meta.get('cuda_visible_devices', '?')}`")
        lines.append(f"- rollout_cards: `{meta.get('rollout_cards', '?')}`")
        lines.append(f"- TensorBoard: `{run['tb_dir']}`")
        lines.append(f"- raw log: `{run['log_file']}`")
        lines.append(f"- GPU CSV: `{run['gpu_csv']}`")
        if tb.get("missing"):
            lines.append("- 状态: 缺少 TensorBoard event，无法统计。")
            lines.append("")
            continue
        if not tb.get("train_steps") or not tb.get("timing", {}).get("step"):
            lines.append("- 状态: 未完成或没有训练步 timing，跳过阶段统计。")
            if run["weight_syncs"]:
                vals = run["weight_syncs"]
                lines.append(
                    f"- 已记录 update_weights 日志: mean={mean(vals):.2f}s, "
                    f"min={min(vals):.2f}s, max={max(vals):.2f}s, n={len(vals)}"
                )
            lines.append("")
            continue

        step_total = timing_mean(run, "step")
        lines.append("")
        lines.append("### 阶段均值")
        lines.append("")
        lines.append("| 阶段 | mean_s | min_s | max_s | step占比 | n |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for stage in STAGE_ORDER:
            vals = tb["timing"].get(stage, [])
            if not vals:
                continue
            pct = (mean(vals) / step_total * 100) if step_total else None
            lines.append(
                f"| {stage} | {fmt(mean(vals))} | {fmt(min(vals))} | {fmt(max(vals))} | "
                f"{fmt(pct, 1)}% | {len(vals)} |"
            )
        extra_stages = sorted(set(tb["timing"]) - set(STAGE_ORDER))
        for stage in extra_stages:
            vals = tb["timing"][stage]
            pct = (mean(vals) / step_total * 100) if step_total else None
            lines.append(
                f"| {stage} | {fmt(mean(vals))} | {fmt(min(vals))} | {fmt(max(vals))} | "
                f"{fmt(pct, 1)}% | {len(vals)} |"
            )
        lines.append("")

        lines.append("### 每步耗时")
        lines.append("")
        lines.append("| step_id | step | gen | old_log_prob | ref | adv | update_actor | update_weights |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in tb["per_step"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["step_id"]),
                        fmt(row.get("step")),
                        fmt(row.get("gen")),
                        fmt(row.get("old_log_prob")),
                        fmt(row.get("ref")),
                        fmt(row.get("adv")),
                        fmt(row.get("update_actor")),
                        fmt(row.get("update_weights")),
                    ]
                )
                + " |"
            )
        lines.append("")

        lines.append("### 显存和利用率")
        lines.append("")
        lines.append("| GPU | role | name | peak_mem_gb | total_gb | max_util | mean_util | samples |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for idx in sorted(run["gpu"]):
            item = run["gpu"][idx]
            lines.append(
                f"| {idx} | {role_for_gpu(idx, meta)} | {item['name']} | "
                f"{fmt(item['max_mem_gb'])} | {fmt(item['total_mem_gb'])} | "
                f"{fmt(item['max_util'], 1)}% | {fmt(item['mean_util'], 1)}% | {item['samples']} |"
            )
        lines.append("")

        if tb["perf"]:
            lines.append("### worker 内部显存")
            lines.append("")
            lines.append("| metric | mean_gb | max_gb | n |")
            lines.append("| --- | ---: | ---: | ---: |")
            for tag in sorted(tb["perf"]):
                vals = tb["perf"][tag]
                lines.append(f"| {tag} | {fmt(mean(vals))} | {fmt(max(vals))} | {len(vals)} |")
            lines.append("")

        if run["weight_syncs"]:
            vals = run["weight_syncs"]
            initial = vals[0]
            later = vals[1:]
            lines.append("### 权重同步日志")
            lines.append("")
            lines.append(f"- initial update_weights: {initial:.2f}s")
            if later:
                lines.append(
                    f"- per-step update_weights log: mean={mean(later):.2f}s, "
                    f"min={min(later):.2f}s, max={max(later):.2f}s, n={len(later)}"
                )
            lines.append("")

        if tb["reward"]:
            lines.append("### reward / score")
            lines.append("")
            for tag, vals in tb["reward"].items():
                lines.append(f"- {tag}: " + ", ".join(f"{value:.3f}" for value in vals))
            lines.append("")

    lines.append("## 初步观察")
    lines.append("")
    lines.append("- `gen` 是 rollout 卡数扩展最直接影响的阶段；`ref` 卡配置在本 sweep 内固定，理论上不随 rollout 卡数变化。")
    lines.append("- `update_actor` 固定在 A800 上，随 rollout 卡数变化不应明显下降。")
    lines.append("- `update_weights` 会随 rollout replica 数增加而变化，需要重点看 1/2/3/4 卡的均值和尾部。")
    lines.append("- 本报告只使用本次 sweep 的新日志，不引用旧的、不准确的时间。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python RL/summarize_grpo_sweep.py RUN_GROUP [LOG_DIR]", file=sys.stderr)
        return 2
    run_group = sys.argv[1]
    log_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else LOG_DIR

    runs = []
    for rollout_count in [1, 2, 3, 4]:
        run_name = f"{run_group}_rollout{rollout_count}"
        tb_dir = log_dir / f"tb_{run_name}"
        log_file = log_dir / f"{run_name}.log"
        gpu_csv = log_dir / f"{run_name}_gpu.csv"
        meta_file = log_dir / f"{run_name}_meta.env"
        if not any(path.exists() for path in [tb_dir, log_file, gpu_csv, meta_file]):
            continue
        runs.append(
            {
                "run_name": run_name,
                "tb_dir": str(tb_dir),
                "log_file": str(log_file),
                "gpu_csv": str(gpu_csv),
                "meta": read_meta(meta_file),
                "tb": read_tb(tb_dir),
                "gpu": read_gpu_csv(gpu_csv),
                "weight_syncs": read_weight_syncs(log_file),
            }
        )

    report = build_report(run_group, runs)
    out_path = log_dir / f"{run_group}_report.md"
    out_path.write_text(report)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
