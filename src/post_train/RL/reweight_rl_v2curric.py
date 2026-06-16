#!/usr/bin/env python3
"""构建 RL v2 课程数据池(诊断结论详见 docs/rl_degradation_diagnosis_20260613.md)。

背景:v1 全量混合(train_rl_s4clean_fix)里 ~40% 是竞赛集(numina/openr1/big-math/
deepscaler/dapo),基线 solve_rate ~0-4%,采 16 个基本全错 → 对 GRPO 零有效梯度,
还经 std 归一把 shaping 噪声放大成"啰嗦/套格式"伪梯度,污染中等带。

v2 策略(用户拍板 60:30:10 精神):
  - 主干 = 去掉竞赛集后的全部(compute_cot 技能题 + calc/orca/metamath/gsm8k/
    chinese/hendrycks/infinity 应用题),约 90 万条,广度(170+ 题型)全保留;
  - 难题彩票 = 从 numina/openr1(两个最大、p~0.04、G=16 至少中1个概率 ~48% 的集)
    随机抽 HARD_KEEP 条,作为"偶尔能撞对"的探索燃料;
  - 剔除 dapo(基线 0%,真死签)/big-math(~2%)/deepscaler(小且低),v1 不掺。

安全阀:本轮配 norm_adv_by_std=False + 砍 shaping,使全错组 ≈ 零方差零梯度,
即便没接 filter_groups,掺进来的难题也只在"中奖组"贡献梯度,不污染。filter_groups
(在线按 correct 方差过滤 + 移动前沿)留作 v2 的效率/课程升级。
"""
from __future__ import annotations
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np

SRC = "/data/zilu/data_unified_v2/parquet/train_rl_s4clean_fix.parquet"
OUT = "/data/zilu/data_unified_v2/parquet/train_rl_s4_v2curric.parquet"

# 竞赛集:整体剔出主干
COMPETITION = {
    "numinamath-1.5", "openr1-math-220k", "big-math-rl-verified",
    "deepscaler-preview", "dapo-math-17k-dedup",
}
# 难题彩票只从这两个抽(p~0.04,G=16 中奖率最高)
LOTTERY_SRC = {"numinamath-1.5", "openr1-math-220k"}
HARD_KEEP = 100_000
SEED = 20260613


def main():
    t = pq.read_table(SRC)
    ds = np.array(t.column("data_source").to_pylist())
    n = len(ds)

    backbone_mask = ~np.isin(ds, list(COMPETITION))
    lottery_pool = np.where(np.isin(ds, list(LOTTERY_SRC)))[0]

    rng = np.random.default_rng(SEED)
    keep_hard = rng.choice(lottery_pool, size=min(HARD_KEEP, len(lottery_pool)), replace=False)

    keep = np.zeros(n, dtype=bool)
    keep[backbone_mask] = True
    keep[keep_hard] = True
    idx = np.where(keep)[0]
    rng.shuffle(idx)

    out = t.take(pa.array(idx))
    pq.write_table(out, OUT)

    # 报告
    kept_ds = ds[idx]
    from collections import Counter
    c = Counter(kept_ds)
    cc = sum(v for k, v in c.items() if k.startswith("compute_cot"))
    print(f"源 {n} -> 输出 {len(idx)} 条  ({OUT})")
    print(f"  compute_cot 技能题: {cc} ({100*cc/len(idx):.1f}%)")
    hard_total = sum(c[s] for s in COMPETITION)
    print(f"  难题彩票(numina/openr1): {hard_total} ({100*hard_total/len(idx):.1f}%)")
    print("  真实应用集:")
    for s, v in sorted(((k, v) for k, v in c.items() if not k.startswith("compute_cot")),
                       key=lambda x: -x[1]):
        print(f"    {s:26s} {v:8d} ({100*v/len(idx):.1f}%)")


if __name__ == "__main__":
    main()
