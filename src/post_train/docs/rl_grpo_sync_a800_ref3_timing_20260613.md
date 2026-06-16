# GRPO 同步 split 效率分析 · A800 actor + GPU3 ref + A4000 rollout · 2026-06-13

> 本文记录重新做的同步 GRPO 时间监控结果。之前旧时间不再作为依据。
> 目标是拆清同步条件下 `gen / old_log_prob / ref / update_actor / update_weights`
> 各阶段耗时和显存行为，为后续 rollout 卡数、ref 放置、权重同步方式选择提供依据。
>
> 过程中的问题、原因分析和解决方案单独记录在
> `src/post_train/docs/rl_grpo_sync_issue_log_20260613.md`。

## 1. 实验配置

- 起点模型: `/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf`。
- 数据: `/data/zilu/data_unified_v2/rl_smoke/{train,val}.parquet`。
- 算法: GRPO, train batch 64, rollout `n=8`, 每步 512 条采样。
- 序列长度: `max_prompt_length=1024`, `max_response_length=1024`。
- actor 训练: 全参训练, `actor_rollout_ref.model.lora_rank=0`。
- actor/update: 物理 GPU1, NVIDIA A800-SXM4-80GB。
- ref forward: 物理 GPU3, NVIDIA RTX A4000, `ref_count=1`。
- rollout: 物理 GPU4/5/6/7, NVIDIA RTX A4000, 每张卡 1 个 vLLM replica。
- old logprob: 使用 rollout/sample 侧 logprob bypass, 不额外做 actor old forward。
- 每个配置跑 5 个训练 step。

本页包含三类记录:

- `SERIAL_ROLLOUT_WEIGHT_SYNC=0 + NCCL_REBUILD_GROUP=1 + NCCL_P2P_DISABLE=1`:
  VERL 原生并行 rollout weight sync 路径,已经完成 rollout=1/2/3/4 的 5-step
  正式 sweep；这是当前主要结论。
- `SERIAL_ROLLOUT_WEIGHT_SYNC=1`: 我加的串行 rollout weight sync fallback, 完整跑完
  rollout=1/2/3/4, 只作为故障期间可跑基线和对照。
- 早期 `SERIAL_ROLLOUT_WEIGHT_SYNC=0` 但未禁用 NCCL P2P 的复测: rollout=2
  曾卡在初始多 replica NCCL 权重同步处。该问题已经定位为 A800 + 多张 A4000
  混合 PCIe 拓扑下 NCCL 默认 P2P 路径 hang, 详见问题复盘文档。

正式 P2P-off 并行同步 sweep 的 run group:

```text
grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631
```

完整机器生成报告:

```text
/data/zilu/QseekLLM/src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_report.md
```

串行 fallback 完整 sweep 的 run group:

```text
grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400
```

完整机器生成报告:

```text
/data/zilu/QseekLLM/src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_report.md
```

早期未禁用 P2P 的并行 update 复测 run group:

```text
grpo_sync_a800_refcard3_a4000_parallelsync_formal_20260613_113540
```

关键日志:

```text
/data/zilu/QseekLLM/src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallelsync_formal_20260613_113540_rollout1_timing.txt
/data/zilu/QseekLLM/src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallelsync_formal_20260613_113540_rollout2.log
```

## 2. 关键结论

1. 当前正式表采用 VERL 原生并行 rollout weight sync, 固定环境为:

```bash
SERIAL_ROLLOUT_WEIGHT_SYNC=0
NCCL_REBUILD_GROUP=1
NCCL_P2P_DISABLE=1
```

   rollout=1/2/3/4 都已完成 5 个训练 step。

2. 正式并行同步下, rollout 从 1 张 A4000 增到 4 张 A4000, step 从
   137.54s 降到 92.65s, 端到端加速 1.48x。

3. rollout 扩卡主要降低 `gen`: 74.55s -> 44.65s -> 30.55s -> 25.14s,
   从 1 卡到 4 卡采样阶段加速 2.97x。

4. `ref` 基本固定在 26.7-27.5s, 因为 ref 只在物理 GPU3 上 forward,
   不随 rollout 卡数变化。

5. `update_actor` 基本固定在 29.9-30.9s, 因为 actor 全参训练始终在物理 GPU1 A800 上。

6. `update_weights` 在并行同步路径下从 3.41s 增到 6.55s, 仍会随 replica 数增加,
   但不是串行 fallback 的近线性增长。串行 fallback 的 4 卡 `update_weights` 为 14.89s,
   不能作为 VERL 并行同步性能结论。

7. rollout=4 是本轮最快配置: 92.65s/step。rollout=3 为 96.67s/step,
   只比 rollout=4 慢约 4.2%, 但少占一张 A4000, 是当前更好的成本/吞吐折中点。

8. rollout=4 正式结果仍使用 `gpu_memory_utilization=0.5`。第一次 4 卡 0.5 尝试在
   权重同步合并阶段出现过一次近阈值 OOM: 需要额外申请 1.14GiB, 当时单张 rollout
   A4000 只剩 1.07GiB。清理后用同样 0.5 重跑已完整通过, 因此这不是“0.5 必然爆显存”,
   而是并行同步瞬时 buffer 与 vLLM 常驻显存叠加后的余量问题。

## 3. 阶段定义

| 阶段 | 含义 |
| --- | --- |
| `gen` | rollout/vLLM 采样生成；本实验中也包含 rollout 侧 sampled-token logprob 返回。 |
| `old_log_prob` | bypass 模式下复用 rollout logprob 写入 old logprob, 预期接近 0。 |
| `ref` | 物理 GPU3 A4000 上 ref model forward, 计算 KL 所需 ref logprob。 |
| `adv` | reward/advantage 计算与字段整理。 |
| `update_actor` | 物理 GPU1 A800 上 actor forward/backward/optimizer step。 |
| `update_weights` | actor 权重同步到 rollout vLLM replica, 包括 pause/resume、NCCL group、load weights 等。 |
| `step` | 同步训练循环端到端墙钟。 |

## 4. VERL 并行 update + P2P-off 正式结果

正式命令:

```bash
SERIAL_ROLLOUT_WEIGHT_SYNC=0 \
NCCL_REBUILD_GROUP=1 \
NCCL_P2P_DISABLE=1 \
RUN_GROUP=grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631 \
bash src/post_train/RL/run_grpo_sync_a800_ref3_a4000_sweep.sh 5
```

环境确认:

```text
serial_rollout_weight_sync=0
nccl_rebuild_group=1
nccl_p2p_disable=1
VERL_SERIAL_ROLLOUT_WEIGHT_SYNC=0
VERL_NCCL_REBUILD_GROUP=1
```

### 4.1 均值总览

| rollout 卡数 | rollout 物理卡 | step | gen | old_log_prob | ref | adv | update_actor | update_weights |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 137.54s | 74.55s | 0.31s | 26.73s | 0.64s | 29.90s | 3.41s |
| 2 | 4,5 | 109.68s | 44.65s | 0.32s | 27.39s | 0.55s | 30.54s | 4.45s |
| 3 | 4,5,6 | 96.67s | 30.55s | 0.32s | 27.29s | 0.67s | 30.60s | 5.43s |
| 4 | 4,5,6,7 | 92.65s | 25.14s | 0.33s | 27.48s | 0.69s | 30.85s | 6.55s |

### 4.2 相对 1 张 rollout 的加速

| rollout 卡数 | gen speedup | step speedup |
| ---: | ---: | ---: |
| 1 | 1.00x | 1.00x |
| 2 | 1.67x | 1.25x |
| 3 | 2.44x | 1.42x |
| 4 | 2.97x | 1.48x |

### 4.3 nvidia-smi 外部采样峰值

| rollout 卡数 | A800 GPU1 actor | A4000 GPU3 ref | A4000 rollout 峰值 |
| ---: | ---: | ---: | --- |
| 1 | 75.44GB / 80.00GB | 9.57GB / 15.99GB | GPU4 12.26GB |
| 2 | 75.62GB / 80.00GB | 13.72GB / 15.99GB | GPU4 12.96GB, GPU5 12.67GB |
| 3 | 71.32GB / 80.00GB | 14.30GB / 15.99GB | GPU4 13.96GB, GPU5 14.10GB, GPU6 13.79GB |
| 4 | 70.74GB / 80.00GB | 14.07GB / 15.99GB | GPU4 14.46GB, GPU5 14.58GB, GPU6 14.31GB, GPU7 14.49GB |

### 4.4 actor worker 内部显存

| rollout 卡数 | cpu_memory_used mean/max | max_memory_allocated mean/max | max_memory_reserved mean/max |
| ---: | ---: | ---: | ---: |
| 1 | 67.51 / 67.64GB | 45.74 / 45.74GB | 74.03 / 74.32GB |
| 2 | 71.07 / 71.18GB | 45.91 / 45.93GB | 74.06 / 74.44GB |
| 3 | 74.67 / 74.86GB | 45.74 / 45.93GB | 76.52 / 76.52GB |
| 4 | 97.81 / 97.95GB | 45.67 / 45.87GB | 74.98 / 77.05GB |

### 4.5 rollout=4 0.5 显存说明

正式结果保持 `actor_rollout_ref.rollout.gpu_memory_utilization=0.5`。
rollout=4 第一次 0.5 尝试在训练第 1 步后的并行 `update_weights` 合并阶段出现过一次 OOM:

```text
Tried to allocate 1.14 GiB. GPU 0 ... 1.07 GiB is free.
Process ... has 12.89 GiB memory in use.
```

该 OOM 发生在 vLLM 常驻显存之外, checkpoint worker 临时接收/合并权重时还需要额外 buffer。
它和 `gpu_memory_utilization=0.5` 不矛盾:这个参数主要约束 vLLM KV cache 规划,不是整张卡的
总显存硬上限。清理后用同样 0.5 重跑 rollout=4 已完整通过 5 step, 因此正式表仍记录为 0.5。

### 4.6 历史失败点

早期未设置 `NCCL_P2P_DISABLE=1` 时,rollout=2 曾卡在初始并行 `update_weights`,
日志停在 `build_process_group done` 后没有 `send weights done` / `receive weights done`。
最小 Ray NCCL broadcast 证明默认 P2P 在同一组 GPU 上也会卡住；加入
`NCCL_P2P_DISABLE=1` 后 1MB/512MB broadcast 和完整 GRPO 都通过。

## 5. 串行 fallback 时间结果

### 5.1 均值总览

| rollout 卡数 | rollout 物理卡 | step | gen | old_log_prob | ref | adv | update_actor | update_weights |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 143.26s | 79.18s | 0.44s | 27.31s | 0.70s | 30.19s | 3.42s |
| 2 | 4,5 | 110.95s | 43.21s | 0.32s | 27.77s | 0.57s | 30.54s | 7.40s |
| 3 | 4,5,6 | 102.19s | 31.62s | 0.33s | 27.79s | 0.68s | 30.74s | 10.55s |
| 4 | 4,5,6,7 | 99.96s | 25.13s | 0.34s | 27.76s | 0.58s | 30.78s | 14.89s |

### 5.2 相对 1 张 rollout 的加速

| rollout 卡数 | gen speedup | step speedup |
| ---: | ---: | ---: |
| 1 | 1.00x | 1.00x |
| 2 | 1.83x | 1.29x |
| 3 | 2.50x | 1.40x |
| 4 | 3.15x | 1.43x |

### 5.3 每步明细

#### rollout=1

| step_id | step | gen | old_log_prob | ref | adv | update_actor | update_weights |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 155.59 | 81.39 | 0.95 | 28.88 | 0.65 | 31.82 | 3.43 |
| 2 | 128.80 | 70.33 | 0.32 | 25.39 | 0.61 | 28.37 | 3.40 |
| 3 | 144.47 | 81.39 | 0.32 | 27.48 | 1.12 | 30.30 | 3.45 |
| 4 | 139.58 | 78.40 | 0.32 | 26.72 | 0.55 | 29.78 | 3.42 |
| 5 | 147.86 | 84.41 | 0.31 | 28.09 | 0.57 | 30.71 | 3.38 |

#### rollout=2

| step_id | step | gen | old_log_prob | ref | adv | update_actor | update_weights |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 124.26 | 49.24 | 0.35 | 29.38 | 0.63 | 32.85 | 7.73 |
| 2 | 107.33 | 41.21 | 0.33 | 27.66 | 0.58 | 29.98 | 7.18 |
| 3 | 111.25 | 43.21 | 0.31 | 28.16 | 0.58 | 31.40 | 7.22 |
| 4 | 109.93 | 44.22 | 0.32 | 27.41 | 0.51 | 29.91 | 7.17 |
| 5 | 101.96 | 38.18 | 0.33 | 26.25 | 0.55 | 28.56 | 7.70 |

#### rollout=3

| step_id | step | gen | old_log_prob | ref | adv | update_actor | update_weights |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 113.15 | 36.45 | 0.34 | 30.44 | 0.58 | 33.25 | 11.58 |
| 2 | 98.20 | 30.16 | 0.33 | 27.20 | 0.58 | 29.56 | 9.99 |
| 3 | 101.82 | 31.15 | 0.32 | 28.02 | 1.08 | 30.11 | 10.59 |
| 4 | 96.64 | 29.15 | 0.32 | 26.19 | 0.58 | 30.04 | 9.97 |
| 5 | 101.11 | 31.16 | 0.32 | 27.07 | 0.57 | 30.75 | 10.61 |

#### rollout=4

| step_id | step | gen | old_log_prob | ref | adv | update_actor | update_weights |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 104.51 | 28.15 | 0.37 | 28.18 | 0.64 | 30.89 | 15.56 |
| 2 | 97.66 | 24.13 | 0.32 | 27.21 | 0.53 | 30.11 | 14.96 |
| 3 | 103.07 | 25.13 | 0.33 | 29.10 | 0.62 | 32.53 | 14.93 |
| 4 | 99.17 | 25.13 | 0.33 | 27.45 | 0.54 | 30.14 | 15.11 |
| 5 | 95.42 | 23.13 | 0.34 | 26.86 | 0.55 | 30.23 | 13.91 |

## 6. 串行 fallback 显存结果

### 6.1 nvidia-smi 外部采样峰值

| rollout 卡数 | A800 GPU1 actor | A4000 GPU3 ref | A4000 rollout 峰值 |
| ---: | ---: | ---: | --- |
| 1 | 66.67GB / 80.00GB | 14.10GB / 15.99GB | GPU4 13.34GB |
| 2 | 71.81GB / 80.00GB | 13.77GB / 15.99GB | GPU4 13.07GB, GPU5 12.75GB |
| 3 | 71.44GB / 80.00GB | 14.01GB / 15.99GB | GPU4 14.03GB, GPU5 13.79GB, GPU6 13.78GB |
| 4 | 68.87GB / 80.00GB | 13.79GB / 15.99GB | GPU4 14.63GB, GPU5 14.80GB, GPU6 14.00GB, GPU7 14.02GB |

### 6.2 actor worker 内部显存

| rollout 卡数 | cpu_memory_used mean/max | max_memory_allocated mean/max | max_memory_reserved mean/max |
| ---: | ---: | ---: | ---: |
| 1 | 82.36 / 82.51GB | 45.43 / 45.85GB | 56.12 / 64.53GB |
| 2 | 85.81 / 85.95GB | 45.88 / 45.88GB | 76.67 / 76.67GB |
| 3 | 89.12 / 89.78GB | 44.79 / 45.06GB | 67.03 / 75.55GB |
| 4 | 91.65 / 91.85GB | 45.32 / 45.76GB | 71.43 / 77.11GB |

## 7. 串行 fallback 中 update_weights 增长原因

这次正式 sweep 使用的是 serial rollout weight sync fallback:

```bash
SERIAL_ROLLOUT_WEIGHT_SYNC=1
VERL_SERIAL_ROLLOUT_WEIGHT_SYNC=1
NCCL_REBUILD_GROUP=1
```

因此 `update_weights` 不是 VERL 原本并行同步路径。当前路径会对每个 rollout replica
逐个执行:

```text
build_process_group -> trainer.update_weights + rollout.update_weights -> finalize
```

所以 `update_weights` 会近似随 rollout replica 数线性增加。日志里的典型单 replica 行为:

```text
build_process_group: 1.3-1.5s
transfer/load weights: 1.8-2.0s
finalize: 0.1-0.6s
replica_total: 3.3-3.9s
```

这解释了均值:

| rollout 卡数 | update_weights 均值 |
| ---: | ---: |
| 1 | 3.42s |
| 2 | 7.40s |
| 3 | 10.55s |
| 4 | 14.89s |

注意: VERL 原本并行路径仍在 `CheckpointEngineManager.update_weights()` 的 `else` 分支。
如果并行路径稳定, `update_weights` 应接近一次建组/finalize + 最慢 rollout replica 的 load 时间,
不应按 rollout 卡数线性增长。

## 8. ref model 和 old logprob 说明

### 8.1 ref model 放置和 forward 路径

- ref checkpoint 与 actor 起点一致:

```text
/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf
```

- 本实验 ref 只放在物理 GPU3 A4000, 不是 3 张 ref 卡。
- ref forward 路径:

```text
PPOTrainer._compute_ref_log_prob()
  -> self.ref_policy_wg.compute_ref_log_prob(batch)
  -> ActorRolloutRefWorker.compute_ref_log_prob()
  -> self.ref.infer_batch(data=data)
```

### 8.2 old logprob 来源

当前配置:

```text
actor_rollout_ref.rollout.calculate_log_probs=True
algorithm.rollout_correction.bypass_mode=True
```

因此 `old_log_prob` 是采样侧 behavior policy 的概率, 从 rollout/vLLM 侧随样本返回,
再在训练循环里 bypass 写入 old logprob 字段。A800 上 `update_actor` 里的 forward/backward
计算的是当前 actor/new logprob, 用于 PPO ratio:

```text
ratio = exp(new_log_prob - old_log_prob)
```

所以这次 `old_log_prob` 阶段只有 0.3-0.4s, 它不是一次额外 actor old forward。

## 9. 参数选择建议

### 9.1 当前建议

- 首选运行环境固定为:

```bash
SERIAL_ROLLOUT_WEIGHT_SYNC=0 \
NCCL_REBUILD_GROUP=1 \
NCCL_P2P_DISABLE=1 \
bash src/post_train/RL/run_grpo_sync_a800_ref3_a4000_sweep.sh 5
```

- `SERIAL_ROLLOUT_WEIGHT_SYNC=0`: 关闭我加的串行 fallback, 回到 VERL 原本并行 update 分支。
- `NCCL_REBUILD_GROUP=1`: 保留 NCCL group 每轮 rebuild/destroy, 避免 group 复用导致 remote error。
- `NCCL_P2P_DISABLE=1`: 绕开本机 A800+A4000 混合 PCIe 拓扑下默认 NCCL P2P hang。

- 当前吞吐/资源折中建议优先看 rollout=3:
  - 96.67s/step, 比 1 卡快 1.42x。
  - 相比 rollout=4 只慢 4.02s/step, 约 4.2%, 但少占一张 A4000。
  - A4000 rollout 峰值约 13.79-14.10GB, 仍有一点余量。

- rollout=4 是本轮最快:
  - 92.65s/step, 比 1 卡快 1.48x。
  - `gen` 已降到 25.14s, 但 ref/update_actor 固定开销占比升高。
  - A4000 rollout 峰值约 14.31-14.58GB, 16GB 卡上余量较小；建议正式长跑前确认机器上无残留进程。

- 不建议用串行 fallback 的 4 卡表做参数选择。串行 fallback 的 4 卡 `update_weights=14.89s`,
  而正式并行 P2P-off 的 4 卡 `update_weights=6.55s`, 两者代表不同同步路径。

### 9.2 后续优化方向

- 若目标是进一步降低 step, ref 单卡 27s 和 actor update 31s 已经成为主要固定项,
  后续要考虑 ref 并行/异步化或 actor 训练侧优化。
- 若目标是减少同步开销,优先比较 LoRA-only 同步与全参同步；本轮 `lora_rank=0`
  是全参权重广播。
- 若换机器或换 GPU 拓扑,先跑 `RL/nccl_broadcast_smoke.py` 验证 1MB 和 512MB broadcast,
  再跑完整 GRPO。
- 若需要重新做 NCCL debug, 再额外打开:

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,COLL,P2P,NET
```

## 10. 关键文件

- 单实验脚本: `src/post_train/RL/run_grpo_sync_a800_ref3_a4000_one.sh`
- sweep 脚本: `src/post_train/RL/run_grpo_sync_a800_ref3_a4000_sweep.sh`
- split placement 入口: `src/post_train/RL/main_grpo_sync_split.py`
- 时间解析: `src/post_train/RL/parse_timing.py`
- sweep 汇总: `src/post_train/RL/summarize_grpo_sweep.py`
- checkpoint manager: `src/post_train/verl/verl/checkpoint_engine/base.py`
- NCCL checkpoint engine: `src/post_train/verl/verl/checkpoint_engine/nccl_checkpoint_engine.py`

## 11. 原始日志

### 11.1 正式 P2P-off 并行 update sweep

| rollout 卡数 | raw log | timing summary | gpu csv |
| ---: | --- | --- | --- |
| 1 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout1.log` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout1_timing.txt` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout1_gpu.csv` |
| 2 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout2.log` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout2_timing.txt` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout2_gpu.csv` |
| 3 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout3.log` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout3_timing.txt` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout3_gpu.csv` |
| 4 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout4.log` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout4_timing.txt` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout4_gpu.csv` |

补充:

```text
src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_report.md
src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout4_oom_gmem05.log
src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout4_oom_gmem05_meta.env
```

### 11.2 早期未禁用 P2P 的并行 update 复测

| rollout 卡数 | raw log | timing summary | gpu csv |
| ---: | --- | --- | --- |
| 1 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallelsync_formal_20260613_113540_rollout1.log` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallelsync_formal_20260613_113540_rollout1_timing.txt` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallelsync_formal_20260613_113540_rollout1_gpu.csv` |
| 2 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallelsync_formal_20260613_113540_rollout2.log` | 未生成, 卡在初始 `update_weights` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallelsync_formal_20260613_113540_rollout2_gpu.csv` |

### 11.3 串行 fallback 完整 sweep

| rollout 卡数 | raw log | timing summary | gpu csv |
| ---: | --- | --- | --- |
| 1 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout1.log` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout1_timing.txt` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout1_gpu.csv` |
| 2 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout2.log` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout2_timing.txt` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout2_gpu.csv` |
| 3 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout3.log` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout3_timing.txt` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout3_gpu.csv` |
| 4 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout4.log` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout4_timing.txt` | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_serialsync_formal_20260613_102400_rollout4_gpu.csv` |
