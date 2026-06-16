# GRPO 同步效率分析问题复盘 · 2026-06-13

> 目的:记录本轮同步 GRPO 初步效率分析过程中遇到的问题、判断依据、解决方案和当前验证状态,
> 方便后续汇报时解释为什么重跑、为什么权重同步表需要区分串行/并行,以及后续参数选择的依据。

## 1. 当前实验目标

本轮目标不是追求最终最优吞吐,而是先在同步 GRPO 条件下拆清每个阶段的时间开销:

| 阶段 | 关注点 |
| --- | --- |
| `gen` | rollout/vLLM 采样耗时,观察 rollout 卡数扩展收益 |
| `old_log_prob` | old policy logprob 获取/写入耗时 |
| `ref` | ref model forward 耗时 |
| `update_actor` | actor 全参训练 forward/backward/optimizer step |
| `update_weights` | actor 权重同步到 rollout vLLM replicas |
| 显存 | actor/ref/rollout 各物理卡峰值占用 |

固定实验放置:

| 角色 | 物理 GPU | 说明 |
| --- | --- | --- |
| actor/update | GPU1 A800 80GB | 全参训练,`lora_rank=0` |
| ref forward | GPU3 A4000 16GB | 只有 1 张 ref 卡,不是 3 张 ref 卡 |
| rollout/vLLM | GPU4/5/6/7 A4000 16GB | rollout=1/2/3/4 sweep |

## 2. 问题一:旧时间表不能作为依据

### 现象

早期已有一份同步 GRPO 时间表,但用户明确指出“之前的时间不需要参考,都不准确”,需要重新做时间监控。

### 分析

旧表的问题主要有两类:

1. 实验语义没有完全固定:ref 放置、old logprob 来源、权重同步方式需要重新确认。
2. 后续发现完整 1/2/3/4 表实际跑在 `SERIAL_ROLLOUT_WEIGHT_SYNC=1` 串行 fallback 上,
   因此 `update_weights` 随 rollout 卡数近似线性增加,不能代表 VERL 标准并行同步路径。

### 解决方案

重新固定实验配置和日志字段,每个 rollout 卡数跑 5 个 step,记录:

```text
rollout卡数, rollout物理卡, step总时长, gen采样, old_log_prob,
ref forward, adv, actor更新, 权重同步, GPU显存峰值
```

并将串行 fallback 表明确标注为“可跑基线/对照”,不作为并行同步结论。

## 3. 问题二:ref model 和 old_log_prob 的语义需要澄清

### ref model 放置与 forward

本轮用户要求的是“3 号卡放 ref”,不是“3 张卡 ref”。最终实验语义为:

- ref model 常驻物理 GPU3 A4000。
- `ref` 阶段是在 GPU3 上做 ref policy forward,计算 KL 所需 ref logprob。
- rollout 卡数变化只影响 GPU4/5/6/7 的 vLLM replicas,不改变 ref forward 并行度。

因此时间表中 `ref` 维持在约 27-28s 是合理的:它没有随 rollout 扩卡。

### old_log_prob 来源

`old_log_prob` 是 GRPO/PPO 类算法中 old policy 对已采样 token 的 logprob,用于 policy ratio。
本轮配置下:

- rollout/vLLM 侧开启 `calculate_log_probs=True`。
- 采样时返回 sample 侧 logprob。
- 训练侧 old logprob 走 bypass/复用路径,不再额外做一次 actor old forward。

因此 `old_log_prob` 阶段耗时只有约 0.3-0.4s,主要是字段整理/搬运,不是一次完整 forward。

## 4. 问题三:update_weights 随 rollout 卡数增加

### 现象

串行 fallback 表里 `update_weights` 从 1 卡到 4 卡明显增加:

| rollout 卡数 | update_weights |
| ---: | ---: |
| 1 | 3.42s |
| 2 | 7.40s |
| 3 | 10.55s |
| 4 | 14.89s |

### 分析

这不是 VERL 标准并行同步的预期行为,而是因为当时脚本默认启用了:

```bash
SERIAL_ROLLOUT_WEIGHT_SYNC=1
```

串行 fallback 会逐个 rollout replica 做一次权重同步,所以 rollout 卡数越多,同步总耗时越接近线性增长。
这条路径是为了先跑通正式 sweep 和定位阶段瓶颈临时保留的兜底路径。

### 解决方案

切回 VERL 原生并行路径:

```bash
SERIAL_ROLLOUT_WEIGHT_SYNC=0
```

VERL 原生路径在 `CheckpointEngineManager.update_weights()` 中已经实现:

- 一次性为所有 rollout replicas 建 temporary worker group。
- 并发 dispatch `trainer.update_weights(...)` 和 `rollout.update_weights(...)`。
- 用同一个 NCCL group 完成 trainer root 到多 rollout replicas 的 broadcast。

也就是说,并行同步本身不是本轮重新实现的；本轮主要是在脚本默认值、环境变量、日志和死锁排查上做工作。

## 5. 问题四:VERL 标准并行同步在 rollout=2 卡住

### 现象

切回并行路径后,rollout=1 可以完成 5 step；rollout=2 在初始 `update_weights` 卡住。
关键日志停在:

```text
[checkpoint-manager] build_process_group done
```

之后没有:

```text
send weights done
receive weights done
transfer/load weights done
Training Progress
```

外部观察到 GPU1/4/5 util 100%,但没有训练进度推进。

### 分析过程

排查时依次验证了几个可能层级:

| 怀疑点 | 结论 | 依据 |
| --- | --- | --- |
| trainer/rollout 两端是否并发到场 | 排除 | 两端均进入 `update_weights` |
| NCCL group 是否建成 | 排除 | `build_process_group done` 已出现 |
| ZMQ 元数据握手是否失败 | 排除 | rollout rank 收到 metadata |
| state_dict 生成是否慢 | 排除 | 卡点发生在 broadcast,不是 state_dict |
| CuPy/torch buffer 类型问题 | 排除 | 切 torch master buffer 后仍卡 |
| NCCL group 复用问题 | 部分规避 | `NCCL_REBUILD_GROUP=1` 保留,但不是根因 |
| NCCL 跨卡传输层 | 命中 | 三个 rank 均进入第一个 broadcast,无一完成 |

进一步用最小 Ray NCCL broadcast 脱离完整 GRPO 复现:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,4,5 \
PYTHONPATH=/data/zilu/QseekLLM/src/post_train/verl \
/data/zilu/QseekLLM/src/post_train/.venv/bin/python \
  RL/nccl_broadcast_smoke.py --world-size 3 --mb 1
```

默认 NCCL P2P 下,1MB broadcast 也会卡住。这证明根因不在 GRPO 训练循环,
而在本机 A800 + 多 A4000 混合 PCIe 拓扑的 NCCL P2P 传输路径。

### 根因判断

本机拓扑特征:

- GPU 间无 NVLink。
- GPU1 A800 与 GPU4/5 A4000 是跨代 GPU。
- A800 与 rollout A4000 跨 PCIe bridge。

在该拓扑下,NCCL 默认 P2P 路径可能不报错而直接 hang。完整 GRPO 中看到的
“100% GPU util 但无进度”符合 NCCL collective busy-poll 卡住的表现。

## 6. 问题四的解决方案:NCCL_P2P_DISABLE=1

最终有效环境:

```bash
SERIAL_ROLLOUT_WEIGHT_SYNC=0
NCCL_REBUILD_GROUP=1
NCCL_P2P_DISABLE=1
```

含义:

| 环境变量 | 作用 |
| --- | --- |
| `SERIAL_ROLLOUT_WEIGHT_SYNC=0` | 使用 VERL 原生并行 rollout 权重同步 |
| `NCCL_REBUILD_GROUP=1` | 每次 finalize 后销毁 NCCL group,避免 group 复用问题 |
| `NCCL_P2P_DISABLE=1` | 禁用跨 GPU P2P,让 NCCL 走 SHM/host 中转,绕开混合 PCIe P2P hang |

验证结果:

| 验证 | 结果 |
| --- | --- |
| Ray NCCL broadcast,world=3,1MB,默认 P2P | 卡住 |
| Ray NCCL broadcast,world=3,1MB,`NCCL_P2P_DISABLE=1` | 通过 |
| Ray NCCL broadcast,world=3,512MB,`NCCL_P2P_DISABLE=1` | 通过,约 0.115-0.118s |
| 完整 GRPO rollout=2,1 step,并行同步 + P2P off | 通过,`update_weights=4.46s` |
| 完整 GRPO rollout=1/2/3/4,5 step,并行同步 + P2P off | 通过,正式表已生成 |

完整 GRPO smoke 命令:

```bash
SERIAL_ROLLOUT_WEIGHT_SYNC=0 \
NCCL_REBUILD_GROUP=1 \
NCCL_P2P_DISABLE=1 \
RUN_GROUP=grpo_sync_a800_refcard3_a4000_parallel_update_p2poff_smoke_20260613_1321 \
bash src/post_train/RL/run_grpo_sync_a800_ref3_a4000_one.sh 2 1
```

关键日志:

```text
Rank 0 send weights done, time cost: 2.10s
Rank 1 receive weights done, total_params: 259, time cost: 2.15s, bandwidth: 3.96 GB/s
Rank 2 receive weights done, total_params: 259, time cost: 2.15s, bandwidth: 3.96 GB/s
```

## 7. 问题五:rollout=4 在 0.5 下的一次瞬时 OOM

### 现象

正式 P2P-off 并行 sweep 中,rollout=4 第一次仍按:

```text
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
rollout_cards=4,5,6,7
SERIAL_ROLLOUT_WEIGHT_SYNC=0
NCCL_REBUILD_GROUP=1
NCCL_P2P_DISABLE=1
```

启动后可以完成初始权重同步和第 1 个训练 step,但在第 1 步后的并行 `update_weights`
阶段报过一次 OOM。关键错误:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.14 GiB.
GPU 0 has a total capacity of 15.72 GiB of which 1.07 GiB is free.
Including non-PyTorch memory, this process has 1.75 GiB memory in use.
Process ... has 12.89 GiB memory in use.
```

### 分析

这次 OOM 不能理解为“`gpu_memory_utilization=0.5` 必然爆”。原因是:

1. `gpu_memory_utilization=0.5` 主要是 vLLM 用来规划 KV cache 的预算,不是整张 GPU
   的总显存硬限制。
2. rollout GPU 上除了 vLLM server,并行权重同步时还会同时启动 checkpoint worker,
   接收 broadcast 权重并在 `merge_weight_chunks`/load 阶段临时申请额外 buffer。
3. 串行 fallback 之前更不容易触发这个瞬时峰值,因为它逐个 rollout replica 同步；
   VERL 并行路径会让多个 rollout replica 同时进入同步,时间上更集中。
4. 失败点的余量非常窄:还差约 70MiB 左右就能满足那次 1.14GiB 申请,属于近阈值瞬时 OOM。

### 解决和当前状态

我没有把正式配置改成 0.4。清理残留进程后,用同样的 `gpu_memory_utilization=0.5`
重跑 rollout=4,已经完整完成 5 step。正式结果为:

| rollout 卡数 | step | gen | ref | update_actor | update_weights | rollout 显存峰值 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 92.65s | 25.14s | 27.48s | 30.85s | 6.55s | GPU4 14.46GB, GPU5 14.58GB, GPU6 14.31GB, GPU7 14.49GB |

因此报告里应表述为:rollout=4、0.5 在 16GB A4000 上可跑,但显存余量较小；
并行 update 的临时 buffer 可能造成近阈值瞬时 OOM。长跑前要确认没有残留进程,
必要时再考虑降低 `gpu_memory_utilization` 或调小同步 bucket,但本轮正式数据保持 0.5。

## 8. 本轮做过的代码/脚本支持

| 文件 | 作用 |
| --- | --- |
| `src/post_train/RL/run_grpo_sync_a800_ref3_a4000_one.sh` | 默认切到并行同步,保留 `NCCL_REBUILD_GROUP=1`,新增并导出 `NCCL_P2P_DISABLE=1`,meta 记录该环境；rollout 显存/util/max_num_batched_tokens/max_num_seqs 支持环境变量覆盖,默认仍是 0.5/8192/128 |
| `src/post_train/RL/nccl_broadcast_smoke.py` | 最小 Ray NCCL broadcast 复现/验证脚本 |
| `src/post_train/verl/verl/checkpoint_engine/nccl_checkpoint_engine.py` | 加入环境开关和 trace,用于定位 broadcast 卡点 |
| `src/post_train/verl/verl/workers/engine_workers.py` | 加入 checkpoint-manager 阶段日志 |
| `src/post_train/verl/verl/workers/engine/fsdp/transformer_impl.py` | 加入 state_dict/get_per_tensor_param trace,排除 state_dict 慢的问题 |

这些改动的核心作用是“观测和运行开关”,不是重写 VERL 原有并行同步算法。

## 9. 当前可汇报结论

1. 同步 GRPO 的主要可扩展阶段是 `gen`: rollout 从 1 卡到多卡后采样时间明显下降。
2. `ref` 不随 rollout 扩展,因为 ref 固定在 GPU3 单卡 forward。
3. `update_actor` 不随 rollout 扩展,因为 actor 全参训练固定在 GPU1 A800。
4. 早期 `update_weights` 线性增加是串行 fallback 的结果,不能作为 VERL 并行同步性能结论。
5. VERL 原本就有并行同步；本轮耗时长的原因是并行路径在本机混合 GPU PCIe 拓扑下触发 NCCL P2P hang。
6. `NCCL_P2P_DISABLE=1` 已通过最小 broadcast、完整 GRPO rollout=2 smoke 和 rollout=1/2/3/4 正式 sweep 验证,是当前机器上的必要运行前提。
7. 正式 1/2/3/4 rollout 五步均值表已经基于 `SERIAL_ROLLOUT_WEIGHT_SYNC=0 + NCCL_REBUILD_GROUP=1 + NCCL_P2P_DISABLE=1` 生成。
8. 正式结果中 rollout=4 保持 `gpu_memory_utilization=0.5`;一次 OOM 是近阈值瞬时 buffer 问题,同配置清理后重跑已通过。

## 10. 后续建议

1. 正式报告只采用 P2P-off 并行同步 sweep 的结果作为“并行同步”表。
2. 串行 fallback 表保留为故障期间可跑基线,但标题必须明确标注“串行同步”。
3. 如果后续换机器或换 GPU 拓扑,先跑 `nccl_broadcast_smoke.py` 验证 1MB 和 512MB broadcast,再跑完整 GRPO。
4. 当前参数选择上,rollout=3 是更好的成本/吞吐折中点；rollout=4 最快但 A4000 显存余量较小。
5. 若继续优化吞吐,优先比较 LoRA-only 同步与全参同步,以及 ref 单卡瓶颈是否需要并行化或异步化。

## 11. 相关文档和日志

| 类型 | 路径 |
| --- | --- |
| 同步 GRPO timing 文档 | `src/post_train/docs/rl_grpo_sync_a800_ref3_timing_20260613.md` |
| NCCL 死锁定位文档 | `src/post_train/docs/rl_nccl_weightsync_deadlock_20260613.md` |
| P2P-off 正式报告 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_report.md` |
| rollout=4 0.5 OOM 证据 | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631_rollout4_oom_gmem05.log` |
| P2P-off smoke timing | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_update_p2poff_smoke_20260613_1321_rollout2_timing.txt` |
| P2P-off smoke log | `src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_update_p2poff_smoke_20260613_1321_rollout2.log` |
| 当前正式 sweep run group | `grpo_sync_a800_refcard3_a4000_parallel_p2poff_formal_20260613_133631` |
