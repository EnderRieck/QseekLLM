# fully async GRPO split ref 调研与 smoke 记录 · 2026-06-13

> 目标:评估“全量异步 GRPO”在当前仓库里的实现边界:train、rollout、reward、ref 尽量流水线并行。
> 本次资源放置沿用同步 smoke:actor 全参训练放 A800 GPU1,ref 放 A4000 GPU3,rollout/vLLM 放 A4000 GPU4-7。

## 1. 结论

当前仓库原本已经有 VERL 的 `experimental.fully_async_policy`,覆盖了 train/rollout streaming async 和 rollout 侧 reward async,但没有完全覆盖我们这次想要的形态:

- 原生已有:rollouter 和 trainer 通过 `MessageQueue` 并发运行。
- 原生已有:reward 在 rollout/agent loop worker 里随样本生成执行,trainer 收到 batch 时 reward/score 已经在样本里。
- 原生已有:old logprob 可以由 rollout 侧 `calculate_log_probs=True` 返回,trainer 侧不必再做 actor old forward。
- 原生不足:ref logprob 仍在 trainer step 内同步 forward,并且原 resource mapping 不能表达 actor=GPU1、ref=GPU3、rollout=GPU4-7。
- 本次新增:split fully async runner、独立 ref pool、sample/micro-batch 级 ref service ready queue、actor update `asyncio.to_thread`、更完整 timing/GPU 监控。

当前已经实现到 ref service 版本:rollout sample 一进 `MessageQueue`,trainer 内的 ref service 就按 micro-batch 在 GPU3 上算 ref logprob,算完的 sample 放入 ready queue;训练 step 只从 ready queue 取已带 `ref_log_prob` 的样本拼 batch。1-step smoke 已跑通,证明 train/rollout/reward/ref 都能在目标资源布局下异步流水。

需要注意:这个 1-step smoke 仍有首批 ready queue 冷启动等待,不能代表长稳态吞吐。下一步要用 `trigger_parameter_sync_step=4, staleness_threshold=0.5` 跑多步,验证 ready queue 是否能在 actor update/param sync 期间持续填充。

## 2. 新增/修改入口

| 文件 | 作用 |
|---|---|
| `RL/main_grpo_fully_async_split.py` | 新增 Hydra 入口,创建 actor/ref/rollout 分离 resource pool |
| `RL/run_grpo_fully_async_split_a800_ref3_a4000.sh` | 本次 fully async smoke/formal runner,封装资源、参数、GPU monitor、timing parser |
| `RL/run_grpo_fully_async_fast_a800_ref3_a4000.sh` | 快速异步默认 runner,默认 `sync4/stale0.5/ref_service=True` |
| `verl/experimental/fully_async_policy/fully_async_trainer.py` | 增加 ref service ready queue、batch-level ref future fallback、actor update `to_thread`、metrics flush、actor/ref 设备日志 |
| `verl/experimental/fully_async_policy/fully_async_rollouter.py` | 增加 rollout worker 设备日志 |
| `verl/workers/engine/fsdp/transformer_impl.py` | 修复 rollout weight sync 传输 dtype,避免 A4000 同步阶段 fp32 临时张量 OOM |

核心实现位置:

| 功能 | 位置 |
|---|---|
| split resource pool | `RL/main_grpo_fully_async_split.py:61` |
| split task runner | `RL/main_grpo_fully_async_split.py:98` |
| ref ready sample 数据结构 | `fully_async_trainer.py:72` |
| ref service 启停 | `fully_async_trainer.py:607` |
| ref micro-batch 计算 | `fully_async_trainer.py:674` |
| ready queue 拼训练 batch | `fully_async_trainer.py:726` |
| batch-level ref future fallback | `fully_async_trainer.py:791` |
| actor update 让出事件循环 | `fully_async_trainer.py` |
| metrics flush,避免结束时额外 sync | `fully_async_trainer.py` |
| bf16 rollout weight transfer | `transformer_impl.py:964` |

## 3. 资源与参数

5-step 验证命令:

```bash
cd /data/zilu/QseekLLM/src/post_train
RUN_GROUP=grpo_fully_async_split_a800_ref3_a4000_5step_bf16sync \
NSTEP=5 \
bash RL/run_grpo_fully_async_split_a800_ref3_a4000.sh 4
```

资源放置:

| 角色 | 物理 GPU | 验证日志 |
|---|---|---|
| actor/update | GPU1 A800 80GB | `[FullyAsyncTrainer] actor CUDA_VISIBLE_DEVICES=['1']` |
| ref forward | GPU3 A4000 16GB | `[FullyAsyncTrainer] ref CUDA_VISIBLE_DEVICES=['3']` |
| rollout/vLLM | GPU4/5/6/7 A4000 16GB | `[FullyAsyncRollouter] rollout CUDA_VISIBLE_DEVICES=['4', '5', '6', '7']` |

核心参数:

| 参数 | 值 | 说明 |
|---|---:|---|
| `algorithm.adv_estimator` | `grpo` | GRPO |
| actor | full-param,`lora_rank=0` | A800 单卡 FSDP(NO_SHARD) |
| `PPO_MINI_BATCH_SIZE` | 32 | trainer mini batch |
| `REQUIRE_BATCHES` | 2 | 每次 trainer 取 2 个 mini batch |
| 每步 prompt 数 | 64 | `32*2` |
| `rollout.n` | 8 | 每个 prompt 采 8 条 |
| 每步 trajectory 数 | 512 | `64*8` |
| `rollout.total_rollout_steps` | 320 | 5 train steps |
| `async_training.staleness_threshold` | 0.1 | 允许少量 stale,本次无 drop |
| `async_training.trigger_parameter_sync_step` | 1 | 每个 actor update 后同步一次参数 |
| `async_training.partial_rollout` | `True` | 参数同步时保留/恢复未完成 rollout |
| `rollout.gpu_memory_utilization` | 0.5 | 只约束 vLLM KV/cache 规划,不覆盖 weight sync 临时显存 |
| `rollout.max_num_batched_tokens` | 8192 | vLLM 调度 token 上限 |
| `rollout.max_num_seqs` | 128 | 单 replica 同时调度序列数上限 |
| `actor_rollout_ref.rollout.calculate_log_probs` | `True` | rollout 侧返回 old logprob |
| `actor_rollout_ref.actor.use_rollout_log_probs` | `True` | trainer 侧使用 rollout old logprob |
| `algorithm.rollout_correction.bypass_mode` | `True` | 跳过 actor old forward |
| reward workers | 16 | agent loop/reward 侧并发 |
| `fully_async_split.ref_service` | `True` | 启用独立 ref service ready queue |
| `fully_async_split.ref_micro_batch_size` | 16 | ref service 每次最多处理 16 个 prompt sample,即最多 128 条 response |
| `fully_async_split.ref_micro_batch_timeout_s` | 0.2 | 未凑满 micro-batch 时的 flush 等待 |
| `fully_async_split.ready_queue_size` | 128 | trainer 侧已算 ref 样本队列容量 |

5-step 日志路径:

| 类型 | 路径 |
|---|---|
| stdout log | `logs/grpo_fully_async_split_a800_ref3_a4000_5step_bf16sync_rollout4_steps5.log` |
| TensorBoard | `logs/tb_grpo_fully_async_split_a800_ref3_a4000_5step_bf16sync_rollout4_steps5` |
| GPU monitor CSV | `logs/grpo_fully_async_split_a800_ref3_a4000_5step_bf16sync_rollout4_steps5_gpu.csv` |
| timing parse | `logs/grpo_fully_async_split_a800_ref3_a4000_5step_bf16sync_rollout4_steps5_timing.txt` |

## 4. batch-level ref future 5-step 对照

本次命令退出码为 0。TensorBoard scalar 正常落盘:

```text
tag 数=97
step 数=6 (steps [0]..[5])
total time=867.69s
```

`total time` 包含 Ray/vLLM/FSDP 初始化、CUDA graph capture 和结束清理,不应直接作为单步训练开销。训练步用 `timing_s/step` 看。

### 4.1 全部 5 个训练步均值

| 阶段 | 时间 | 占 step |
|---|---:|---:|
| `timing_s/step` | 119.40s | 100.0% |
| `gen` | 85.25s | 71.4% |
| `prefetch_wait` | 57.07s | 47.8% |
| `update_actor_async` | 31.09s | 26.0% |
| `update_actor` | 31.09s | 26.0% |
| `ref_async_total` | 26.95s | 22.6% |
| `ref` | 26.95s | 22.6% |
| `ref_async_join` | 26.57s | 22.2% |
| `param_sync` | 4.63s | 3.9% |
| `ref_async_submit` | 0.38s | 0.3% |
| `adv` | 0.03s | 0.0% |
| `reward` | 0.00s | 0.0% |

step1 包含明显冷启动/首批 rollout 长尾,`gen=191.45s`,`prefetch_wait=192.20s`,会把均值拉高。

### 4.2 稳态近似:step2-step5 均值

| 阶段 | step2-step5 均值 | 观察 |
|---|---:|---|
| `timing_s/step` | 84.13s | 多步后稳定在 83-85s |
| `gen` | 58.71s | 其中一部分已和上一轮 update/sync 重叠 |
| `prefetch_wait` | 23.29s | trainer 仍明显等待下一批样本 |
| `update_actor_async` | 30.04s | actor update 稳定约 29-31s |
| `ref_async_total` | 26.52s | ref forward 约 25-28s |
| `ref_async_join` | 26.23s | ref forward 大部分仍在 join 阶段等待 |
| `param_sync` | 4.52s | 首次以外的权重同步约 4.5s |
| `fully_async/ref/overlap_s` | 0.29s | 当前 batch-level future 对 ref 本身 overlap 很小 |

解释:

- 当前实现已经让 prefetch 任务在 actor update 前启动,因此下一批 rollout collection 可以吃到一部分 update window。
- 但 ref future 是“整批样本收齐后”才提交,当前 rollout collection 仍较慢,导致 ref forward 往往刚提交就进入下一步 join,所以 ref 本身 overlap 不明显。
- 这个问题已经用 ref service 版本继续推进,见下一节。

### 4.3 参数同步

本次 5-step 有一次初始同步和 5 次训练后同步:

| 同步点 | 耗时 |
|---|---:|
| initial global step 0 | 26.9980s |
| train step 1 | 5.0393s |
| train step 2 | 4.5483s |
| train step 3 | 4.4673s |
| train step 4 | 4.5211s |
| train step 5 | 4.5567s |

结束阶段没有额外多做一次无意义 `update_weights`;日志在 step5 sync 后直接 `Training stopped by queue termination signal`。

### 4.4 显存与 GPU 利用率

外部 `nvidia-smi` 采样峰值:

| GPU | 角色 | 峰值显存 | 峰值 util | 平均 util |
|---:|---|---:|---:|---:|
| 0 | 非本次主动使用 | 3.29/15.99GB | 48% | 0.9% |
| 1 A800 | actor/update | 73.49/80.00GB | 100% | 22.2% |
| 2 | 非本次主动使用 | 3.29/15.99GB | 49% | 0.6% |
| 3 A4000 | ref | 13.92/15.99GB | 100% | 15.1% |
| 4 A4000 | rollout | 13.75/15.99GB | 100% | 17.7% |
| 5 A4000 | rollout | 13.75/15.99GB | 100% | 17.3% |
| 6 A4000 | rollout | 13.71/15.99GB | 100% | 42.5% |
| 7 A4000 | rollout | 13.90/15.99GB | 100% | 17.3% |

actor worker 内部显存:

| 指标 | mean | max |
|---|---:|---:|
| `actor/perf/max_memory_allocated_gb` | 44.42GB | 45.37GB |
| `actor/perf/max_memory_reserved_gb` | 76.19GB | 76.48GB |
| `actor/perf/cpu_memory_used_gb` | 103.39GB | 103.72GB |

### 4.5 reward 与样本

| 指标 | 趋势 |
|---|---|
| `critic/rewards/mean` | 0.141,0.175,0.182,0.198,0.162 |
| `critic/score/mean` | 0.141,0.175,0.182,0.198,0.162 |
| `timing_s/reward` | 0.000,0.000,0.000,0.000,0.000 |

`timing_s/reward≈0` 不代表 reward 没算,而是 reward 已在 rollout/agent loop 侧异步完成。若后面要汇报 reward verifier 的细分耗时,需要在 reward worker/agent loop 内再加 timer。

## 5. ref service 1-step smoke 结果

ref service 1-step smoke 命令:

```bash
cd /data/zilu/QseekLLM/src/post_train
RUN_GROUP=grpo_fully_async_refsvc_smoke2_20260613 \
NSTEP=1 \
TRIGGER_PARAMETER_SYNC_STEP=1 \
STALENESS_THRESHOLD=0.5 \
REF_SERVICE=True \
REF_MICRO_BATCH_SIZE=16 \
REF_MICRO_BATCH_TIMEOUT_S=0.2 \
bash RL/run_grpo_fully_async_split_a800_ref3_a4000.sh 4
```

结果:

| 阶段 | 时间 |
|---|---:|
| `timing_s/step` | 87.01s |
| `ready_wait` / `gen` | 51.22s |
| `update_actor_async` | 31.30s |
| `update_actor` | 31.30s |
| `ref_service_compute` | 30.29s |
| `ref_service_submit` | 0.58s |
| `ref_service_queue_to_ready_mean` | 5.00s |
| `param_sync` | 4.46s |
| `ref_async_join` | 0.00s |
| `ref_async_total` | 0.00s |
| `ref` | 0.00s |

ref service 关键日志:

```text
[FullyAsyncTrainer][RefService] starting micro_batch_size=16 timeout_s=0.2 ready_queue_size=128
[FullyAsyncTrainer][RefService] batch_id=7 samples=16 seqs=128 assemble=0.03s submit=0.12s ref=8.08s ready_q=16
[FullyAsyncTrainer][RefService] batch_id=8 samples=16 seqs=128 assemble=0.02s submit=0.13s ref=7.60s ready_q=16
[FullyAsyncTrainer][RefService] ready samples 64/64, ready_q=2
```

解释:

- trainer step 内已经不再同步算 ref:`timing_s/ref=0`,`ref_async_join=0`,`ref_async_total=0`。
- GPU3 上 ref service 实际完成了 9 个 micro-batch,累计 `ref_service_compute=30.29s`。
- `ready_wait=51.22s` 是首批 ready queue 冷启动等待,因为 1-step 从空队列开始,不代表多步稳态。
- 结束后出现一次 `not enough ready samples: 9/64`,这是 rollout 结束信号到达后的尾部停止,已完成的 step1 不受影响。

显存峰值:

| GPU | 角色 | 峰值显存 | 峰值 util | 平均 util |
|---:|---|---:|---:|---:|
| 1 A800 | actor/update | 56.21/80.00GB | 100% | 11.0% |
| 3 A4000 | ref service | 13.52/15.99GB | 100% | 9.9% |
| 4 A4000 | rollout | 14.43/15.99GB | 100% | 12.2% |
| 5 A4000 | rollout | 14.50/15.99GB | 100% | 11.0% |
| 6 A4000 | rollout | 14.21/15.99GB | 100% | 11.2% |
| 7 A4000 | rollout | 14.27/15.99GB | 100% | 11.3% |

日志路径:

| 类型 | 路径 |
|---|---|
| stdout log | `logs/grpo_fully_async_refsvc_smoke2_20260613_rollout4_steps1.log` |
| TensorBoard | `logs/tb_grpo_fully_async_refsvc_smoke2_20260613_rollout4_steps1` |
| GPU monitor CSV | `logs/grpo_fully_async_refsvc_smoke2_20260613_rollout4_steps1_gpu.csv` |
| timing parse | `logs/grpo_fully_async_refsvc_smoke2_20260613_rollout4_steps1_timing.txt` |

## 6. 遇到的问题、分析和解决

### 6.1 原 fully async 不能把 ref 单独放 GPU3

现象:VERL 原 `separation/utils.py` 的 resource mapping 会把 `Actor/ActorRollout/Critic/RefPolicy` 都映射到 trainer pool,无法表达 actor=GPU1、ref=GPU3、rollout=GPU4-7。

解决:新增 `RL/main_grpo_fully_async_split.py`,创建独立 `actor_pool` 和 `ref_pool`,rollout replicas 继续走 `rollout.n_gpus_per_node`。

### 6.2 原 ref 仍是 trainer 内同步 forward

现象:原 fully async trainer 每个 step 里直接算 ref logprob,ref forward 是 trainer 关键路径的一段同步阻塞。

阶段一解决:新增 `RefLogProbFuture` 和 `PrefetchedBatch`。prefetch 阶段收 batch 后立即提交 ref forward,trainer step 通过 future join。这个版本能把 ref worker 从 trainer 主线程剥离,但因为 ref 是整批提交,稳态 overlap 仍有限。

阶段二解决:新增 ref service ready queue。ref service 从 `MessageQueue` 直接消费 rollout sample,按 `ref_micro_batch_size` 或 timeout 组 micro-batch 算 ref logprob,再把带 `ref_log_prob` 的 `RefReadySample` 放入 ready queue。trainer step 从 ready queue 取够 64 个 sample 后再 assemble/balance batch,所以 step 内 `timing_s/ref=0`。

### 6.3 ref service 首版尾部 sample 不够

现象:第一次 ref service smoke 中,`rollout.total_rollout_steps` 刚好等于训练需要的 64 个 sample。由于 ref service 是异步消费,rollout 结束信号到达时 ready queue 只凑到 `57/64`,没有进入训练步。

分析:全异步 service 需要 tail buffer。rollout 生产、MessageQueue、ref micro-batch、ready queue 之间存在队列滞后;如果 total rollout 数量和训练消费数量完全相等,最后一个训练 batch 容易被结束信号截断。

解决:runner 在 `REF_SERVICE=True` 时默认把 `rollout.total_rollout_steps` 设为训练需求量加 `REF_SERVICE_ROLLOUT_BUFFER`。当前默认 buffer 等于 `REF_MICRO_BATCH_SIZE=16`,所以 1-step smoke 实际生产 `64+16=80` 个 sample,已跑通。

### 6.4 actor update 会阻塞 async event loop

现象:actor update 包含反传和 optimizer step,如果直接同步执行,会阻塞同一 async actor 的事件循环,使 prefetch task 无法推进。

解决:把 `_fit_update_actor` 包成 `asyncio.to_thread`,对应 timing 字段为 `update_actor_async`。5-step 中 `update_actor_async` 和 `update_actor` 时间一致,说明只是让出事件循环,没有改变训练逻辑。

### 6.5 metrics flush 原本依赖下一次 sync

现象:原 fully async metrics aggregator 主要在 `_fit_update_weights()` 后 flush。若删除结束时额外 sync,最后一批 timing 可能不落盘。

解决:新增 `_flush_metrics_aggregator()`,在 `_fit_postprocess_step()` 主动 flush。5-step 解析到 `tag 数=97`,说明 timing/reward/perf scalar 已正常写入。

### 6.6 结束阶段多一次无意义 update_weights

现象:原逻辑停止后会再调用一次 `_fit_update_weights()` 以 flush metrics,会污染 timing,也会多一次 rollout 权重同步。

解决:结束时只 flush metrics,不再额外同步。5-step 日志只有 initial sync 和每个 train step 后的 sync。

### 6.7 为什么 `gpu_memory_utilization=0.5` 仍然可能 OOM

现象:第一次 5-step 在 train step 1 的 `update_weights` 阶段 OOM,A4000 上 vLLM 已占约 13GB,checkpoint worker 还要临时申请约 1.14GiB。

分析:

- `gpu_memory_utilization=0.5` 只影响 vLLM 对 KV/cache 的规划,不限制参数同步时额外加载权重的临时显存。
- 当前 standalone vLLM server 的 `release_kv_cache()` 对这条路径基本没有释放出足够空间,所以 sync 阶段仍会叠加显存压力。
- actor 是单卡 FSDP NO_SHARD,`module.state_dict()` 返回普通 tensor。原代码只把 `DTensor` path 转为 bf16,普通 floating tensor 仍按 fp32 传给 rollout,导致 A4000 同步时临时显存过高。

解决:在 `transformer_impl.py` 的 rollout weight transfer 路径中,对非 DTensor floating params 也转成 bf16:

```python
if isinstance(param, DTensor):
    converted = param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
elif torch.is_floating_point(param):
    converted = param.to(device=device, dtype=torch.bfloat16, non_blocking=True)
else:
    converted = param
```

修复后同样 `gpu_memory_utilization=0.5` 的 5-step 已跑通,step1 后续同步耗时约 4.5-5.0s,无 OOM。

### 6.8 Ray 在 sandbox 内无法绑定本地端口

现象:sandbox 内启动 Ray GCS 失败:

```text
Operation not permitted ... bind 0.0.0.0:0
```

解决:这类 RL/Ray smoke 需要非 sandbox 权限运行。代码层面无需修改。

## 7. 当前能说明什么

已经说明:

- split fully async 入口能在目标资源布局上完整跑 5 个 GRPO train steps。
- actor/ref/rollout 资源放置符合要求:GPU1/GPU3/GPU4-7。
- reward 在 rollout/agent loop 侧异步计算,trainer 侧不再承担 reward 计算。
- old logprob 从 rollout 侧返回,不需要 actor old forward。
- ref service 在 GPU3 上按 sample/micro-batch 计算 ref logprob,trainer step 内 `timing_s/ref=0`。
- 5-step 下参数同步稳定,之前 A4000 sync OOM 已修复。
- ref service 1-step smoke 跑通,并修复了首版 tail buffer 不够的问题。
- metrics、GPU 监控和 timing parse 已正常落盘。

还需要继续验证:

- 真正异步权重版本下的效率,即 `trigger_parameter_sync_step>1` 和更大的 `staleness_threshold`。
- ref service 在多步稳态下是否能让 ready queue 提前填满,降低 `ready_wait`。
- reward/verifier 细分耗时。目前只有 trainer 侧 `timing_s/reward≈0`,还缺 reward worker 内部 timer。
- 更长步数压力测试,建议 20 step 以上,观察 stale drop、rollout pending queue 和显存峰值。

## 8. 建议下一步

先跑真正异步参数,可以直接用 fast wrapper:

```bash
cd /data/zilu/QseekLLM/src/post_train
RUN_GROUP=grpo_fully_async_fast_refsvc_sync4_stale05 \
NSTEP=5 \
bash RL/run_grpo_fully_async_fast_a800_ref3_a4000.sh 4
```

对比重点:

| 指标 | 预期观察 |
|---|---|
| `prefetch/ready_at_step_start` | 多步后应升高,说明下一批已提前准备好 |
| `prefetch_wait` | 应下降,否则 trainer 仍在等 rollout |
| `ready_wait` | 多步稳态应下降,否则 trainer 仍在等 ref-ready 样本 |
| `ref_service_compute` | 应主要在后台发生,不再出现在 `timing_s/ref` |
| `trainer/idle_ratio` | 应下降 |
| `stale_trajectory_processed` / `dropped_stale_samples` | 观察 staleness 阈值是否过紧 |
| `param_sync` | 单次同步约 4.5s,但 sync4 会降低同步频率 |

如果 `sync4/stale0.5` 仍主要卡在 `ready_wait`,说明瓶颈不是 trainer 同步 ref,而是 rollout 产样或单卡 ref service 吞吐跟不上;下一步应考虑提高 rollout 并发、调大 ref micro-batch、或给 ref 增加 GPU。
