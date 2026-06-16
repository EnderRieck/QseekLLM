# 异步 GRPO 系统架构说明

> 本文是整体架构的导览文档，回答"我们用了 verl 的什么、自己加了什么、如何组合在一起"。
> 各个组件的排障细节、timing 数据、退化诊断见同目录下对应的专项文档。

---

## 1. verl 提供的骨架

verl 的 `fully_async_policy` 模块提供了一个**双 actor + 消息队列**的异步架构：

```
FullyAsyncRollouter          MessageQueue          FullyAsyncTrainer
  （采样侧 Ray actor）   →→→ 样本流过去 →→→     （训练侧 Ray actor）
  vLLM async server
  每个 prompt 采 n 条
  reward 计算（异步）
                                                   actor update
                                                   （backward + optim）
         ←←←←←←← 权重同步（NCCL broadcast） ←←←←←←←
```

Rollouter 不停地让 vLLM 产样本、算 reward、写入 MessageQueue。
Trainer 从队列消费 batch、训练，定期把新权重同步回 Rollouter 的 vLLM。
两者通过队列解耦，异步并行，互不阻塞。

verl 原生已有：
- rollouter / trainer 并发通过 MessageQueue 运行
- reward 在 rollout/agent loop 侧异步计算
- old logprob 从 rollout 侧返回（`calculate_log_probs=True`），trainer 不额外做一次 actor old forward
- `paused` / `_resume_event` 锁机制（用于权重同步期间暂停采样）
- NCCL checkpoint engine（actor → rollout 的权重广播）

verl 原生不支持的（我们全部新加）：
- Actor / Ref / Rollout 放不同 GPU（资源分离）
- Ref logprob 的异步化（ref service）
- 过程评测复用采样侧不新开卡
- per-update / per-ref-batch 粒度的事件日志

---

## 2. 我们的整体结构

```
RL/main_grpo_fully_async_split.py
│
├── actor_pool → GPU1 (A800 80G)
│     FullyAsyncTrainer（verl，我们大幅扩展）
│       ├── actor FSDP 全参训练
│       └── ref service task（后台）──→ GPU3 ──→ ready_queue
│
├── ref_pool → GPU3 (A4000 16G)
│     RefPolicy worker（verl 原生，DetachActorWorker）
│
├── rollout → GPU4-7 (A4000 16G × 4)
│     FullyAsyncRollouter（verl，我们小幅扩展）
│       ├── vLLM async server（多卡 DP 副本）
│       ├── reward 计算 → data_pipeline/reward.py
│       └── pause / resume（用于过程评测）
│
└── MessageQueue（verl 原生，连接 Rollouter → Trainer）
```

### 启动命令

```bash
cd /data/zilu/QseekLLM/src/post_train

# 正式跑（v2 配置，4 张 rollout 卡）
RUN_GROUP=grpo_v2curric_s3r1_normadvF_kl005_noshaping_n16_t12 \
TRAIN_FILES=parquet/train_rl_s4_v2curric.parquet \
VAL_FILES=parquet/val_rl_s4clean_fix.parquet \
ROLLOUT_N=16 ROLLOUT_TEMPERATURE=1.2 \
TEST_FREQ=10 SAVE_FREQ=50 VAL_BEFORE_TRAIN=True \
bash RL/run_grpo_fully_async_split_a800_ref3_a4000.sh 4

# smoke（1 步验证）
NSTEP=1 bash RL/run_grpo_fully_async_split_a800_ref3_a4000.sh 4 1
```

---

## 3. 我们新增的各组件

### 3.1 资源分离放卡

**问题**：verl `separation/utils.py` 的 `create_resource_pool_manager` 把
`[Actor, ActorRollout, Critic, RefPolicy]` 全部映射到同一个 `trainer_pool`。
同一 pool 内的角色会被 `create_colocated_worker_cls` 打包成 colocated worker，
调度到同一批物理 GPU 上。无论 `ref_in_actor` 是 True 还是 False，都做不到
"Actor 在 GPU1、Ref 在 GPU3" 这种物理隔离。

**解法**：新写 `RL/main_grpo_fully_async_split.py`，内含
`create_split_resource_pool_manager`，拆出两个独立 pool：

```python
resource_pool_spec["actor_pool"] = [1]   # GPU1
mapping[Role.Actor] = "actor_pool"

resource_pool_spec["ref_pool"]   = [1]   # GPU3
mapping[Role.RefPolicy] = "ref_pool"     # 单独一个 pool
```

Runner 通过 `CUDA_DEVICE_ORDER=PCI_BUS_ID` + 各侧的 `CUDA_VISIBLE_DEVICES` 控制
Ray 调度到正确的物理卡。

详见：`RL/main_grpo_fully_async_split.py:61`

### 3.2 Ref Service（ref logprob 异步化）

**问题**：verl 原版在 trainer step 内同步算 ref logprob，耗时 ~27s，是关键路径的阻塞段。
即便有 ref prefetch（先提交、后 join），整批样本一次性提交，ref 和 actor update 的
重叠也很有限。

**解法**：在 `FullyAsyncTrainer` 里加 ref service——一个独立的 `asyncio.Task`，
从 MessageQueue 的出口直接消费 rollout 样本，按 `ref_micro_batch_size` 组 batch
在 GPU3 上算 ref logprob，算完的样本放入 `ready_queue`。Trainer step 从 `ready_queue`
取已带 `ref_log_prob` 的样本拼 batch，step 内 `timing_s/ref=0`。

```
MessageQueue ──→ Rollouter 消费（训练采样）
             └─→ ref service（GPU3，后台 task）──→ ready_queue ──→ Trainer step
```

注意事项：
- **tail buffer**：rollout 总量需比训练消费量多 `REF_SERVICE_ROLLOUT_BUFFER`（默认等于
  `REF_MICRO_BATCH_SIZE`），否则 rollout 结束信号到达时 ready_queue 凑不满最后一个 batch。
- **actor update 不阻塞 event loop**：actor 反传 + optim step 包进 `asyncio.to_thread`，
  让 asyncio event loop 在训练期间仍能推进 ref service task。

关键参数（runner 环境变量）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `REF_SERVICE` | `True` | 是否启用 ref service |
| `REF_MICRO_BATCH_SIZE` | 16 | ref service 每次最多处理的 prompt 数 |
| `REF_MICRO_BATCH_TIMEOUT_S` | 0.2 | 未凑满 micro-batch 时的 flush 等待 |
| `REF_SERVICE_ROLLOUT_BUFFER` | 等于 `REF_MICRO_BATCH_SIZE` | tail buffer 大小 |

详见：`verl/experimental/fully_async_policy/fully_async_trainer.py`，类 `EventLogger`
之后的 `RefReadySample` / `_start_ref_service_if_needed` / `_ref_service_loop`

### 3.3 过程评测复用采样侧

**需求**：训练过程中定期评测，不想新开 GPU，也不能让 eval 和采样抢 vLLM 引擎。

**解法**：复用 Rollouter 那侧的 vLLM 引擎跑 val（`use_trainer_do_validate=False`，
走 `rollouter.do_validate.remote()`，n=1 贪心）。评测前先暂停采样：

```python
# fully_async_rollouter.py（新增）
async def pause_for_eval(drain_timeout_s=180):
    # 复用已有的 paused / _resume_event 锁（权重同步时也用这套）
    # 暂停、等 in-flight 请求 drain（有超时保护，绝不无限挂起）

async def resume_after_eval():
    # 恢复采样
```

Trainer 侧包一层保障：

```python
await rollouter.pause_for_eval.remote()
try:
    await rollouter.do_validate.remote(...)
finally:
    await rollouter.resume_after_eval.remote()   # 即使 eval 抛异常也恢复
```

评测结果全量落盘 `logs/<RUN>_val_dumps/<step>.jsonl`（含 input / output / gts / score / correct 等字段）。

关键参数：

| 变量 | 默认 | 含义 |
|---|---|---|
| `TEST_FREQ` | 1000000（关闭）| 每 N 个 param_version 评测一次 |
| `VAL_BEFORE_TRAIN` | False | 是否在第 0 步做基线评测 |
| `VAL_FILES` | smoke 路径 | val parquet 路径 |
| `VAL_DUMP_DIR` | `logs/<RUN>_val_dumps` | 评测 I/O 落盘目录 |
| `LOG_VAL_GENERATIONS` | 20 | 抽样进 TensorBoard 的生成条数 |

### 3.4 EventLogger（事件日志）

**问题**：TensorBoard 在 `sync>1` 时把多次 actor update 聚合成一条记录，
per-update 粒度丢失；ref service 的逐 micro-batch 信息只在 stdout 飘过，无法事后分析。

**解法**：在 `FullyAsyncTrainer` 里加 `EventLogger`，把关键事件写成带绝对时间戳的 JSONL。
写日志失败不中断训练（全程 try/except）。

| 事件 `ev` | 记录内容 |
|---|---|
| `ref_micro_batch` | batch_id / samples / seqs / assemble_s / compute_s / ready_q 深度 |
| `update_actor_start/end` | 每次 actor update 起止，还原 sync>1 抹平的 per-update 粒度 |
| `param_sync_start/end` | 每次权重同步起止及耗时 |
| `validate_start/end` | 每次过程评测起止 |
| `step` | 每个 logged step 的全量 timing 快照 |

分析器：

```bash
python3 RL/analyze_events.py logs/<RUN_NAME>_events.jsonl
# 输出：per-update 墙钟分布、ref ready_q 深度趋势、ref-compute 被 update 隐藏的并发比例
```

路径由 runner 通过 `EVENT_LOG_PATH` 注入，默认 `logs/<RUN>_events.jsonl`。

---

## 4. Bug 修复

### 4.1 权重同步 OOM（bf16 转换缺失）

**现象**：rollout 侧权重同步时 A4000 OOM。

**根因**：`transformer_impl.py` 只对 `DTensor` 类型参数转 bf16，
普通 floating tensor 按 fp32 传输（1.7B 全参 = 6.8GB），与 vLLM 同卡显存叠加导致 OOM。

**修复**（`verl/workers/engine/fsdp/transformer_impl.py`）：

```python
if isinstance(param, DTensor):
    converted = param.to(device).full_tensor().to(torch.bfloat16)
elif torch.is_floating_point(param):          # ← 新增
    converted = param.to(device=device, dtype=torch.bfloat16)
else:
    converted = param
```

传输量 6.8GB → 3.4GB，OOM 消失，同步耗时约 4.5s/次。

### 4.2 NCCL P2P 死锁

**现象**：rollout=2 时权重同步卡死，GPU 100% util 但无进展。

**根因**：本机零 NVLink，纯 PCIe，A800(GPU1) 与 A4000(GPU4/5) 跨代且跨 PCIe bridge，
NCCL 默认 P2P 路径在此拓扑下不报错、直接 hang。

**修复**：`NCCL_P2P_DISABLE=1`（走 SHM/host 中转），`NCCL_REBUILD_GROUP=1`（每次
finalize 后销毁 NCCL group）。已写入 runner 默认值。

验证脚本：`RL/nccl_broadcast_smoke.py`（最小 Ray NCCL broadcast 复现/验证）。

详见：`docs/rl_nccl_weightsync_deadlock_20260613.md`

### 4.3 判分器假阳性

`data_pipeline/reward.py` 共修了两处：

1. **math_verify 裸字符串退化**：`parse()` 对非 `$...$` 包裹的字符串退化为"抽取数字"，
   导致 `6+9i` vs `6-3i` 被判同。修复：进 `parse()` 前包上 `$...$`。
2. **MCQ 裸字母假阳**：无格式时扫尾部 80 字符 `\b[A-D]\b`，数学正文里的 `sinA` 会撞上
   gold=A。修复：无格式时只认明确作答声明或末行单独字母。

详见：`docs/reward_verifier_fix_20260612.md`

---

## 5. GRPO 训练经验：v1 退化与 v2 修复

### v1 退化现象（param_version 0→50）

| 难度带 | v0 | v50 | 结论 |
|---|---|---|---|
| easy（>90% 基线） | 99.6% | 78.4% | 本来稳对的被破坏 |
| 可学习带（10-90%） | 36.2% | 26.4% | 单调下滑，未转正 |
| format | 82% | 94% | 先薅满格式分 |

翻转矩阵：对→错 498 vs 错→对 181，**破坏是建设的 3 倍**。

### 根因

1. `norm_adv_by_std=True` × 全错组：40% 竞赛题全部采样错，correct 方差为零，
   但 format/长度 shaping 仍有连续差异，除以极小 std 把 shaping 噪声放大成伪梯度。
2. shaping 过重：`format_bonus=0.1 + think_len_bonus≤0.2 = 0.3`，在难题上喧宾夺主。
3. KL 太松（0.001）：锚不住 SFT 已有能力，策略漂走丢知识。
4. 数据：40% 竞赛集 p≈0，零有效梯度 + 上述伪梯度来源。

### v2 修复包（当前正在运行）

| 改动 | v1 → v2 | 目的 |
|---|---|---|
| `norm_adv_by_std_in_grpo` | True → **False** | 消除全错组伪梯度放大 |
| `kl_loss_coef` | 0.001 → **0.005** | 锚住 SFT 已有能力 |
| `REWARD_FORMAT_BONUS` | 0.1 → **0.05** | correct 主导，降格式分空间 |
| `REWARD_THINK_LEN_MAX_BONUS` | 0.2 → **0** | 去掉低收益长度分 |
| 训练数据 | 全量混合 → v2 课程池 | 剔零梯度源，留中等带 905K + 10% 难题彩票 |
| 温度 / group size | 1.2 / 16 保留 | 高温探索是 hard 区真增益来源 |

v2 当前结果：easy 恢复至 ~94%，mid 稳在 31-32%（未转正但不再纯跌）。

详见：`docs/rl_degradation_diagnosis_20260613.md`

---

## 6. 关键文件索引

| 类别 | 路径 | 说明 |
|---|---|---|
| **入口** | `RL/main_grpo_fully_async_split.py` | 资源分离入口，创建 split pool |
| **Runner** | `RL/run_grpo_fully_async_split_a800_ref3_a4000.sh` | 正式/smoke runner，所有参数可通过 env 覆盖 |
| **快速 Runner** | `RL/run_grpo_fully_async_fast_a800_ref3_a4000.sh` | 默认 sync4/stale0.5/ref_service=True |
| **Trainer** | `verl/experimental/fully_async_policy/fully_async_trainer.py` | ref service、EventLogger、pause/resume for eval、actor to_thread |
| **Rollouter** | `verl/experimental/fully_async_policy/fully_async_rollouter.py` | pause_for_eval / resume_after_eval |
| **权重同步** | `verl/workers/engine/fsdp/transformer_impl.py` | bf16 transfer fix |
| **权重同步引擎** | `verl/checkpoint_engine/base.py` | serial fallback + 阶段计时 |
| **奖励函数** | `data_pipeline/reward.py` | 统一判分（SFT/eval/RL 共用） |
| **verl 适配** | `RL/reward_verl.py` | reward 薄 wrapper，接 verl custom_reward_function |
| **数据重配** | `RL/reweight_rl_v2curric.py` | v2 课程池切片脚本 |
| **事件分析** | `RL/analyze_events.py` | events.jsonl 离线分析 |
| **NCCL 验证** | `RL/nccl_broadcast_smoke.py` | 最小 broadcast 复现脚本 |
| **退化诊断** | `docs/rl_degradation_diagnosis_20260613.md` | v1 退化根因 + v2 修复 |
| **NCCL 死锁** | `docs/rl_nccl_weightsync_deadlock_20260613.md` | P2P hang 定位与解锁 |
| **判分修复** | `docs/reward_verifier_fix_20260612.md` | math_verify + MCQ 假阳修复 |
| **ref/log/eval** | `docs/rl_logging_and_process_eval_20260613.md` | EventLogger + pause/resume 详细实现 |
| **ref service** | `docs/rl_fully_async_split_ref3_20260613.md` | ref service 设计与 5-step timing |
| **训练计划** | `docs/training_plan.md` | 阶段路线图（F2/S3/S4）与数据资产 |
