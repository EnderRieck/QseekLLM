# NCCL 并行权重同步死锁 · 定位、解锁与运行规范 · 2026-06-13

> 排障结论:`checkpoint_engine` nccl 后端的并行权重同步曾卡在**第一个 NCCL broadcast 的跨卡传输层**,
> 不是编排 / ZMQ / state_dict / buffer 类型的问题。最终确认是 A800 + 多张 A4000 混合 PCIe 拓扑下
> NCCL 默认 P2P 路径会 hang；`NCCL_P2P_DISABLE=1` 已通过最小 broadcast、完整 GRPO rollout=2 smoke,
> 以及 rollout=1/2/3/4 五步正式 sweep 验证。
> VERL 原本就有标准并行同步路径, 我们只是为了先跑通 sweep 临时走过串行 fallback。

## 0. 现场与结论更新

- 入口:`RL/main_grpo_sync_split.py`(`SplitSyncTaskRunner`)→ split placement:A800(card1)训练, GPU3 A4000 ref, GPU4/5/... A4000 rollout 副本。
- 后端:verl 原生 `verl/checkpoint_engine/nccl_checkpoint_engine.py`(backend=nccl),全量同步(`lora_rank=0`,512MB bucket)。
- 现象:连续多个 debug run(zmqsync/trace/deeptrace/notrace/syncbroadcast/torchbuf/torchbuf_state)全部卡在权重同步,0 个成功 step。
- 2026-06-13 13:20 后更新:加 `NCCL_P2P_DISABLE=1` 后,3-rank 512MB 最小 broadcast 正常；完整 GRPO rollout=2、1 step 正常完成。
- 2026-06-13 15:00 后更新:同一环境下 rollout=1/2/3/4 的 5-step 正式 sweep 已完成。rollout=4 曾出现一次 0.5 近阈值 OOM,清理后同样 0.5 重跑通过。
- 当前运行脚本 `src/post_train/RL/run_grpo_sync_a800_ref3_a4000_one.sh` 已默认:

```bash
SERIAL_ROLLOUT_WEIGHT_SYNC=0
NCCL_REBUILD_GROUP=1
NCCL_P2P_DISABLE=1
```

这表示后续默认走 VERL 标准并行同步,同时绕开本机混合 GPU 拓扑上的 NCCL P2P hang。

## 1. 铁证:trace 显示死锁在第一个 broadcast 的 NCCL 传输

最新 run `..._torchbuf_state_debug_20260613_130414` 的 `nccl-checkpoint-trace`:

```
rank=0 (A800, 根):  ... broadcast start → previous broadcast wait start   # 卡在等第一个 broadcast 完成
rank=1 (A4000):     recv metadata done → broadcast start                  # 进了 broadcast,未 done
rank=2 (A4000):     recv metadata done → broadcast start                  # 进了 broadcast,未 done
```

**三个 rank 全部 `broadcast start`,无一 `broadcast done`。** 关键旁证:
- `[checkpoint-manager] build_process_group done: 2.08s` → 建组成功。
- rank1/rank2 都打出 `recv metadata done` → **ZMQ PUB/SUB 元数据握手成功**。
- 三方在**同一个**(第一个,bytes=536870912=512MB)broadcast 里。

## 2. 逐层排除

| 层 | 状态 | 依据 |
|---|---|---|
| `base.py:573` 两端编排 | ✅ 正确 | `trainer.update_weights(...) + rollout.update_weights(...)` 拼 future 并发 dispatch,两端都到场 |
| NCCL 组成员/拓扑(`nccl_..engine.build_topology`) | ✅ 正确 | rank0=trainer 根,rank1..N=rollout 收,world=N+1=3 |
| build_process_group | ✅ 成功 | 2.08s 返回 |
| ZMQ 元数据握手(嫌疑 A:slow-joiner / bucket 数不齐) | ✅ 排除 | rank1/2 均 `recv metadata done`,三方同一 broadcast |
| state_dict()/get_per_tensor_param 慢(之前的假设) | ✅ 排除 | trainer world=1,gather 平凡;A800 100% util 是 **NCCL kernel busy-poll**,不是在算 |
| buffer 类型(CuPy vs torch master buffer) | ✅ 排除 | 本 run 已开 `VERL_NCCL_MASTER_TORCH_BUFFER`,照卡 |
| **NCCL broadcast 跨卡传输(嫌疑 B)** | ❌ **死锁点** | 三方进 broadcast 无一完成 |

→ 之前换的 flag(CuPy→torch buffer / ZMQ delay / rebuild group / state_dict trace)**都打在错误的层**。真凶在 NCCL 传输层,而**全程未开 `NCCL_DEBUG`**,等于在传输层外盲调。

## 3. 根因:无 NVLink + 跨代 GPU + 跨 PCIe bridge 的 P2P 死锁

`nvidia-smi topo -m` 实测:

- 全机 **零 NVLink**,GPU 间**纯 PCIe**。GPU0-3 互为 PIX,GPU4-7 互为 PIX,**两组之间 PXB(跨多个 PCIe bridge)**。单 NUMA(node0)。
- A800(GPU1)与 A4000 是**不同代 GPU**。跨代 GPUDirect P2P over PCIe 常不被支持,NCCL 第一个 collective **不干净报错,而是 hang**——正是观察到的症状。

## 4. 推荐解锁与已验证方案

### 4.1 先开 NCCL 日志,看传输层选了什么路、卡在哪
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,P2P,NET
```
关注 `via P2P` / `via SHM` / `via NET` 字样和 `Connected` 是否出现在 A800↔A4000 通道上。

### 4.2 强制绕开跨代 PCIe P2P(已验证一键解锁)
```bash
export NCCL_P2P_DISABLE=1     # 关 GPU 直连 P2P,走 SHM/host 中转
# 若仍卡,再叠加试:
# export NCCL_P2P_LEVEL=SYS
# export NCCL_SHM_DISABLE=1   # 极端情况下连 SHM 也关,逼它走 NET(localhost socket)
```
无 NVLink + 混代 GPU + 跨 PCIe bridge 的机器,`NCCL_P2P_DISABLE=1` 是最常见解锁。代价:带宽降一截,但能先**跑通闭环**,再谈快慢。

本机验证结果:

| 测试 | 环境 | 结果 |
| --- | --- | --- |
| Ray NCCL broadcast, world=3, 1MB | 默认 NCCL P2P | 卡住,30s 无初始化结果 |
| Ray NCCL broadcast, world=3, 1MB | `NCCL_P2P_DISABLE=1` | 通过,约 0.002s |
| Ray NCCL broadcast, world=3, 512MB | `NCCL_P2P_DISABLE=1` | 通过,约 0.115-0.118s |
| 完整 GRPO rollout=2, 1 step | `SERIAL_ROLLOUT_WEIGHT_SYNC=0 NCCL_REBUILD_GROUP=1 NCCL_P2P_DISABLE=1` | 通过,训练步 `update_weights=4.46s` |
| 完整 GRPO rollout=1/2/3/4, 5 step | `SERIAL_ROLLOUT_WEIGHT_SYNC=0 NCCL_REBUILD_GROUP=1 NCCL_P2P_DISABLE=1` | 通过,正式表已生成 |

### 4.3 排查 ray.util.collective 与 vLLM 的 NCCL/CUDA context 冲突(若 4.2 无效)
broadcast 用 `ray.util.collective`,而 rollout 端 vLLM 自带 NCCL。确认两者 communicator/CUDA context 不打架;必要时给权重同步组单独的 `group_name` 与 device 绑定。

### 4.4 复现命令与有效解法

最小复现脚本:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,4,5 \
PYTHONPATH=/data/zilu/QseekLLM/src/post_train/verl \
/data/zilu/QseekLLM/src/post_train/.venv/bin/python \
  RL/nccl_broadcast_smoke.py --world-size 3 --mb 1
```

默认 P2P 下,该 1MB broadcast 也会卡。加入:

```bash
NCCL_P2P_DISABLE=1
```

后,1MB 和 512MB 均通过。

完整 GRPO smoke 命令:

```bash
SERIAL_ROLLOUT_WEIGHT_SYNC=0 \
NCCL_REBUILD_GROUP=1 \
NCCL_P2P_DISABLE=1 \
RUN_GROUP=grpo_sync_a800_refcard3_a4000_parallel_update_p2poff_smoke_20260613_1321 \
bash src/post_train/RL/run_grpo_sync_a800_ref3_a4000_one.sh 2 1
```

关键日志证据:

```text
Rank 0 send weights done, time cost: 2.10s
Rank 1 receive weights done, total_params: 259, time cost: 2.15s, bandwidth: 3.96 GB/s
Rank 2 receive weights done, total_params: 259, time cost: 2.15s, bandwidth: 3.96 GB/s
timing_s/update_weights:4.4566
```

原始文件:

```text
src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_update_p2poff_smoke_20260613_1321_rollout2.log
src/post_train/logs/grpo_sync_a800_refcard3_a4000_parallel_update_p2poff_smoke_20260613_1321_rollout2_timing.txt
```

## 5. 过程复盘:为什么改了这么久

这次耗时主要不是在实现并行同步,而是在排除错误层级:

1. VERL 原本就有并行同步: `CheckpointEngineManager.update_weights()` 的标准路径会一次性把所有 rollout workers 组成 worker group,然后 `trainer.update_weights(...) + rollout.update_weights(...)` 并发等待。
2. 旧 sweep 为了先产出可跑基线,走了我加的 `SERIAL_ROLLOUT_WEIGHT_SYNC=1` 串行 fallback,导致 `update_weights` 随 rollout 卡数线性增长。
3. 切回 VERL 标准并行路径后,rollout=2 卡住。早期怀疑过 ZMQ slow-joiner、state_dict 生成、CuPy/torch buffer 混用、NCCL group 复用,所以加了 trace 和 rebuild/finalize 保护。
4. 最后用最小 Ray NCCL broadcast 证明:完整 GRPO 之外,同一组物理 GPU 上 1MB broadcast 默认也卡。因此根因在 NCCL P2P 传输路径,不是 GRPO 编排。
5. `NCCL_P2P_DISABLE=1` 使最小 broadcast、完整 GRPO rollout=2 smoke 和 rollout=1/2/3/4 正式 sweep 均通过,所以后续正式运行应以此作为固定环境前提。

正式 P2P-off 并行同步均值:

| rollout 卡数 | rollout 物理卡 | step | gen | ref | update_actor | update_weights |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 137.54s | 74.55s | 26.73s | 29.90s | 3.41s |
| 2 | 4,5 | 109.68s | 44.65s | 27.39s | 30.54s | 4.45s |
| 3 | 4,5,6 | 96.67s | 30.55s | 27.29s | 30.60s | 5.43s |
| 4 | 4,5,6,7 | 92.65s | 25.14s | 27.48s | 30.85s | 6.55s |

## 6. 两个方向性提醒

1. **退回 LoRA-only 同步**。本 run `lora_rank=0` 是全量 3.4GB 跨卡 PCIe 广播,即便 NCCL 通了也很慢(估算 2-5s/次),与 CLAUDE.md "优先验证 LoRA-only(0.1-0.3s)、先看现成 TensorLoRARequest 热挂载够不够用"的稳健路线相反。通了之后建议立刻切 LoRA 比对。
2. **通路已经验证,后续看瓶颈**。正式并行表显示 4 卡 `update_weights=6.55s`, 已明显低于串行 fallback 的 14.89s；继续提速时应同时看 ref 单卡 27s、actor update 31s 和全参同步带宽。

## 7. 复现/验证接口

- 拓扑:`nvidia-smi topo -m`
- 死锁 trace:`grep nccl-checkpoint-trace logs/<run>_rollout2.log`
- 阶段计时:`grep checkpoint-manager logs/<run>_rollout2.log`
- 后端源码:`verl/verl/checkpoint_engine/{base.py:389 build_process_group, 500-600 update_weights}`、`nccl_checkpoint_engine.py:60-114 _AsyncBroadcast, 195 build_topology`
