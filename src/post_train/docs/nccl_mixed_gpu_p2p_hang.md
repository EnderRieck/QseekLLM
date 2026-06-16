# 混合 GPU 拓扑下的 NCCL P2P Hang：分析与解决

> 适用场景：多张不同代 GPU 混用（如 A800 + A4000），通过 PCIe 互联，无 NVLink，
> 用 NCCL 做跨卡通信时卡死。
> 本文记录完整的排查思路，供换机器或复现时参考。

---

## 1. 背景

在异步 GRPO 训练中，每次 actor update 完成后需要把新权重广播给 rollout 侧的 vLLM 副本。
这个同步走 NCCL broadcast：actor（GPU1 A800）作为 root，rollout 副本（GPU4/5/... A4000）
作为 receiver。

切到 verl 原生并行同步路径（`SERIAL_ROLLOUT_WEIGHT_SYNC=0`）后，rollout=2 时程序卡死。

---

## 2. 现象

GPU 利用率 100%，但训练完全没有进展。日志停在：

```
[checkpoint-manager] build_process_group done: 2.08s
```

之后没有任何输出。外部 `nvidia-smi` 观察到 GPU1/4/5 util 持续 100%，但 step 计数不动。

加 trace 后看到三个 rank 的状态：

```
rank=0 (A800 root):  broadcast start → 等第一个 broadcast 完成  ← 卡在这里
rank=1 (A4000):      recv metadata done → broadcast start         ← 进了，没出来
rank=2 (A4000):      recv metadata done → broadcast start         ← 进了，没出来
```

**所有 rank 都进入了第一个 broadcast（512MB），无一完成。**

---

## 3. 排查过程

### 3.1 被依次排除的错误方向

| 怀疑层 | 排查方法 | 结论 |
|---|---|---|
| trainer / rollout 两端没并发到场 | 加 trace 看两端是否进入 `update_weights` | 排除：两端都进了 |
| NCCL group 建组失败 | 看 `build_process_group done` 是否出现 | 排除：建组成功，耗时 2.08s |
| ZMQ 元数据握手失败（slow-joiner） | 看 rollout rank 是否收到 metadata | 排除：两个 rank 都打出了 `recv metadata done` |
| state_dict 生成太慢 | 给 `get_per_tensor_param` 加计时 | 排除：卡点在 broadcast，state_dict 已完成 |
| CuPy / torch buffer 类型不兼容 | 开 `VERL_NCCL_MASTER_TORCH_BUFFER` 强制 torch buffer | 排除：换完仍然卡 |
| NCCL communicator 复用冲突 | 加 `NCCL_REBUILD_GROUP=1`，每次 finalize 销毁 group | 部分规避，但不是根因 |

上述排查全部在 GRPO 编排层进行，每次改动都要跑完整流程才能看结果，非常耗时。

### 3.2 关键转折：最小化复现

意识到应该把 GRPO 完全剥离，单独测 broadcast 本身。写了 `RL/nccl_broadcast_smoke.py`：

```python
# 核心逻辑：在指定 GPU 上启动 Ray，做一次 NCCL broadcast，计时打印结果
ray.util.collective.allreduce(tensor, group_name="test")
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1,4,5 \       # A800 + 两张 A4000
python RL/nccl_broadcast_smoke.py --world-size 3 --mb 1
```

结果：**1MB 的 broadcast，默认 NCCL P2P，直接卡死。**

这一步 30 秒就定位到根因在 NCCL 传输层本身，与 GRPO 的任何编排逻辑无关。

---

## 4. 根因

### 4.1 拓扑

```bash
nvidia-smi topo -m
```

```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0     X    PIX   PIX   PIX   PXB   PXB   PXB   PXB
GPU1    PIX    X    PIX   PIX   PXB   PXB   PXB   PXB
GPU2    PIX   PIX    X    PIX   PXB   PXB   PXB   PXB
GPU3    PIX   PIX   PIX    X    PXB   PXB   PXB   PXB
GPU4    PXB   PXB   PXB   PXB    X    PIX   PIX   PIX
...
```

- GPU0-3（含 A800）和 GPU4-7（A4000）之间：**PXB（跨多个 PCIe bridge）**
- 全机：**零 NVLink**，纯 PCIe
- GPU1 A800 与 GPU4/5 A4000：**不同代 GPU**

### 4.2 原因

NCCL 在没有 NVLink 时默认尝试 **GPUDirect P2P over PCIe**。但在以下条件叠加时，
P2P 往往不被驱动支持：

- 跨代 GPU（A800 ↔ A4000，Ampere 不同子型号）
- 跨 PCIe bridge（PXB 拓扑）

关键：NCCL 在 P2P 不可用时**不会报错退化**，而是直接 hang（busy-poll 等待一个
永远完成不了的传输）。这就是为什么看到"GPU 100% util 但无进展"——NCCL kernel
在无限自旋等待。

> **为什么这么难查**：NCCL 的 hang 行为完全静默，没有错误日志，只有 GPU util 100%
> 这一个外部表现。如果不用 `NCCL_DEBUG=INFO` 或做最小化复现，很难和编排 bug 区分开。

---

## 5. 解决

```bash
export NCCL_P2P_DISABLE=1
```

关掉 GPU 直连 P2P，强制 NCCL 走 **SHM（共享内存）或 CPU host 中转**。
代价是带宽低一些，但对于权重同步（几 GB、低频）完全够用。

### 验证结果

| 测试 | 环境 | 结果 |
|---|---|---|
| 1MB broadcast，world=3 | 默认 NCCL | 卡死，30s 无结果 |
| 1MB broadcast，world=3 | `NCCL_P2P_DISABLE=1` | 通过，0.002s |
| 512MB broadcast，world=3 | `NCCL_P2P_DISABLE=1` | 通过，0.115s |
| 完整 GRPO rollout=2，1 step | `NCCL_P2P_DISABLE=1` | 通过，`update_weights=4.46s` |
| 完整 GRPO rollout=1/2/3/4，各 5 step | `NCCL_P2P_DISABLE=1` | 全部通过 |

实测全参 1.7B 权重同步带宽：**3.96 GB/s**（A800 → A4000，PCIe Gen3 SHM 路径），
单次 3.4GB 约 4.5s。

### 当前 runner 默认配置（已固化）

```bash
NCCL_P2P_DISABLE=1       # 关闭 P2P，走 SHM/host 中转
NCCL_REBUILD_GROUP=1     # 每次 finalize 后销毁 NCCL group，防 communicator 复用冲突
```

---

## 6. 换机器时的检查流程

**第一步：看拓扑**

```bash
nvidia-smi topo -m
```

如果训练侧 GPU 和 rollout 侧 GPU 之间是 PXB 或 PHB（跨 bridge），
且不同代型号，就有 P2P hang 的风险。

**第二步：先跑最小 broadcast**

在跑任何 GRPO 之前，用最小脚本验证 broadcast 是否通过：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=<trainer_gpu>,<rollout_gpu1>,<rollout_gpu2> \
python RL/nccl_broadcast_smoke.py --world-size 3 --mb 1
```

- 通过（有结果打印）→ 正常，可以不加 `NCCL_P2P_DISABLE`
- 卡死（30s 无输出）→ 加 `NCCL_P2P_DISABLE=1` 再试

**第三步：如果 P2P_DISABLE 仍然卡**

```bash
# 开 NCCL 详细日志，看它选了什么传输路径
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,P2P,NET
```

关注日志里的 `via P2P` / `via SHM` / `via NET`，
以及 A800↔A4000 通道的 `Connected` 是否出现。极端情况下可以再叠加：

```bash
export NCCL_SHM_DISABLE=1   # 连 SHM 也关，逼它走 NET（localhost socket）
```

---

## 7. 关键文件

| 文件 | 用途 |
|---|---|
| `RL/nccl_broadcast_smoke.py` | 最小化复现 / 验证脚本 |
| `RL/run_grpo_fully_async_split_a800_ref3_a4000.sh` | 当前 runner，`NCCL_P2P_DISABLE=1` 已写入默认值 |
| `verl/checkpoint_engine/nccl_checkpoint_engine.py` | 加了 `VERL_NCCL_TRACE_WEIGHTS` trace 开关，用于死锁定位 |
| `verl/workers/engine_workers.py` | checkpoint-manager 各阶段计时日志 |
| `docs/rl_nccl_weightsync_deadlock_20260613.md` | 原始排查记录（含完整 trace 日志片段） |
