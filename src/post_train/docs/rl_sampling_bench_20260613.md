# GRPO 采样吞吐基准 · 方法与数据 · 2026-06-13

> 目的:为"超快 GRPO"的多卡/异步优化提供**实测地基**——标定单卡采样吞吐、
> 量化 verl 同步循环的"非 decode 开销"、估算多卡/异步的真实收益。
> ⚠️ 本文每个数字都标清**怎么测的、测的是什么**,供后续报告直接引用。

## 0. 被测对象与不变量

- **模型**:S3-R1(`sft_s3r1/global_step_3874_hf`,1.6B,数字切分 tokenizer)。
- **工作量(复刻 GRPO rollout 一步)**:64 prompt × n=8 = **512 条序列**,max_tokens=1024,
  temperature=1.0,top_p=1.0。prompt 取自 `rl_smoke/train.parquet` 前 64 条(big-math 收割带,prompt 均长 ~116 tok)。
- **vLLM**:0.8.5.post1,bf16,enforce_eager=False(开 cudagraph),max_model_len=2048。
- 机器:8 卡无 NVLink,PCIe Gen3 x16。card1=A800-80G(HBM2e ~2039 GB/s),card0/2-7=A4000-16G(GDDR6 ~448 GB/s)。

## 1. 两种"采样耗时"的口径(务必分清)

| 口径 | 含义 | 用途 |
|---|---|---|
| **A. verl 内 `timing_s/gen`** | GRPO 训练循环里每步采样段墙钟,**含** vLLM wake、权重刷入引擎、KV 重建、reshard 等非 decode 开销 | 反映**真实训练步**里采样占用 |
| **B. 干净 vLLM bench(`RL/bench_sampling.py`)** | load 一次→generate 一次的**纯 decode** 墙钟,不含每步 wake/重建 | 反映**硬件 decode 上限**,标定卡间倍率 |

两者**不可直接比**。A−B 的差 ≈ verl 同步循环的"非 decode 开销"(可优化项)。

## 2. 实测数据

### 2.1 单卡纯 decode(口径 B,`bench_sampling.py`,mem_util=0.85)

| 卡 | gen 墙钟 | 输出 token | 吞吐 | avg_resp | 备注 |
|---|---|---|---|---|---|
| **A4000 ×1**(card0) | **65.2s** | 226424 | **3473 tok/s** | 442 | load 98.7s(一次性,不计入) |
| A800 ×1(card1) | _待_ | _待_ | _待_ | | smoke 占用中,跑完补测 |

> 测法:`CUDA_VISIBLE_DEVICES=<card> python RL/bench_sampling.py 64 8 1024 0.85`。
> 输出 JSON 含 gen_s / tok_per_s / avg_resp_len。avg_resp 442 略高于 verl 内 417——因 bench 用初始 S3-R1、未经 RL 更新,且 mem_util 高→并发足、无截断压力。

### 2.2 verl 同步循环内采样(口径 A,从 `logs/tb_grpo_smoke` 取,10 步均值)

| 指标 | 值 | 说明 |
|---|---|---|
| `timing_s/gen` | **103.9s** | 单 A800,mem_util=0.5,512 条 |
| throughput(整步) | 1688 tok/s | 含训练等所有阶段 |
| 采样 per-token | 0.487 ms/tok | |

### 2.3 关键对照(口径 A vs B,初步)

单 A4000 纯 decode(65s)< verl 内单 A800 gen(104s)。**不是 A4000 比 A800 快**,而是:
- B 是纯 decode;A 含每步 wake/KV 重建/reshard
- B mem_util 0.85(KV 大、并发高)vs A 0.5
→ **verl 同步采样存在 ~30-40s/步 的非 decode 开销**,是"超快 GRPO"的可砍项之一。
(待 §2.1 补齐 A800 纯 decode 后,才能给出干净的 A800 vs A4000 硬件倍率。)

### 2.4 多卡 A4000 data-parallel 扩展曲线(口径 B,采固定 512 条墙钟)

测法:把 64 prompt 拆到 N 张卡(每卡 64/N prompt × n=8),N 个进程**同时**起,
各卡独立 vLLM、解码期间不通信;**DP 墙钟 = max(各卡 gen_s)**。mem_util=0.85,max_tok=1024。

| 卡数 N | 每卡条数 | DP 墙钟 | 加速比 | **每卡吞吐** |
|---|---|---|---|---|
| 1 | 512 | 65.2s | 1.0× | 3473 tok/s |
| 2 | 256 | 33.3s | **2.0×** | 3460 tok/s |
| 4 | 128 | 22.7s | 2.9× | 2680 tok/s |
| 7 | ~73 | 19.5s | 3.3× | ~1990 tok/s |

**为什么亚线性(不是 N×)——解码是显存带宽受限,有固定的"权重读取税"**:

> 单卡每解码步耗时 ≈ **T_权重(固定)** + b × c
> - T_权重 = 每步把整个模型(3.4GB)从显存读一遍,**与 b 无关,每卡每步都付**
> - b = 该卡并发条数,c = 每条边际成本
>
> 固定 512 条拆到 N 卡(每卡 b=512/N):
> **加速比 = (T_权重 + 512c) / (T_权重 + 512c/N)**
> - c 主导(batch 大)→ 加速比→N(线性);T_权重 主导(batch 小)→ 加速比→1(堆卡无效)

**实测佐证**:每卡吞吐在 **≥256 条/卡时饱和**(3460≈3473,几乎不掉),<256 才掉(128→2680, 73→1990)。
→ 512 条的甜点是 **~2 卡(近线性 2.0×)**;4/7 卡每卡欠载,边际收益递减。

**澄清不是原因的**:① 非 PCIe/通信(DP 解码各卡独立、无通信,无 NVLink 无妨);② 非 A4000 弱(同卡满载3473/半载2680,纯 batch 效应)。

**对"超快 GRPO"的启示**:要喂饱多卡,**global batch 必须随卡数同涨**(用满 7 卡需 ≥~1800 条/步)。
否则堆卡浪费——快 GRPO = 多卡 **+ 大 batch**(+ 异步重叠),三者缺一不可。

## 3. 待补测(本轮计划)

1. **A800 单卡纯 decode**(smoke 跑完即测)→ 得 A800 vs A4000 硬件倍率(预期 A800 ~1.5-3× 单 A4000,decode 带宽受限)。
2. **多卡 A4000 data-parallel**(2/4/7 卡)→ 实测聚合吞吐与扩展性(是否近线性,还是被 PCIe/调度拖累)。
3. 由实测倍率重算 §异步+split GRPO 的 step 提速(替换之前纸面推算的 ~2.3×)。

## 4. 复现接口

- 基准脚本:`RL/bench_sampling.py`(参数:n_prompt n maxtok memutil)
- 数据:`/data/zilu/data_unified_v2/rl_smoke/train.parquet`
- verl 内时间:`python RL/parse_timing.py logs/tb_grpo_smoke`
- 原始时间戳:`logs/grpo_smoke_timing.txt`
