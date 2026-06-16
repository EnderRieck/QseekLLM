# GRPO 同步 smoke · 时间开销分析 · 2026-06-13

> CLAUDE.md 阶段4 要求:同步版本 smoke,跟踪**采样/训练/权重同步**各部分耗时,
> 判断 GRPO 是否生效、为后续异步配置与瓶颈分析打底。本文记录第一次成功 smoke 的实测。

## 1. 背景与配置

- **起点模型**:S3-R1(`sft_s3r1/global_step_3874_hf`,1.6B)——经能力分析其 pass@8 最高、
  RL 收割带最宽(见 `model_capability_analysis_20260612.md`),是最佳 RL 起点。
- **数据**:`rl_smoke`(3000 训 / 200 val)。取自 `train_rl_s4clean.jsonl` 中
  **big-math 收割带**(solve_rate∈[0.15,0.85],组内有对有错→GRPO 有梯度),全 `math_verify` 可判分。
- **判分**:`RL/reward_verl.py` 复用评测同口径判分器(`data_pipeline.reward`),
  reward = correct(1/0) + 0.1·format + 有界 thinking 长度奖励 - 循环重复惩罚。
  当前默认 thinking 长度从 64 到 512 token-ish 线性给分,最多 +0.2,重复惩罚最多 -0.4；额外记录
  `think_len_tokens` / `think_len_bonus` / `repeat_penalty` 方便监控。
- **算法/规模**:GRPO,LoRA rank32,batch 64,group n=8(512 rollout/step),
  max_prompt 1024 / max_response 1024,lr 1e-6,KL low_var coef 0.001。
- **部署**:**colocated hybrid engine,单张 A800(80G)**。同步 SPMD rollout。
  vLLM gpu_mem_util 0.5,ref param_offload。

## 2. 环境踩坑(重要,交接必读)

主仓库 `verl/` 是 **0.8.0.dev**,它**移除了同步 SPMD rollout、只剩 async server**,
且 async server 依赖 vLLM≥0.9 的 API(`run_headless`、`--logprobs-mode`),
**与本机 vLLM 0.8.5 不兼容**(实测连续报 `ImportError: run_headless`、
`unrecognized arguments: --logprobs-mode`)。

**解法**:用隔离 worktree **verl v0.5.0**(`verl_v050/`,依赖声明正好 `vllm 0.7.3~0.8.5 + torch 2.6`,
保留 `vllm_rollout_spmd.py` 同步 rollout),经 `PYTHONPATH` 覆盖,**不动主 verl/**
(主 verl/ 保留了 SFT 的 `multiturn_sft_dataset.py` 改动)。启动脚本 `RL/run_grpo_smoke.sh` 已固化。

> 后续若要用主 verl 0.8.0.dev 的异步 server,需先把 vLLM 升到 0.9+(会牵连 torch/CUDA 与评测管线,
> 需单独评估)。smoke/同步阶段统一用 v0.5.0。

## 3. 单步时间拆解(实测,161s/step)

单 A800,batch 64,n=8(512 条 rollout),resp≤1024:

| 阶段 | 耗时 | 占整步 | 含义 |
|---|---|---|---|
| **gen(采样)** | 100.3s | **62.2%** | vLLM 生成 512 条 → **头号瓶颈** |
| update_actor(训练) | 30.7s | 19.1% | actor fwd+bwd+optim |
| reward(判分) | 12.4s | 7.7% | 判分器跑 512 条(首步含冷启,次步降到 4.7s) |
| old_log_prob | 10.1s | 6.3% | actor 重算 rollout logprob |
| ref(KL) | 7.6s | 4.7% | ref 模型 logprob |
| **reshard(权重同步)** | 1.6s | **1.0%** | FSDP↔vLLM sleep/wake + 权重刷入 |
| adv | 0.06s | ~0% | GRPO 组内优势 |
| **step 合计** | **161.2s** | 100% | |

## 4. 结论(对照 CLAUDE.md 瓶颈预判)

1. **采样是绝对瓶颈(62%)**,训练仅 19%。模型小,训练不吃力;**所有效率优化都应指向 rollout**
   (多卡并行采样、缩短 max_response、vLLM 调参/更高并发、过滤超长生成)。

2. **权重同步几乎免费(1.0%,1.6s)**。印证 CLAUDE.md "LoRA-only 同步 0.1–0.3s" 的方向:
   colocated + LoRA 下同步不是瓶颈。**推论:把训练/采样拆成 A800/A4000 split placement 的优化
   收益存疑**——同步本就便宜,拆开反而引入跨卡(PCIe 5GB/s)权重传输与调度空泡。
   除非目的是"用 A4000 多卡把 62% 的采样并行掉",那才是 split 的真正价值点。

3. **采样 100s 是在单卡 vLLM(mem_util 0.5)上跑 512 条 resp≤1024**。
   异步/多卡可把这块并行:7 张 A4000 理论上能把采样墙钟摊薄数倍 → 这是下一步异步版本要验证的核心。

## 4b. 细致拆解(per-token / MFU / 长度 / 优化信号,4 步均值)

**① 每-token 效率（瓶颈本质）**

| 阶段 | ms/token | 相对 gen |
|---|---|---|
| 采样 gen | **0.511** | 1× |
| 训练 update_actor | 0.119 | 快 4.3× |
| ref | 0.029 | 快 17× |

采样不仅"量大",**单 token 就比训练慢 4.3×**——自回归解码逐 token、内存带宽受限,
训练是整序列并行前向。这是 rollout 占 62% 的根因:**优化采样的杠杆远大于优化训练**。

**② 算力利用 / 吞吐**:MFU(actor) **34.3%**,throughput **1628 token/s**,单步 ~26.7 万 token。
训练侧 MFU 不算低,但采样阶段(低 FLOP 利用)把整体有效利用率拉下来。

**③ response 长度**:mean ~407 token,**10% 撞 1024 上限(clip_ratio 0.1)**,min ~45,prompt mean ~115。
→ 超长的 10% 不成比例拖慢采样(批内最慢序列 gate 整批)。**砍 max_response / 加长度惩罚可直接削 62% 瓶颈**。
采样耗时随步上涨(100→111s):策略更新后生成略变长 + KV cache 效应。

**④ GRPO 优化信号(解释 reward 为何平)**

| 信号 | 值 | 解读 |
|---|---|---|
| grad_norm | **0.013** | 梯度极小 |
| lr | 1e-6 | 保守 |
| kl_loss | 0.0004 | 策略几乎没离开 ref |
| pg_clipfrac | 0.0001 | 无裁剪,健康(on-policy) |
| entropy | 1.03→0.89 | 略收敛 |
| advantages/mean | -0.18 | GRPO 组内归一化,正常 |

**grad_norm 0.013 × lr 1e-6 = 有效更新量微乎其微**,模型基本没动 → reward 平。
对 smoke(验管线)正常;要真学起来需调大 lr,且收割带里全错组(优势=0)仍偏多、信号偏弱。

**⑤ 显存**:峰值 allocated **56.5G**(reserved 报 96.7G,疑 verl 把 vLLM KV 重复计,以 allocated 为准)。
A800 80G 有富余 → 可加大 batch/n 提采样并行度。

## 5. GRPO 是否生效(reward 趋势)

- **基线(step0 验证)**:reward 0.118 / correct **4.5%** / format 0.73(temp=1.0 采样,收割带题偏难)。
- **训练 rollout reward**:step1 0.131 → step2 0.128(_待更多步_)。
- **状态**:smoke 在跑(20 step,~54min)。reward 是否爬升需 ≥8–10 步趋势,跑完追加本节 + 末次验证对比。

## 6. 关键接口(交接用)

- 启动:`bash RL/run_grpo_smoke.sh [card=1] [n_step=20]`
- 奖励函数:`RL/reward_verl.py`(custom_reward_function,复用评测判分器)
- 数据:`/data/zilu/data_unified_v2/rl_smoke/{train,val}.parquet`(构建逻辑见本目录脚本注释)
- 时间/趋势解析:`python RL/parse_timing.py logs/tb_grpo_smoke`
- 日志:`logs/grpo_smoke.log`;tb:`logs/tb_grpo_smoke/`;时间戳:`logs/grpo_smoke_timing.txt`
- 隔离 verl:`verl_v050/`(v0.5.0 worktree,RL 专用)
