# 异步 GRPO 环境升级:verl 0.8.0.dev × vllm 调研与配方 · 2026-06-13

> 目标:用主仓库 verl **0.8.0.dev**(支持 async server rollout,便于后续改"超快异步 GRPO")。
> 但它与本机原环境(vllm 0.8.5)不兼容。本文记录**为什么不兼容、硬约束、验证过程、可用配方**。
> ⚠️ 每个结论都经实测,供报告/交接直接引用。

## 1. 不兼容的根因(verl 自己的问题)

- verl 0.8.0.dev `setup.py` 声明 `vllm>=0.8.5,<=0.12.0`——**0.8.5 在范围内,我们合规**。
- 但它的 async server 代码实际用了**更新的 vllm API**,0.8.5 里没有:
  1. `from vllm.entrypoints.cli.serve import run_headless` → 0.8.5 无此符号(0.9 才加)
  2. serve CLI 参数 `--logprobs-mode` → **0.10 才加**(0.9.2 仍只有 `--max-logprobs`)
- ⇒ **verl dev 分支代码跑到了它自己声明的依赖下界前面**;`>=0.8.5` 是过时的谎报。
  真正能跑 async server 的最低 vllm 是 **0.10.x**。

## 2. 硬约束:GPU 驱动(不可改)

- Driver **550.163.01 / CUDA 12.4**(共享借用机,**不能升驱动**)。
- 新 vllm 常用 **cu128**(CUDA 12.8)wheel,**R550 跑不了 cu128**。
- ✅ 实测:**cu126**(CUDA 12.6)wheel **能**在 R550 上跑(minor-version 兼容)。
  - torch 2.7.1+cu126:`torch.cuda.is_available()=True`,GPU bf16 matmul 通过。

## 3. 验证过程(临时探针 venv `/tmp/probe_cu126`,不碰主 .venv)

| 步骤 | 结果 |
|---|---|
| torch 2.7.1+cu126 装 + GPU matmul | ✅ 驱动兼容 |
| vllm 0.9.2 | 带 torch 2.7.0+**cu126**;有 run_headless;**但无 `--logprobs-mode`**(不够) |
| vllm 0.9.2 + transformers **5.12** | ❌ 崩:`'aimv2' is already used`(transformers 5.x 与 vllm 配置注册冲突) |
| vllm 0.9.2 + transformers 4.51.3 | ✅ 加载 S3-R1、generate 正确(`1+1→2`) |
| **vllm 0.10.1.1** + transformers 4.57 | ✅ torch 2.7.1+**cu126**;**run_headless ✓ + `--logprobs-mode` ✓**;generate 正确 |

## 4. 可用配方(经实测,R550/12.4 驱动)

```
torch==2.7.1            (+cu126,index https://download.pytorch.org/whl/cu126)
torchaudio==2.7.1  torchvision==0.22.1
vllm==0.10.1.1          # run_headless + --logprobs-mode 都有
transformers>=4.55,<5   # 实测解析到 4.57.6;不可上 5.x(aimv2 冲突);vllm0.10 要求 >=4.55
xformers==0.0.31  xgrammar==0.1.21  triton==3.3.1
```
- flash-attn / flashinfer:探针**未装也能 generate**(vllm 0.10 自带注意力后端)。
  训练侧 actor 前向若需 FA,再补 torch2.7 版的 flash-attn(原 `+cu12torch2.6` 的不兼容,要换)。

## 5. 升级方式与回滚

- **方式**:in-place 升级主 `.venv`(用户决定不隔离,就用这一份)。
- **保命快照**:升级前 `uv pip freeze > requirements_backup_20260613.txt`(207 包,torch2.6/vllm0.8.5 的可用态)。
  若评测/SFT 被搞坏,`uv pip install -r requirements_backup_20260613.txt` 还原。
- **风险**:torch 2.6→2.7.1 会使原 flash-attn(`+cu12torch2.6`)/flashinfer(`cu124torch2.6`)ABI 失配,需一并处理;
  transformers 4.51→4.57 对自定义数字切分 tokenizer 影响需回归(AutoTokenizer 加载验证)。
- **影响面**:`.venv` 同时被评测(final_eval)/SFT 用;升级后需回归这两条。

## 6. 现状

- [x] 配方验证完成(探针 venv)
- [x] 主 `.venv` in-place 升级
- [x] verl 0.8.0.dev async server smoke 跑通(1-step 闭环)
- [ ] 回归 final_eval / SFT 在新栈下可用

## 7. 2026-06-13 async GRPO smoke 结果

命令:

```bash
bash RL/run_grpo_async_smoke.sh 1 1
```

环境:

```text
torch 2.7.1+cu126
vllm 0.10.1.1
ray 2.55.1
verl 0.8.0.dev
transformers 4.57.6
```

配置摘要:

| 项 | 值 |
|---|---|
| GPU | card1, A800 80G, colocated actor/ref/vLLM async server |
| 起点 | `/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf` |
| 数据 | `/data/zilu/data_unified_v2/rl_smoke` |
| 训练方式 | GRPO, LoRA rank32, group n=8, batch64, response<=1024 |
| rollout | `actor_rollout_ref.rollout.mode=async`, vLLM HTTP server |
| TensorBoard | `logs/tb_grpo_async_smoke/events.out.tfevents.1781334586.ubuntu-ESC8000-G4.1706775.0` |

结论:async server 能完整启动并跑通 1 个训练 step。实际通过了:

- 数据集加载:train 3000 / val 200。
- actor FSDP 初始化。
- vLLM async HTTP server 启动、模型加载、CUDA graph capture。
- actor -> vLLM LoRA/权重同步:`update_weights done, time cost:0.76s`。
- 初始 validation generation 与 reward 统计。
- step=1 训练 metrics 打印完成,命令退出码为 0。

单独解析本次 event 文件得到的 1-step timing:

| 阶段 | 时间 |
|---|---:|
| step | 91.97s |
| gen | 29.15s |
| old_log_prob | 14.31s |
| ref | 10.00s |
| adv | 0.02s |
| update_actor | 35.07s |
| update_weights | 3.00s |

显存/吞吐:

| 指标 | 值 |
|---|---:|
| actor max_memory_allocated | 27.23GB |
| actor max_memory_reserved | 28.55GB |
| actor CPU memory used | 69.37GB |
| total_num_tokens | 281,855 |
| throughput | 3,064.64 tokens/s |

validation step0:

| 指标 | 值 |
|---|---:|
| reward mean@1 | 0.0381 |
| correct mean@1 | 0.0300 |
| has_format mean@1 | 0.7650 |
| think_len_tokens mean@1 | 218.12 |
| think_len_bonus mean@1 | 0.0721 |
| repeat_penalty mean@1 | 0.1405 |

注意事项:

- `logs/tb_grpo_async_smoke` 目录里混有旧 event 文件;做正式汇总时不要直接聚合整个目录,要按本次 event 或新 run dir 解析。
- 训练结束阶段 Ray worker 打过一次 traceback:`DataLoader worker ... is killed by signal: Killed.` 但训练 step 已完成、metrics 已写入、外层命令退出码为 0、GPU 已释放。它目前更像收尾/cleanup 噪声,但正式长跑前需要用 5-20 step 再确认是否复现或累积成真实失败。
