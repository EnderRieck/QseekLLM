# SFT —— 阶段1-2 基础 SFT（hybrid_sft_rl_design.md）

前期纯 SFT，广混大打基础：①算术执行力(compute_cot worked) + ②题目理解(各类数学题) + ③通用防退化。
RL(GRPO) 到阶段3 才开。训练在 **A800(card1)**，过程评测异步在 **A4000(card2)**，互不阻塞。

---

## 启动（两个 tmux）

```bash
# tmux 1 — 训练（card1 A800）
cd /data/zilu/QseekLLM/src/post_train
bash SFT/run_foundation_sft.sh /data/zilu/fastrl/checkpoints/sft_foundation

# tmux 2 — 异步过程评测（自动在 A4000 2-7 里挑空闲卡，监视 checkpoint）
cd /data/zilu/QseekLLM/src/post_train
CUDA_DEVICE_ORDER=PCI_BUS_ID \
  .venv/bin/python -m eval.async_eval \
  --ckpt-dir /data/zilu/fastrl/checkpoints/sft_foundation --watch
```
- **自动选卡**：每次评测前在候选 `2,3,4,5,6,7` 里挑最空闲(≥8G)的卡；全忙则等。不要设 `CUDA_VISIBLE_DEVICES`（要让进程看到所有卡）。
- 可调：`--gpu-candidates "4,5,6,7"` / `--min-free-gb 10` / `--device cuda:3`(强制某卡)。
- tmux：`Ctrl-b d` 脱离，`tmux a -t sft` 回看；上滑 `Ctrl-b [` 后方向键/PageUp，`q` 退出。
- 改超参：脚本后追加覆盖，如 `... optim.lr=2e-5`。

---

## 最终配置（已写死在 run_foundation_sft.sh）

| 项 | 值 | 说明 |
|---|---|---|
| base | `/data/zilu/fastrl/checkpoints/qseek_digitsplit_base` | 1.7B Llama，**数字切分** tokenizer，16k |
| 数据 | `/data/zilu/data_unified/parquet/train_sft_foundation_8k.parquet` | **118.9万**，≤8192 token，重采"易重难轻" |
| max_length / token预算 | 8192 / 24576 | dynamic bsz 按 token 预算打包 |
| 加速 | **liger 0.5.10** + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | MFU ~0.5（基线 0.14） |
| batch / lr / 优化器 | 256 / 1e-5 / AdamW 全量微调(fp32主权重, bf16计算) | |
| epoch / save_freq | **2**（9288步）/ 500 | 每 500 步存档→自动评测 |
| 显存峰值 | reserved ~58G / 80G（余量 21G） | 安全 |
| 预计时长 | 2 epoch ~10-15h | MFU~0.5 |

数据配比（重采后）：①compute_cot 算术骨干为主 + 易题(gsm8k/orca/calc-ape) + 竞赛难题降权陪练(numina/openr1/bespoke) + 通用~24%(中英)。详见 `../data_pipeline/README.md`。

---

## 过程跟踪（card2 异步 eval）

每 500 步存档后，`eval/async_eval.py` 自动在 **2018 题 held-out**（训练外、无泄漏，8 来源全谱）上生成+判分，产出到 `…/sft_foundation/eval_dumps/`：
- `metrics.jsonl` —— 每步 acc / format_rate / 各 source×难度准确率
- `step_N/heldout.jsonl` —— **完整输入输出 dump**（追溯命根，对照 v3 经验）
- `tb/` —— tensorboard

**看什么**：`train/loss` 下降、`grad_norm` 别爆(>100)；eval 早期 acc 低正常，重点看 **format_rate（学会 `<think>…#### \boxed{}` 没）** + easy 源(gsm8k/cmath/compute_cot)准确率起色；末尾还在涨可加 epoch，plateau 则停。

---

## 踩坑记录（2026-06-09，供日后排查）

1. **Liger 版本错配**：Verl 依赖里 liger/transformers 都没 pin，pip 抓了最新 liger 0.8.0（要 transformers≥4.52），但环境是 4.51.3（vllm 等锁的）→ 运行时 `use_liger=True` 报版本错。
   **解**：`uv pip install --python .venv/bin/python "liger-kernel<0.6"` 降到 **0.5.10**（要求 transformers≥4.44.2，兼容），实测 Verl 集成正常、swiglu 融合核生效。**绝不升 transformers**（会破坏 Verl）。
2. **token 预算过大 OOM**：`max_token_len_per_gpu=49152` + max_length=8192 → 反传时峰值 68G+ 后 OOM。
   **解**：降到 **24576**（reserved 58G，余量 21G）+ 开 `expandable_segments` 抗碎片。32768 也能跑但 reserved 74G 太顶，弃用。
3. **OOM 后僵尸进程占显存**：torchrun worker OOM 崩溃后，孤儿进程仍占 ~69G（util 0%、状态 Ssl、父 torchrun 已死）→ 后续任何运行瞬间 OOM。
   **排查**：`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 1`；util 0% 但占显存 = 僵尸。
   **解**：`kill -9 <pid>` 清理。
4. **数据 max_length 截断风险**：openthoughts3 的 R1 长链 p50=16751 token（超模型 16k 上限），`truncation=right` 会截掉答案。
   **解**：基础池预过滤 ≤8192 token（丢 37990，几乎全是 openthoughts3）；它们留全量池给阶段2-3。

---

## 关键路径/接口
- 训练脚本：`SFT/run_foundation_sft.sh`
- 异步评测：`eval/async_eval.py`（`--step N` 只评某档；`--watch` 持续）
- held-out 重建：`python -m eval.build_heldout`
- 数据管线：`data_pipeline/`（build/build_general/reweight_sft/to_verl_parquet，见其 README）
- 产出：`/data/zilu/fastrl/checkpoints/sft_foundation/`（checkpoint + eval_dumps + tb/wandb）
- 后端：Verl + FSDP + torchrun（为衔接阶段3-4 的 GRPO，全程一套栈）
