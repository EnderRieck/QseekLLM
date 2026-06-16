# GRPO Compute-Cot Only + Full CC-Reserved Eval, S4 Start, High-Entropy Sampling

Launched: 2026-06-15 04:24 Asia/Shanghai

## Previous Run Stopped

Stopped tmux session `grpo_cconly_fullcc` after early eval degradation. GPUs were released before launching this run.

## Run

- tmux session: `grpo_cconly_s4_temp13`
- launcher: `src/post_train/RL/launch_grpo_cconly_fullcc_s4_highentropy_20260615.sh`
- run name: `grpo_cconly_fullcc_s4_temp13_async_refsvc_20260615_rollout4_steps300`
- base checkpoint: `/data/zilu/fastrl/checkpoints/sft_s4_anneal/global_step_1140_HFFIX`
- output dir: `src/post_train/logs/ckpt_grpo_cconly_fullcc_s4_temp13_async_refsvc_20260615_rollout4_steps300`
- main log: `src/post_train/logs/grpo_cconly_fullcc_s4_temp13_async_refsvc_20260615_rollout4_steps300.log`
- eval dumps: `src/post_train/logs/grpo_cconly_fullcc_s4_temp13_async_refsvc_20260615_rollout4_steps300_val_dumps`

## Data

- train parquet: `src/post_train/data/rl_compute_cot_only/train.parquet`
- train rows: 392,553, only `compute_cot`
- val parquet: `src/post_train/data/rl_compute_cot_only/cc_reserved_val.parquet`
- val rows: 37,477, full CC-reserved

## Key Differences

- start checkpoint changed from S3R1 SFT to S4-1140
- rollout sampling temperature changed from `1.0` to `1.3`
- top-p/top-k remain fully open through base config: `top_p=1`, `top_k=-1`

## Other Config

- actor/update: card1 A800
- ref-service: card3 A4000
- rollout: cards4-7 A4000
- async ref service: enabled
- process eval: every 10 versions on full CC-reserved
- save frequency: every 50 versions
- train versions: 300
- GRPO batch: `PPO_MINI_BATCH_SIZE=32`, `REQUIRE_BATCHES=2`
- rollout samples per prompt: `n=8`
- `NORM_ADV_BY_STD=False`
- `KL_LOSS_COEF=0.005`
- `REWARD_FORMAT_BONUS=0.05`
- `REWARD_THINK_LEN_MAX_BONUS=0.0`

## Monitor

```bash
tmux attach -t grpo_cconly_s4_temp13
tail -f /data/zilu/QseekLLM/src/post_train/logs/grpo_cconly_fullcc_s4_temp13_async_refsvc_20260615_rollout4_steps300.log
nvidia-smi -i 1,3,4,5,6,7
```
