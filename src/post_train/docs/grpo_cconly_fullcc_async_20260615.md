# GRPO Compute-Cot Only + Full CC-Reserved Eval Run

Launched: 2026-06-15 03:37 Asia/Shanghai

## Run

- tmux session: `grpo_cconly_fullcc`
- launcher: `src/post_train/RL/launch_grpo_cconly_fullcc_async_20260615.sh`
- run name: `grpo_cconly_fullcc_async_refsvc_20260615_0339_rollout4_steps300`
- base checkpoint: `/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf`
- output dir: `src/post_train/logs/ckpt_grpo_cconly_fullcc_async_refsvc_20260615_0339_rollout4_steps300`
- main log: `src/post_train/logs/grpo_cconly_fullcc_async_refsvc_20260615_0339_rollout4_steps300.log`
- gpu log: `src/post_train/logs/grpo_cconly_fullcc_async_refsvc_20260615_0339_rollout4_steps300_gpu.csv`
- event log: `src/post_train/logs/grpo_cconly_fullcc_async_refsvc_20260615_0339_rollout4_steps300_events.jsonl`
- process eval dumps: `src/post_train/logs/grpo_cconly_fullcc_async_refsvc_20260615_0339_rollout4_steps300_val_dumps`

## Data

- train parquet: `src/post_train/data/rl_compute_cot_only/train.parquet`
- train rows: 392,553
- selection: filtered from `/data/zilu/data_unified_v2/parquet/train_rl_s4clean_fix.parquet` with `data_source` starting `compute_cot`
- val parquet: `src/post_train/data/rl_compute_cot_only/cc_reserved_val.parquet`
- val rows: 37,477
- selection: full `Compute_Cot/data/clean/test/id_test.jsonl`, converted to GRPO parquet schema

## Key Config

- split: actor/update card1 A800, ref-service card3 A4000, rollout cards 4-7 A4000
- async ref service: enabled
- train versions: `NSTEP=300`
- process eval: `TEST_FREQ=10`
- eval before train: disabled
- trainer validation path: rollouter-side validation (`USE_TRAINER_DO_VALIDATE=False`)
- save frequency: every 50 versions
- staleness threshold: `2.0`
- port range: `[30000,45000]`
- GRPO batch: `PPO_MINI_BATCH_SIZE=32`, `REQUIRE_BATCHES=2`
- rollout: `n=8`, temperature `1.0`
- RL stability overrides: `NORM_ADV_BY_STD=False`, `KL_LOSS_COEF=0.005`, `REWARD_FORMAT_BONUS=0.05`, `REWARD_THINK_LEN_MAX_BONUS=0`

## Startup Check

Observed after startup:

- `ray::FullyAsyncTrainer.fit` running
- `ray::FullyAsyncRollouter.fit` running
- 4 `ray::vLLMHttpServer` processes running
- ref worker and 16 reward workers running
- GPU memory/utilization:
  - card1 A800: ~8.8 GB
  - card3 A4000: ~11.0 GB
  - cards4-7 A4000: ~11.6-12.2 GB, ~78-83% util

## Monitor

```bash
tmux attach -t grpo_cconly_fullcc
tail -f /data/zilu/QseekLLM/src/post_train/logs/grpo_cconly_fullcc_async_refsvc_20260615_0339_rollout4_steps300.log
nvidia-smi -i 1,3,4,5,6,7
```
