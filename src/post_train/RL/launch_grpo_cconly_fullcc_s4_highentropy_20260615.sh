#!/bin/bash
set -euo pipefail

cd /data/zilu/QseekLLM/src/post_train

export RUN_GROUP=grpo_cconly_fullcc_s4_temp13_async_refsvc_20260615
export TRAIN_FILES=/data/zilu/QseekLLM/src/post_train/data/rl_compute_cot_only/train.parquet
export VAL_FILES=/data/zilu/QseekLLM/src/post_train/data/rl_compute_cot_only/cc_reserved_val.parquet
export CKPT=/data/zilu/fastrl/checkpoints/sft_s4_anneal/global_step_1140_HFFIX

export TEST_FREQ=10
export VAL_BEFORE_TRAIN=False
export USE_TRAINER_DO_VALIDATE=False
export SAVE_FREQ=50
export NSTEP=300

export ROLLOUT_TEMPERATURE=1.3
export STALENESS_THRESHOLD=2.0
export MASTER_PORT_RANGE='[30000,45000]'
export LOG_VAL_GENERATIONS=20

exec bash RL/run_grpo_fully_async_split_a800_ref3_a4000.sh 4 300
