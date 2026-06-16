#!/bin/bash
# ============================================================================
# GRPO sync split smoke for efficiency measurement.
#   actor/ref/logprob/update: one A800 by default, physical GPU 1
#   rollout/vLLM sampling:   one A4000 by default, physical GPU 0
#   tuning mode: full-parameter, lora_rank=0
#
# This split uses main_ppo_sync's synchronous loop. Rollout is a standalone
# vLLM server and weights are synced through the checkpoint engine.
#
# Usage:
#   bash RL/run_grpo_sync_split_fullparam_a800_a4000.sh [train_card] [rollout_card] [n_step]
# ============================================================================
set -xeuo pipefail
cd "$(dirname "$0")/.."

TRAIN_CARD=${1:-1}
ROLLOUT_CARD=${2:-0}
NSTEP=${3:-3}

CKPT=/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf
DATA=/data/zilu/data_unified_v2/rl_smoke
RUN_NAME=grpo_sync_split_fullparam_a800_a4000
LOG_DIR=/data/zilu/QseekLLM/src/post_train/logs
TB_DIR="$LOG_DIR/tb_${RUN_NAME}"
OUT="$LOG_DIR/ckpt_${RUN_NAME}"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
GPU_LOG="$LOG_DIR/${RUN_NAME}_gpu.csv"

mkdir -p "$LOG_DIR" "$OUT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${TRAIN_CARD},${ROLLOUT_CARD}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
export PYTHONPATH=/data/zilu/QseekLLM/src/post_train/verl${PYTHONPATH:+:$PYTHONPATH}
export TENSORBOARD_DIR="$TB_DIR"

source .venv/bin/activate 2>/dev/null || true

env -u CUDA_VISIBLE_DEVICES nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,nounits \
    -lms 1000 > "$GPU_LOG" &
MONITOR_PID=$!

cleanup_monitor() {
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup_monitor EXIT

python3 RL/main_grpo_sync_split.py \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/val.parquet" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    custom_reward_function.path=RL/reward_verl.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="$CKPT" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    actor_rollout_ref.actor.checkpoint.load_contents='["model"]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.balance_batch=True \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=qseek_grpo \
    trainer.experiment_name="$RUN_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$NSTEP" \
    trainer.default_local_dir="$OUT" \
    2>&1 | tee "$LOG_FILE"

python3 RL/parse_timing.py "$TB_DIR" "$GPU_LOG"
