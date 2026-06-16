#!/bin/bash
# Run one synchronous GRPO timing case:
#   actor/update: physical GPU 1 (A800)
#   ref forward:  physical GPU 3 (A4000)
#   rollout:      first N GPUs from physical 4,5,6,7 (A4000)
#
# Usage:
#   bash RL/run_grpo_sync_a800_ref3_a4000_one.sh [rollout_gpu_count] [n_step]
set -xeuo pipefail
cd "$(dirname "$0")/.."

ROLLOUT_COUNT=${1:-1}
NSTEP=${2:-5}

TRAIN_CARD=${TRAIN_CARD:-1}
REF_CARDS_CSV=${REF_CARDS_CSV:-3}
ROLLOUT_POOL_CSV=${ROLLOUT_POOL_CSV:-4,5,6,7}
RUN_GROUP=${RUN_GROUP:-grpo_sync_a800_ref3_a4000_$(date +%Y%m%d_%H%M%S)}
REF_STRATEGY=${REF_STRATEGY:-fsdp}
REF_FSDP_SIZE=${REF_FSDP_SIZE:--1}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-False}
REF_RESHARD_AFTER_FORWARD=${REF_RESHARD_AFTER_FORWARD:-False}
REF_REPLICA_SIZE=${REF_REPLICA_SIZE:-1}
CHECKPOINT_BACKEND=${CHECKPOINT_BACKEND:-nccl}
WEIGHT_BUCKET_MB=${WEIGHT_BUCKET_MB:-512}
SERIAL_ROLLOUT_WEIGHT_SYNC=${SERIAL_ROLLOUT_WEIGHT_SYNC:-0}
NCCL_REBUILD_GROUP=${NCCL_REBUILD_GROUP:-1}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-32}

count_csv_items() {
    local value=$1
    local -a items
    IFS=',' read -r -a items <<< "$value"
    echo "${#items[@]}"
}

REF_COUNT=${REF_COUNT:-$(count_csv_items "$REF_CARDS_CSV")}

IFS=',' read -r -a ROLLOUT_POOL <<< "$ROLLOUT_POOL_CSV"
if (( ROLLOUT_COUNT < 1 || ROLLOUT_COUNT > ${#ROLLOUT_POOL[@]} )); then
    echo "ROLLOUT_COUNT must be in [1, ${#ROLLOUT_POOL[@]}], got ${ROLLOUT_COUNT}" >&2
    exit 2
fi

join_by_comma() {
    local IFS=,
    echo "$*"
}

ROLLOUT_CARDS=$(join_by_comma "${ROLLOUT_POOL[@]:0:ROLLOUT_COUNT}")
VISIBLE_CARDS="${TRAIN_CARD},${REF_CARDS_CSV},${ROLLOUT_CARDS}"

CKPT=${CKPT:-/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf}
DATA=${DATA:-/data/zilu/data_unified_v2/rl_smoke}
LOG_DIR=${LOG_DIR:-/data/zilu/QseekLLM/src/post_train/logs}
RUN_NAME="${RUN_GROUP}_rollout${ROLLOUT_COUNT}"
TB_DIR="$LOG_DIR/tb_${RUN_NAME}"
OUT="$LOG_DIR/ckpt_${RUN_NAME}"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
GPU_LOG="$LOG_DIR/${RUN_NAME}_gpu.csv"
PARSE_LOG="$LOG_DIR/${RUN_NAME}_timing.txt"
META_FILE="$LOG_DIR/${RUN_NAME}_meta.env"

mkdir -p "$LOG_DIR" "$OUT"

{
    echo "run_name=${RUN_NAME}"
    echo "rollout_count=${ROLLOUT_COUNT}"
    echo "nstep=${NSTEP}"
    echo "train_card=${TRAIN_CARD}"
    echo "ref_cards=${REF_CARDS_CSV}"
    echo "ref_count=${REF_COUNT}"
    echo "rollout_cards=${ROLLOUT_CARDS}"
    echo "cuda_visible_devices=${VISIBLE_CARDS}"
    echo "checkpoint=${CKPT}"
    echo "data=${DATA}"
    echo "old_log_prob_mode=rollout_bypass"
    echo "ref_fsdp_use_torch_compile=false"
    echo "ref_strategy=${REF_STRATEGY}"
    echo "ref_fsdp_size=${REF_FSDP_SIZE}"
    echo "ref_param_offload=${REF_PARAM_OFFLOAD}"
    echo "ref_reshard_after_forward=${REF_RESHARD_AFTER_FORWARD}"
    echo "ref_replica_size=${REF_REPLICA_SIZE}"
    echo "checkpoint_backend=${CHECKPOINT_BACKEND}"
    echo "weight_bucket_mb=${WEIGHT_BUCKET_MB}"
    echo "serial_rollout_weight_sync=${SERIAL_ROLLOUT_WEIGHT_SYNC}"
    echo "nccl_rebuild_group=${NCCL_REBUILD_GROUP}"
    echo "nccl_p2p_disable=${NCCL_P2P_DISABLE}"
    echo "rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    echo "rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    echo "rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    echo "rollout_agent_num_workers=${ROLLOUT_AGENT_NUM_WORKERS}"
} > "$META_FILE"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$VISIBLE_CARDS"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VERL_FSDP_INIT_TRACE=${VERL_FSDP_INIT_TRACE:-0}
export VERL_SERIAL_ROLLOUT_WEIGHT_SYNC="$SERIAL_ROLLOUT_WEIGHT_SYNC"
export VERL_NCCL_REBUILD_GROUP="$NCCL_REBUILD_GROUP"
export NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE"
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
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.loss_type=ppo_clip \
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
    actor_rollout_ref.rollout.n_gpus_per_node="$ROLLOUT_COUNT" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend="$CHECKPOINT_BACKEND" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$WEIGHT_BUCKET_MB" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_NUM_WORKERS" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.strategy="$REF_STRATEGY" \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp16 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision='{param_dtype:fp16,reduce_dtype:fp32,buffer_dtype:fp32}' \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward="$REF_RESHARD_AFTER_FORWARD" \
    actor_rollout_ref.ref.fsdp_config.fsdp_size="$REF_FSDP_SIZE" \
    +split_sync.ref_n_gpus_per_node="$REF_COUNT" \
    +split_sync.ref_nnodes=1 \
    +split_sync.ref_replica_size="$REF_REPLICA_SIZE" \
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

python3 RL/parse_timing.py "$TB_DIR" "$GPU_LOG" 2>&1 | tee "$PARSE_LOG"
