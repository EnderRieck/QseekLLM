#!/bin/bash
# verl-native one-step-off async GRPO (efficiency-comparison baseline #2).
#
# Architecture (verl upstream `experimental/one_step_off_policy`):
#   - training (actor + REF colocated): physical GPU 1 (A800)
#   - rollout:                          first N GPUs from physical 4,5,6,7 (A4000)
#   - NO separate ref card: ref forward is colocated on the training GPU and runs
#     inside the train step (param_offload=True). This is the key architectural
#     difference vs our fully-async split (which gives ref its own card3 + an
#     async ref-service). card3 is intentionally left idle here.
#
# Overlap model: rollout(t+1) overlaps actor-update(t) ("one step off"), but ref
# stays on the actor's critical path.
#
# Usage:
#   bash RL/run_grpo_one_step_off_a800_a4000.sh [rollout_gpu_count] [n_step]
set -xeuo pipefail
cd "$(dirname "$0")/.."

ROLLOUT_COUNT=${1:-4}
NSTEP=${2:-5}

TRAIN_CARD=${TRAIN_CARD:-1}
ROLLOUT_POOL_CSV=${ROLLOUT_POOL_CSV:-4,5,6,7}
RUN_GROUP=${RUN_GROUP:-grpo_one_step_off_a800_a4000_$(date +%Y%m%d_%H%M%S)}

# --- comparison-critical unified knobs (identical across all three variants) ---
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.005}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-1024}

ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
WEIGHT_BUCKET_MB=${WEIGHT_BUCKET_MB:-512}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
GPU_SAMPLE_MS=${GPU_SAMPLE_MS:-1000}

IFS=',' read -r -a ROLLOUT_POOL <<< "$ROLLOUT_POOL_CSV"
if (( ROLLOUT_COUNT < 1 || ROLLOUT_COUNT > ${#ROLLOUT_POOL[@]} )); then
    echo "ROLLOUT_COUNT must be in [1, ${#ROLLOUT_POOL[@]}], got ${ROLLOUT_COUNT}" >&2
    exit 2
fi
join_by_comma() { local IFS=,; echo "$*"; }
ROLLOUT_CARDS=$(join_by_comma "${ROLLOUT_POOL[@]:0:ROLLOUT_COUNT}")
# training card first so verl assigns visible-index 0 (A800) to the training pool.
VISIBLE_CARDS="${TRAIN_CARD},${ROLLOUT_CARDS}"

CKPT=${CKPT:-/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf}
DATA=${DATA:-/data/zilu/data_unified_v2/rl_smoke}
LOG_DIR=${LOG_DIR:-/data/zilu/QseekLLM/src/post_train/logs}
RUN_NAME="${RUN_GROUP}_rollout${ROLLOUT_COUNT}_steps${NSTEP}"
TB_DIR="$LOG_DIR/tb_${RUN_NAME}"
OUT="$LOG_DIR/ckpt_${RUN_NAME}"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
GPU_LOG="$LOG_DIR/${RUN_NAME}_gpu.csv"
PARSE_LOG="$LOG_DIR/${RUN_NAME}_timing.txt"
META_FILE="$LOG_DIR/${RUN_NAME}_meta.env"
PHASE_EVENT_LOG_PATH=${PHASE_EVENT_LOG_PATH:-$LOG_DIR/${RUN_NAME}_phase.jsonl}
export PHASE_EVENT_LOG_PATH

mkdir -p "$LOG_DIR" "$OUT"

{
    echo "run_name=${RUN_NAME}"
    echo "variant=verl_one_step_off"
    echo "rollout_count=${ROLLOUT_COUNT}"
    echo "nstep=${NSTEP}"
    echo "train_card=${TRAIN_CARD}"
    echo "ref_placement=colocated_on_train_card (no separate ref card; card3 idle)"
    echo "rollout_cards=${ROLLOUT_CARDS}"
    echo "cuda_visible_devices=${VISIBLE_CARDS}"
    echo "checkpoint=${CKPT}"
    echo "data=${DATA}"
    echo "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    echo "train_batch_size=${TRAIN_BATCH_SIZE}"
    echo "rollout_n=${ROLLOUT_N}"
    echo "rollout_temperature=${ROLLOUT_TEMPERATURE}"
    echo "kl_loss_coef=${KL_LOSS_COEF}"
    echo "max_prompt_length=${MAX_PROMPT_LEN}"
    echo "max_response_length=${MAX_RESPONSE_LEN}"
    echo "nccl_p2p_disable=${NCCL_P2P_DISABLE}"
    echo "phase_event_log_path=${PHASE_EVENT_LOG_PATH}"
    echo "gpu_sample_ms=${GPU_SAMPLE_MS}"
} > "$META_FILE"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$VISIBLE_CARDS"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE"
export PYTHONPATH=/data/zilu/QseekLLM/src/post_train/verl${PYTHONPATH:+:$PYTHONPATH}
export TENSORBOARD_DIR="$TB_DIR"

source .venv/bin/activate 2>/dev/null || true

env -u CUDA_VISIBLE_DEVICES nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,nounits \
    -lms "$GPU_SAMPLE_MS" > "$GPU_LOG" &
MONITOR_PID=$!
cleanup_monitor() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup_monitor EXIT

python3 -m verl.experimental.one_step_off_policy.main_ppo \
    hydra.searchpath="[file:///data/zilu/QseekLLM/src/post_train/verl/verl/trainer/config]" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/val.parquet" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    custom_reward_function.path=RL/reward_verl.py \
    custom_reward_function.name=compute_score \
    reward.custom_reward_function.path=RL/reward_verl.py \
    reward.custom_reward_function.name=compute_score \
    reward.num_workers=16 \
    actor_rollout_ref.model.path="$CKPT" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$WEIGHT_BUCKET_MB" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.balance_batch=True \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=qseek_grpo \
    trainer.experiment_name="$RUN_NAME" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$NSTEP" \
    trainer.default_local_dir="$OUT" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node="$ROLLOUT_COUNT" \
    2>&1 | tee "$LOG_FILE"

python3 RL/parse_timing.py "$TB_DIR" "$GPU_LOG" 2>&1 | tee "$PARSE_LOG" || true
