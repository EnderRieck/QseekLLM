#!/bin/bash
# Fully async GRPO split smoke/formal runner:
#   actor/update: physical GPU 1 (A800)
#   ref forward:  physical GPU 3 (A4000)
#   rollout:      first N GPUs from physical 4,5,6,7 (A4000)
#
# Usage:
#   bash RL/run_grpo_fully_async_split_a800_ref3_a4000.sh [rollout_gpu_count] [train_steps]
set -xeuo pipefail
cd "$(dirname "$0")/.."

ROLLOUT_COUNT=${1:-4}
NSTEP=${NSTEP:-${2:-1}}

TRAIN_CARD=${TRAIN_CARD:-1}
REF_CARDS_CSV=${REF_CARDS_CSV:-3}
ROLLOUT_POOL_CSV=${ROLLOUT_POOL_CSV:-4,5,6,7}
RUN_GROUP=${RUN_GROUP:-grpo_fully_async_split_a800_ref3_a4000_$(date +%Y%m%d_%H%M%S)}

PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-2}
TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.1}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
# v2 退化修复(见 docs/rl_degradation_diagnosis_20260613.md):
#   NORM_ADV_BY_STD=False  -> 全错/低方差组不再把 shaping 噪声经 std 归一放大成伪梯度;
#   KL_LOSS_COEF 0.001->0.005 -> 锚住 SFT 已会的简洁正确行为,刹住能力侵蚀;
#   REWARD_FORMAT_BONUS 0.1->0.05 / REWARD_THINK_LEN_MAX_BONUS 0.2->0 -> 砍 shaping,correct 主导。
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.005}
REWARD_FORMAT_BONUS=${REWARD_FORMAT_BONUS:-0.05}
REWARD_THINK_LEN_MAX_BONUS=${REWARD_THINK_LEN_MAX_BONUS:-0.0}
export REWARD_FORMAT_BONUS REWARD_THINK_LEN_MAX_BONUS
# Process eval: reuse sampling side (rollouter) on held-out val, no extra GPUs.
# Defaults keep eval OFF for smoke; formal runs override TEST_FREQ/VAL_BEFORE_TRAIN/VAL_FILES.
TEST_FREQ=${TEST_FREQ:-1000000}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
USE_TRAINER_DO_VALIDATE=${USE_TRAINER_DO_VALIDATE:-False}
SAVE_FREQ=${SAVE_FREQ:--1}
ASYNC_REF=${ASYNC_REF:-True}
REF_SERVICE=${REF_SERVICE:-True}
REF_MICRO_BATCH_SIZE=${REF_MICRO_BATCH_SIZE:-16}
REF_MICRO_BATCH_TIMEOUT_S=${REF_MICRO_BATCH_TIMEOUT_S:-0.2}
REF_SERVICE_ROLLOUT_BUFFER=${REF_SERVICE_ROLLOUT_BUFFER:-$REF_MICRO_BATCH_SIZE}
ROLLOUT_TOTAL_BASE=$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TRIGGER_PARAMETER_SYNC_STEP * NSTEP))
if [[ "$REF_SERVICE" == "True" || "$REF_SERVICE" == "true" || "$REF_SERVICE" == "1" ]]; then
    ROLLOUT_TOTAL_DEFAULT=$((ROLLOUT_TOTAL_BASE + REF_SERVICE_ROLLOUT_BUFFER))
else
    ROLLOUT_TOTAL_DEFAULT=$ROLLOUT_TOTAL_BASE
fi
ROLLOUT_TOTAL_STEPS=${ROLLOUT_TOTAL_STEPS:-$ROLLOUT_TOTAL_DEFAULT}

CHECKPOINT_BACKEND=${CHECKPOINT_BACKEND:-nccl}
WEIGHT_BUCKET_MB=${WEIGHT_BUCKET_MB:-512}
NCCL_REBUILD_GROUP=${NCCL_REBUILD_GROUP:-1}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-32}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}
READY_QUEUE_SIZE=${READY_QUEUE_SIZE:-$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TRIGGER_PARAMETER_SYNC_STEP * 2))}
MASTER_PORT_RANGE=${MASTER_PORT_RANGE:-}

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
TRAIN_FILES=${TRAIN_FILES:-$DATA/train.parquet}
VAL_FILES=${VAL_FILES:-$DATA/val.parquet}
LOG_DIR=${LOG_DIR:-/data/zilu/QseekLLM/src/post_train/logs}
RUN_NAME="${RUN_GROUP}_rollout${ROLLOUT_COUNT}_steps${NSTEP}"
TB_DIR="$LOG_DIR/tb_${RUN_NAME}"
OUT="$LOG_DIR/ckpt_${RUN_NAME}"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
GPU_LOG="$LOG_DIR/${RUN_NAME}_gpu.csv"
PARSE_LOG="$LOG_DIR/${RUN_NAME}_timing.txt"
META_FILE="$LOG_DIR/${RUN_NAME}_meta.env"
# Detailed timestamped event log for post-hoc analysis (read by EventLogger in trainer).
EVENT_LOG_PATH=${EVENT_LOG_PATH:-$LOG_DIR/${RUN_NAME}_events.jsonl}
export EVENT_LOG_PATH
# Per-eval full input/output dump dir: every process eval writes <step>.jsonl with
# input/output/gts/score/correct/has_format/... for full transparency.
VAL_DUMP_DIR=${VAL_DUMP_DIR:-$LOG_DIR/${RUN_NAME}_val_dumps}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-20}

mkdir -p "$LOG_DIR" "$OUT"

{
    echo "run_name=${RUN_NAME}"
    echo "rollout_count=${ROLLOUT_COUNT}"
    echo "train_steps=${NSTEP}"
    echo "train_card=${TRAIN_CARD}"
    echo "ref_cards=${REF_CARDS_CSV}"
    echo "ref_count=${REF_COUNT}"
    echo "rollout_cards=${ROLLOUT_CARDS}"
    echo "cuda_visible_devices=${VISIBLE_CARDS}"
    echo "checkpoint=${CKPT}"
    echo "data=${DATA}"
    echo "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    echo "require_batches=${REQUIRE_BATCHES}"
    echo "trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP}"
    echo "staleness_threshold=${STALENESS_THRESHOLD}"
    echo "partial_rollout=${PARTIAL_ROLLOUT}"
    echo "rollout_total_base=${ROLLOUT_TOTAL_BASE}"
    echo "ref_service_rollout_buffer=${REF_SERVICE_ROLLOUT_BUFFER}"
    echo "rollout_total_steps=${ROLLOUT_TOTAL_STEPS}"
    echo "rollout_n=${ROLLOUT_N}"
    echo "rollout_temperature=${ROLLOUT_TEMPERATURE}"
    echo "norm_adv_by_std=${NORM_ADV_BY_STD}"
    echo "kl_loss_coef=${KL_LOSS_COEF}"
    echo "reward_format_bonus=${REWARD_FORMAT_BONUS}"
    echo "reward_think_len_max_bonus=${REWARD_THINK_LEN_MAX_BONUS}"
    echo "checkpoint_backend=${CHECKPOINT_BACKEND}"
    echo "weight_bucket_mb=${WEIGHT_BUCKET_MB}"
    echo "nccl_rebuild_group=${NCCL_REBUILD_GROUP}"
    echo "nccl_p2p_disable=${NCCL_P2P_DISABLE}"
    echo "rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    echo "rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    echo "rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    echo "rollout_agent_num_workers=${ROLLOUT_AGENT_NUM_WORKERS}"
    echo "reward_num_workers=${REWARD_NUM_WORKERS}"
    echo "ref_service=${REF_SERVICE}"
    echo "ref_micro_batch_size=${REF_MICRO_BATCH_SIZE}"
    echo "ref_micro_batch_timeout_s=${REF_MICRO_BATCH_TIMEOUT_S}"
    echo "ready_queue_size=${READY_QUEUE_SIZE}"
    echo "master_port_range=${MASTER_PORT_RANGE}"
    echo "train_files=${TRAIN_FILES}"
    echo "val_files=${VAL_FILES}"
    echo "test_freq=${TEST_FREQ}"
    echo "val_before_train=${VAL_BEFORE_TRAIN}"
    echo "use_trainer_do_validate=${USE_TRAINER_DO_VALIDATE}"
    echo "save_freq=${SAVE_FREQ}"
    echo "async_ref=${ASYNC_REF}"
    echo "event_log_path=${EVENT_LOG_PATH}"
    echo "val_dump_dir=${VAL_DUMP_DIR}"
    echo "log_val_generations=${LOG_VAL_GENERATIONS}"
} > "$META_FILE"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$VISIBLE_CARDS"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
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

python3 RL/main_grpo_fully_async_split.py \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo="$NORM_ADV_BY_STD" \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.loss_type=ppo_clip \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    custom_reward_function.path=RL/reward_verl.py \
    custom_reward_function.name=compute_score \
    reward.num_workers="$REWARD_NUM_WORKERS" \
    actor_rollout_ref.model.path="$CKPT" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    actor_rollout_ref.actor.checkpoint.load_contents='["model"]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
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
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp16 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision='{param_dtype:fp16,reduce_dtype:fp32,buffer_dtype:fp32}' \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=-1 \
    trainer.balance_batch=True \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=qseek_grpo \
    trainer.experiment_name="$RUN_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.validation_data_dir="$VAL_DUMP_DIR" \
    trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$OUT" \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node="$ROLLOUT_COUNT" \
    rollout.n="$ROLLOUT_N" \
    rollout.total_rollout_steps="$ROLLOUT_TOTAL_STEPS" \
    async_training.staleness_threshold="$STALENESS_THRESHOLD" \
    async_training.trigger_parameter_sync_step="$TRIGGER_PARAMETER_SYNC_STEP" \
    async_training.require_batches="$REQUIRE_BATCHES" \
    async_training.partial_rollout="$PARTIAL_ROLLOUT" \
    async_training.use_trainer_do_validate="$USE_TRAINER_DO_VALIDATE" \
    +fully_async_split.ref_n_gpus_per_node="$REF_COUNT" \
    +fully_async_split.ref_nnodes=1 \
    +fully_async_split.async_ref="$ASYNC_REF" \
    +fully_async_split.ref_service="$REF_SERVICE" \
    +fully_async_split.ref_micro_batch_size="$REF_MICRO_BATCH_SIZE" \
    +fully_async_split.ref_micro_batch_timeout_s="$REF_MICRO_BATCH_TIMEOUT_S" \
    +fully_async_split.ready_queue_size="$READY_QUEUE_SIZE" \
    ${MASTER_PORT_RANGE:+\
    +fully_async_split.master_port_range="$MASTER_PORT_RANGE"} \
    2>&1 | tee "$LOG_FILE"

python3 RL/parse_timing.py "$TB_DIR" "$GPU_LOG" 2>&1 | tee "$PARSE_LOG"
