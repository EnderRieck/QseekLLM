#!/bin/bash
# verl-NATIVE fully-async GRPO (efficiency-comparison baseline #2b = "verl original fully_async").
#
# This is verl's OWN `experimental/fully_async_policy/fully_async_main.py`, WITHOUT our
# split layer. Architecturally it is the direct parent of our (3) fully-async-split:
#   - same FullyAsyncTrainer + MessageQueue fully-async pipeline,
#   - BUT ref is in the TRAINING resource pool (separation/utils.py puts Role.RefPolicy in
#     training_roles) => ref COLOCATED on the training card (card1, A800). No separate ref
#     card, no ref-service. card3 is intentionally left idle.
#
# Therefore (2b) vs (3) is a clean controlled A/B that isolates exactly what our split adds:
# pulling ref out onto its own card3 + async ref-service.
#
# Usage: bash RL/run_grpo_fully_async_nosplit_a800_a4000.sh [rollout_gpu_count] [train_steps]
set -xeuo pipefail
cd "$(dirname "$0")/.."

ROLLOUT_COUNT=${1:-4}
NSTEP=${NSTEP:-${2:-5}}

TRAIN_CARD=${TRAIN_CARD:-1}
ROLLOUT_POOL_CSV=${ROLLOUT_POOL_CSV:-4,5,6,7}
RUN_GROUP=${RUN_GROUP:-grpo_fully_async_nosplit_a800_a4000_$(date +%Y%m%d_%H%M%S)}

# --- unified comparison knobs (identical to the other variants) ---
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-2}
TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-1}
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-2.0}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
# verl-original fully_async computes ref SYNCHRONOUSLY in fit_step (upstream HEAD:
# fit_step -> self._fit_compute_ref_log_prob, blocking). The async-ref prefetch +
# RefService are OUR additions (uncommitted +528 lines). To faithfully reproduce the
# upstream baseline we force async_ref=False here. Set ASYNC_REF=True to instead measure
# "our async-ref prefetch but ref still colocated / no ref-service".
ASYNC_REF=${ASYNC_REF:-False}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.005}
REWARD_FORMAT_BONUS=${REWARD_FORMAT_BONUS:-0.05}
REWARD_THINK_LEN_MAX_BONUS=${REWARD_THINK_LEN_MAX_BONUS:-0.0}
export REWARD_FORMAT_BONUS REWARD_THINK_LEN_MAX_BONUS

ROLLOUT_TOTAL_BASE=$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TRIGGER_PARAMETER_SYNC_STEP * NSTEP))
ROLLOUT_TOTAL_STEPS=${ROLLOUT_TOTAL_STEPS:-$((ROLLOUT_TOTAL_BASE + 32))}
READY_QUEUE_SIZE=${READY_QUEUE_SIZE:-$((PPO_MINI_BATCH_SIZE * REQUIRE_BATCHES * TRIGGER_PARAMETER_SYNC_STEP * 2))}

CHECKPOINT_BACKEND=${CHECKPOINT_BACKEND:-nccl}
WEIGHT_BUCKET_MB=${WEIGHT_BUCKET_MB:-512}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-32}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}
GPU_SAMPLE_MS=${GPU_SAMPLE_MS:-1000}

IFS=',' read -r -a ROLLOUT_POOL <<< "$ROLLOUT_POOL_CSV"
if (( ROLLOUT_COUNT < 1 || ROLLOUT_COUNT > ${#ROLLOUT_POOL[@]} )); then
    echo "ROLLOUT_COUNT must be in [1, ${#ROLLOUT_POOL[@]}], got ${ROLLOUT_COUNT}" >&2; exit 2
fi
join_by_comma() { local IFS=,; echo "$*"; }
ROLLOUT_CARDS=$(join_by_comma "${ROLLOUT_POOL[@]:0:ROLLOUT_COUNT}")
# training card first so verl assigns visible-index 0 (A800) to the training pool (actor+ref).
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
    echo "variant=verl_native_fully_async_nosplit"
    echo "rollout_count=${ROLLOUT_COUNT}"
    echo "train_steps=${NSTEP}"
    echo "train_card=${TRAIN_CARD}"
    echo "ref_placement=colocated_in_training_pool (card1; no separate ref card; card3 idle)"
    echo "rollout_cards=${ROLLOUT_CARDS}"
    echo "cuda_visible_devices=${VISIBLE_CARDS}"
    echo "checkpoint=${CKPT}"
    echo "ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    echo "require_batches=${REQUIRE_BATCHES}"
    echo "trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP}"
    echo "staleness_threshold=${STALENESS_THRESHOLD}"
    echo "rollout_n=${ROLLOUT_N}"
    echo "kl_loss_coef=${KL_LOSS_COEF}"
    echo "weight_bucket_mb=${WEIGHT_BUCKET_MB}"
    echo "rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    echo "phase_event_log_path=${PHASE_EVENT_LOG_PATH}"
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
    --format=csv,nounits -lms "$GPU_SAMPLE_MS" > "$GPU_LOG" &
MONITOR_PID=$!
cleanup_monitor() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup_monitor EXIT

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    hydra.searchpath="[file:///data/zilu/QseekLLM/src/post_train/verl/verl/trainer/config]" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo="$NORM_ADV_BY_STD" \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/val.parquet" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    custom_reward_function.path=RL/reward_verl.py \
    custom_reward_function.name=compute_score \
    reward.custom_reward_function.path=RL/reward_verl.py \
    reward.custom_reward_function.name=compute_score \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
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
    trainer.default_local_dir="$OUT" \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node="$ROLLOUT_COUNT" \
    rollout.n="$ROLLOUT_N" \
    rollout.total_rollout_steps="$ROLLOUT_TOTAL_STEPS" \
    async_training.staleness_threshold="$STALENESS_THRESHOLD" \
    async_training.trigger_parameter_sync_step="$TRIGGER_PARAMETER_SYNC_STEP" \
    async_training.require_batches="$REQUIRE_BATCHES" \
    async_training.partial_rollout="$PARTIAL_ROLLOUT" \
    +fully_async_split.async_ref="$ASYNC_REF" \
    +fully_async_split.ref_service=False \
    2>&1 | tee "$LOG_FILE"

python3 RL/parse_timing.py "$TB_DIR" "$GPU_LOG" 2>&1 | tee "$PARSE_LOG" || true
