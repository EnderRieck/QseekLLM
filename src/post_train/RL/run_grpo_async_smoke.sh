#!/bin/bash
# ============================================================================
# GRPO smoke（异步版本 / async server rollout）· 验证主 verl 0.8.0.dev 异步闭环 + 时间拆解
#   起点: S3-R1(sft_s3r1/global_step_3874_hf)
#   数据: rl_smoke(big-math 收割带, math_verify); 判分: RL/reward_verl.py(评测同口径)
#   部署: 主 verl 0.8.0.dev 的 async server rollout, 单张 A800(card1, 80G), colocated
#   规模: LoRA rank32 / batch64 / group n=8 / resp≤1024 / 限 ~20 step
#
#   ⚠ 环境: 已把主 .venv 升级到 torch2.7.1+cu126 / vllm0.10.1.1 / transformers4.57
#     (适配 verl 0.8.0.dev 的 async server: 需 run_headless + --logprobs-mode, 见 docs/rl_async_env_upgrade)。
#     与同步版(RL/run_grpo_smoke.sh, 用隔离 verl_v050)区分: 本脚本用主 verl + mode=async。
#
#   时间拆解: tensorboard logs/tb_grpo_async_smoke, 用 RL/parse_timing.py 汇总。
#
# 用法: bash RL/run_grpo_async_smoke.sh [card] [n_step]
# ============================================================================
set -xeuo pipefail
cd "$(dirname "$0")/.."   # -> post_train

CARD=${1:-1}
NSTEP=${2:-20}
CKPT=/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf
DATA=/data/zilu/data_unified_v2/rl_smoke
OUT=/data/zilu/fastrl/checkpoints/grpo_async_smoke

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$CARD
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
# 主 verl 0.8.0.dev(async server rollout)+ 升级后 .venv(torch2.7.1/vllm0.10.1.1)
export TENSORBOARD_DIR=/data/zilu/QseekLLM/src/post_train/logs/tb_grpo_async_smoke

source .venv/bin/activate 2>/dev/null || true

python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
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
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
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
    trainer.experiment_name=grpo_async_smoke_s3r1 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$NSTEP" \
    trainer.default_local_dir="$OUT"
