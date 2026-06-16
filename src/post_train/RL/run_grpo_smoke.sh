#!/bin/bash
# ============================================================================
# GRPO smoke（同步版本）· 目的:验证 RL 训练闭环 + 拿到时间拆解,不求训练效果
#   起点: S3-R1(sft_s3r1/global_step_3874_hf) —— 经分析其 pass@8 最高、收割带最宽
#   数据: rl_smoke(3000 题, big-math 收割带 solve_rate∈[0.15,0.85], math_verify 可判分)
#   判分: 复用评测同口径判分器 RL/reward_verl.py(custom_reward_function)
#   部署: colocated hybrid engine(同步 SPMD rollout), 单张 A800(card1, 80G)
#         —— smoke 先不做 A800训/A4000采样 的 split placement(那是 smoke 通过后的优化)
#   规模: LoRA rank32 / batch64 / group n=8 / resp≤1024 / 限 ~20 step
#
#   ⚠ verl 版本: 主 verl/ 是 0.8.0.dev(async-only rollout,需 vllm≥0.9,与本机 vllm 0.8.5 不兼容)。
#     故 RL 用隔离 worktree verl_v050(v0.5.0,依赖 vllm 0.7.3~0.8.5 + torch 2.6,带同步 SPMD rollout),
#     经 PYTHONPATH 覆盖,不影响主 verl/(保留 SFT 改动)。见 docs/rl_smoke_setup。
#
#   ⚠ CLAUDE.md 要求记录时间开销:verl 每步在 console 打 timing_s/gen(采样)、
#     timing_s/update_actor(训练 fwd+bwd)、timing_s/... ,日志见 logs/grpo_smoke.log,
#     跑完用 RL/parse_timing.py 汇总。
#
# 用法: bash RL/run_grpo_smoke.sh [card] [n_step]
# ============================================================================
set -xeuo pipefail
cd "$(dirname "$0")/.."   # -> post_train

CARD=${1:-1}
NSTEP=${2:-20}
CKPT=/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf
DATA=/data/zilu/data_unified_v2/rl_smoke
OUT=/data/zilu/fastrl/checkpoints/grpo_smoke

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$CARD
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
# 用隔离的 verl v0.5.0(适配 vllm 0.8.5),PYTHONPATH 覆盖主 verl/(0.8.0.dev)
export PYTHONPATH=/data/zilu/QseekLLM/src/post_train/verl_v050${PYTHONPATH:+:$PYTHONPATH}
export TENSORBOARD_DIR=/data/zilu/QseekLLM/src/post_train/logs/tb_grpo_smoke

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
    trainer.experiment_name=grpo_smoke_s3r1 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$NSTEP" \
    trainer.default_local_dir="$OUT"
