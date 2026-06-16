#!/bin/bash
# ============================================================================
# S3-R1 · 难度课程 SFT 第一轮(training_plan v2 §4.3)
#   - 热启 F2 终档(global_step_9858 的 HF 转换件,终评时已转好)
#   - 数据: S3-R1 切片(易:中:难=60:30:10 token 口径 + 15% 通用,≤16k,
#           见 data_pipeline/reweight_s3.py 与同名 manifest)
#   - 与 F2 超参差异: lr 1e-5→5e-6 + cosine(热启精调)/ ctx 8k→16k(长轨迹进场)/
#     1 epoch(防背题)/ save_freq 200(总步数少,保证评测曲线密度)
#   评测异步: python -m eval.async_eval --ckpt-dir <save> --gpu-candidates 2,3 --watch
# 用法: bash SFT/run_s3r1_sft.sh [save_path]
# ============================================================================
set -x
cd "$(dirname "$0")/.."   # -> post_train

SAVE_PATH=${1:-/data/zilu/fastrl/checkpoints/sft_s3r1}
BASE=/data/zilu/fastrl/checkpoints/sft_foundation_v2/global_step_9858_hf
DATA=/data/zilu/data_unified_v2/parquet/train_sft_s3r1_16k.parquet

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .venv/bin/activate 2>/dev/null || true

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.sft_trainer \
    data.train_files="$DATA" \
    data.val_files=null \
    data.messages_key=messages \
    data.max_length=16384 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=24576 \
    data.train_batch_size=256 \
    data.truncation=right \
    engine=fsdp \
    model.path="$BASE" \
    model.use_liger=True \
    optim.lr=5e-6 \
    optim.lr_scheduler_type=cosine \
    optim.min_lr_ratio=0.1 \
    optim.lr_warmup_steps_ratio=0.02 \
    optim.weight_decay=0.0 \
    trainer.default_local_dir="$SAVE_PATH" \
    trainer.project_name=qseek-posttrain \
    trainer.experiment_name=sft_s3r1 \
    trainer.total_epochs=1 \
    trainer.save_freq=200 \
    trainer.test_freq=-1 \
    trainer.logger='["console","tensorboard","wandb"]' \
    "${@:2}"
