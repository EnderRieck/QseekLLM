#!/bin/bash
# ============================================================================
# S4 · 最终阶段难题退火 SFT(s4_data_report_20260612.md §2/§7)
#   - 热启 s3r1 终档(global_step_3874 的 HF 转换件,由 eval.async_eval.convert_to_hf 产出)
#   - 数据: S4 退火切片 train_sft_s4_anneal_16k.parquet
#     (29.2 万条 / 0.133B token,hard 38% / medium 33% / easy 10% / general 19%,
#      清洗池 + heldout_v2 隔离,见 data_pipeline/clean_s4.py 与 reweight_s4.log)
#   - 与 S3-R1 超参差异: lr 5e-6→2e-6(退火,起点≈s3r1 中段 lr,终点 2e-7)/
#     其余一致(16k ctx / dynamic bsz / 1 epoch / save_freq 200)
#   - 规模 ~1140 步,A800 单卡预计 ~3.5h
#   评测异步(注意用 heldout v2 + 1024 生成上限,高考档防截断):
#     python -m eval.async_eval --ckpt-dir <save> --heldout eval/heldout_v2.jsonl \
#       --max-new-tokens 1024 --gpu-candidates 2,3 --watch
# 用法: bash SFT/run_s4_anneal_sft.sh [save_path] [base_hf]
# ============================================================================
set -x
cd "$(dirname "$0")/.."   # -> post_train

SAVE_PATH=${1:-/data/zilu/fastrl/checkpoints/sft_s4_anneal}
BASE=${2:-/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf}
DATA=/data/zilu/data_unified_v2/parquet/train_sft_s4_anneal_16k.parquet

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
    optim.lr=2e-6 \
    optim.lr_scheduler_type=cosine \
    optim.min_lr_ratio=0.1 \
    optim.lr_warmup_steps_ratio=0.02 \
    optim.weight_decay=0.0 \
    trainer.default_local_dir="$SAVE_PATH" \
    trainer.project_name=qseek-posttrain \
    trainer.experiment_name=sft_s4_anneal \
    trainer.total_epochs=1 \
    trainer.save_freq=200 \
    trainer.test_freq=-1 \
    trainer.logger='["console","tensorboard","wandb"]' \
    "${@:3}"
