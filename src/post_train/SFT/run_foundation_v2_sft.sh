#!/bin/bash
# ============================================================================
# F2 · Foundation SFT v2（training_plan v2 §4.2,数据=审计修复后的 v2 弹药库）
#   与 F1 的差异:
#   - 数据: data_unified_v2(orca 抽取修复/numina valid 过滤/中文数学池 ~18万/
#           MATH-train 解放/通用过滤器/全部泄漏修复),F2 配方切片
#   - 从 base 重训(不热启 F1——F1 数据含毒,见 docs/data_audit_report_20260610.md)
#   超参沿用 F1(已验证稳定);评测异步: python -m eval.async_eval --ckpt-dir <save> --gpu-candidates 2,3 --watch
# 用法: bash SFT/run_foundation_v2_sft.sh [save_path]
# ============================================================================
set -x
cd "$(dirname "$0")/.."   # -> post_train

SAVE_PATH=${1:-/data/zilu/fastrl/checkpoints/sft_foundation_v2}
BASE=/data/zilu/fastrl/checkpoints/qseek_digitsplit_base
DATA=/data/zilu/data_unified_v2/parquet/train_sft_foundation_8k.parquet

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
    data.max_length=8192 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=24576 \
    data.train_batch_size=256 \
    data.truncation=right \
    engine=fsdp \
    model.path="$BASE" \
    model.use_liger=True \
    optim.lr=1e-5 \
    optim.weight_decay=0.0 \
    trainer.default_local_dir="$SAVE_PATH" \
    trainer.project_name=qseek-posttrain \
    trainer.experiment_name=sft_foundation_v2 \
    trainer.total_epochs=2 \
    trainer.save_freq=500 \
    trainer.test_freq=-1 \
    trainer.logger='["console","tensorboard","wandb"]' \
    "${@:2}"
