#!/bin/bash
# ============================================================================
# 阶段1–2 基础 SFT（hybrid_sft_rl_design.md：前期纯 SFT 广混大打基础）
#   base = qseek_digitsplit_base (1.7B, 数字切分, 16k)
#   data = 统一 SFT parquet（数学 worked + 题目理解 + 通用，181.9万）
#   训练在 A800(card1)；eval 另在 card2,3 异步（本脚本只管训练）
# 用法: bash SFT/run_foundation_sft.sh [save_path]
# ============================================================================
set -x
cd "$(dirname "$0")/.."   # -> post_train

SAVE_PATH=${1:-/data/zilu/fastrl/checkpoints/sft_foundation}
BASE=/data/zilu/fastrl/checkpoints/qseek_digitsplit_base
# 重采后的基础 SFT 池（易题为主、竞赛降权、①算术骨干为主、通用~24%），已过滤 ≤8192 token
DATA=/data/zilu/data_unified/parquet/train_sft_foundation_8k.parquet

# 卡：A800=card1。锁 PCI 顺序（CLAUDE.md/vllm 警告）。训练不需代理。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com
# wandb 离线兜底（无网时仍落本地）；有网可改 online
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 抗显存碎片，降 OOM 风险

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
    trainer.experiment_name=sft_foundation \
    trainer.total_epochs=2 \
    trainer.save_freq=500 \
    trainer.test_freq=-1 \
    trainer.logger='["console","tensorboard","wandb"]' \
    "${@:2}"
