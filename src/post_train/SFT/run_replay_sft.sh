#!/bin/bash
# ============================================================================
# 错题重放 SFT(对照实验 v3-wrongpool 组件③的SFT阶段)。
#   从 RL 当前档(HF)热启,在教师错题数据上做一轮短 SFT。
#   参数全走 env,供 RL/run_v3_wrongpool_experiment.sh 编排器逐轮调用。
# 用法: MODEL_PATH=<hf> DATA=<replay.parquet> SAVE_PATH=<out> EXP_NAME=<name> bash SFT/run_replay_sft.sh
# ============================================================================
set -x
cd "$(dirname "$0")/.."

MODEL_PATH=${MODEL_PATH:?need MODEL_PATH (HF dir to hot-start)}
DATA=${DATA:?need DATA (replay messages parquet)}
SAVE_PATH=${SAVE_PATH:?need SAVE_PATH}
EXP_NAME=${EXP_NAME:-sft_replay}
LR=${LR:-5e-6}
EPOCHS=${EPOCHS:-1}
MAXLEN=${MAXLEN:-16384}
TRAIN_BS=${TRAIN_BS:-64}
SFT_CARD=${SFT_CARD:-1}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$SFT_CARD
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
    data.max_length="$MAXLEN" \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=24576 \
    data.train_batch_size="$TRAIN_BS" \
    data.truncation=right \
    engine=fsdp \
    model.path="$MODEL_PATH" \
    model.use_liger=True \
    optim.lr="$LR" \
    optim.lr_scheduler_type=cosine \
    optim.min_lr_ratio=0.1 \
    optim.lr_warmup_steps_ratio=0.03 \
    optim.weight_decay=0.0 \
    trainer.default_local_dir="$SAVE_PATH" \
    trainer.project_name=qseek-posttrain \
    trainer.experiment_name="$EXP_NAME" \
    trainer.total_epochs="$EPOCHS" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.logger='["console","tensorboard"]' \
    "${@}"
