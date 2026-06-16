#!/bin/bash
# ============================================================================
# S3-R1 终评(补)· 在被占满的机器上"抠"一张闲卡跑
#   背景: 机器 8 卡全被他人 test_xstest(Llama-2-7b device_map 铺满)占了内存,
#   但多数卡 compute util≈0(只占显存不算)。挑一张 util≈0 的卡(默认 0),
#   压低 gpu-mem-util 让 1.6B 塞进剩余 ~9G,不抢它算力 → 慢但能跑完。
#   选卡方法见 docs(采样 util,挑占内存但 0% 的卡;勿选他正在算的卡否则算力饿死)。
#
#   benchmark 顺序: 小而可比的先跑(svamp/math500/cmath/gsm8k 与 F2/Qwen 对齐),
#   大的(gsmplus/cc-reserved)垫后,这样早期就有可用结果,中途断也不亏。
#   每个 bench 跑完即落 <bench>.jsonl;全跑完才写 summary.md(断了可用 rescore 补)。
#
# 用法: bash eval/run_s3r1_final_eval.sh [card] [gpu_mem_util]
#   默认 card=0 util=0.72。改卡先重采样 util 确认该卡 compute≈0。
# ============================================================================
set -x
cd "$(dirname "$0")/.."   # -> post_train

CARD=${1:-0}
MEM=${2:-0.72}
CKPT=/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf

export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .venv/bin/activate 2>/dev/null || true

python -u -m eval.final_eval \
    --ckpt "$CKPT" \
    --gpus "$CARD" \
    --gpu-mem-util "$MEM" \
    --benchmarks "svamp,math500,cmath,gsm8k,gsmplus,cc-reserved" \
    --k 8 \
    --max-new-tokens 2048 \
    --chunk-size 64
