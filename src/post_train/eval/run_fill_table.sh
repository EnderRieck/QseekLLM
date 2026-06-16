#!/bin/bash
# 补齐终评总表:自动等卡。
#   A) Qwen base + instruct 的 gsmplus(各1格)-> GPU2 立刻串行跑
#   B) RL v2 gs300 全6项 -> 等 GPU 0,3 腾空后跑(final_eval 内部自动转HF)
# 日志: logs/fill_*.log
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd -P)
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python"
LOGD="$ROOT/logs"

freemem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
wait_free(){ # 等指定卡(逗号分隔)都空闲(<2000MiB)
  local cards="$1" name="$2"
  while :; do
    local ok=1
    for c in ${cards//,/ }; do [ "$(freemem $c)" -lt 2000 ] 2>/dev/null || ok=0; done
    [ "$ok" = 1 ] && break
    echo "[$(date +%H:%M:%S)] wait $name: GPU $cards busy..."; sleep 60
  done
  echo "[$(date +%H:%M:%S)] GPU $cards free -> launch $name"
}

run_eval(){ # ckpt gpus benchmarks out-dir log
  echo "[$(date +%H:%M:%S)] START: $5"
  $PY -m eval.final_eval --ckpt "$1" --gpus "$2" --benchmarks "$3" \
      ${4:+--out-dir "$4"} --k 8 --max-new-tokens 2048 --chunk-size 64 \
      > "$LOGD/$5" 2>&1
  echo "[$(date +%H:%M:%S)] DONE($?): $5"
}

QB=/data/zilu/fastrl/checkpoints/external/qwen3_1_7b_base
QI=/data/zilu/fastrl/checkpoints/external/qwen3_1_7b
V2=$ROOT/logs/ckpt_grpo_v2curric_s3r1_normadvF_kl005_noshaping_n16_t12_rollout4_steps5850/global_step_300

# ---- B) v2 全6项:后台等 0,3 腾空 ----
( wait_free "0,3" "v2-full"
  run_eval "$V2" "0,3" "svamp,math500,cmath,gsm8k,gsmplus,cc-reserved" "" "fill_v2_gs300.log"
) &
BPID=$!

# ---- A) Qwen 双基线 gsmplus:GPU2 串行 ----
wait_free "2" "qwen-gsmplus"
run_eval "$QB" "2" "gsmplus" "$QB/final_eval_gsmplus" "fill_qwenbase_gsmplus.log"
run_eval "$QI" "2" "gsmplus" "$QI/final_eval_gsmplus" "fill_qweninstruct_gsmplus.log"

wait $BPID
echo "[$(date +%H:%M:%S)] ALL FILL JOBS DONE"
