#!/bin/bash
# ============================================================================
# 对照实验 v3-wrongpool 编排器(组件③:检查点循环)。
# 每个 cycle:  RL(限版,带梯度屏蔽+入池) -> merge成HF -> 用教师错题做一轮重放SFT
#              -> merge成HF -> 下一 cycle 从该HF续。
# 教师worker(组件②)在开头后台拉起,全程消费 pending->sft_ready。
# 对照对象: v2 纯RL。设计/日志: docs/rl_wrongpool_sft_experiment_20260614.md
#
# 用法: bash RL/run_v3_wrongpool_experiment.sh
#   可调 env: N_CYCLES RL_VERSIONS_PER_CYCLE REPLAY_MAX START_HF
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd -P)

# ---- 实验配置 ----
export WRONGPOOL_DIR=${WRONGPOOL_DIR:-/data/zilu/fastrl/wrongpool_v3exp}
export WRONGPOOL_ENABLE=1
export WRONGPOOL_LUCKY_MAX=${WRONGPOOL_LUCKY_MAX:-2}
export WRONGPOOL_MASK_ALLWRONG=${WRONGPOOL_MASK_ALLWRONG:-1}
export WRONGPOOL_HOOK_PATH="$ROOT/RL/wrongpool_hook.py"

EXP=grpo_v3wrongpool
START_HF=${START_HF:-/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf}
DATA=/data/zilu/data_unified_v2/parquet
TRAIN=$DATA/train_rl_s4_v2curric.parquet
VAL=$DATA/val_rl_s4clean_fix.parquet
RL_VERSIONS_PER_CYCLE=${RL_VERSIONS_PER_CYCLE:-50}
N_CYCLES=${N_CYCLES:-6}
REPLAY_MAX=${REPLAY_MAX:-4000}
REPLAY_MIN=${REPLAY_MIN:-64}
WORKDIR=/data/zilu/fastrl/v3wrongpool
mkdir -p "$WORKDIR" "$WRONGPOOL_DIR"
ORCH_LOG="$ROOT/logs/${EXP}_orchestrator.log"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ORCH_LOG"; }

ensure_tokenizer(){  # merge 只产权重/config;tokenizer 从原始HF补齐
  local dir=$1
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model vocab.json merges.txt generation_config.json; do
    [ -f "$START_HF/$f" ] && [ ! -f "$dir/$f" ] && cp "$START_HF/$f" "$dir/" 2>/dev/null
  done
}

merge_to_hf(){  # $1=verl global_step dir  $2=target hf dir
  # verl 档结构: <step>/actor/{huggingface/,model_world_size_*.pt}; merger 要指向含 huggingface 的那层
  local src="$1"
  if [ -d "$1/actor/huggingface" ]; then src="$1/actor"
  elif [ -d "$1/huggingface" ]; then src="$1"
  else log "ERROR: no huggingface/ under $1"; return 1; fi
  log "merge $src -> $2"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${MERGE_CARD:-1} \
    python -m verl.model_merger merge --backend fsdp --local_dir "$src" --target_dir "$2" 2>&1 | tail -6 | tee -a "$ORCH_LOG"
  ensure_tokenizer "$2"
  [ -f "$2/config.json" ] || { log "ERROR: merge produced no config.json in $2"; return 1; }
}

last_ckpt(){ ls -d "$1"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1; }

# ---- 起教师worker(组件②) ----
# 限速(codex-spark 预算保护,见 docs 7节 2026-06-14 烧速事故):low档/2并发/默认60次每小时。
TEACHER_WORKERS=${TEACHER_WORKERS:-2}
TEACHER_EFFORT=${TEACHER_EFFORT:-low}
TEACHER_MAX_PER_HOUR=${TEACHER_MAX_PER_HOUR:-60}
log "start teacher worker (throttled: ${TEACHER_WORKERS}w/${TEACHER_EFFORT}/${TEACHER_MAX_PER_HOUR}per-h) -> $WRONGPOOL_DIR/teacher_worker.log"
nohup python3 RL/teacher_worker.py --pool-dir "$WRONGPOOL_DIR" \
    --workers "$TEACHER_WORKERS" --reasoning-effort "$TEACHER_EFFORT" \
    --max-per-hour "$TEACHER_MAX_PER_HOUR" --batch-size 20 \
    >> "$WRONGPOOL_DIR/teacher_worker.log" 2>&1 &
TEACHER_PID=$!
log "teacher worker pid=$TEACHER_PID"
trap '[ -n "${TEACHER_PID:-}" ] && kill $TEACHER_PID 2>/dev/null' EXIT

CUR_HF=$START_HF
for ((k=1; k<=N_CYCLES; k++)); do
  log "================ CYCLE $k / $N_CYCLES ================"
  log "RL from: $CUR_HF"
  RG=${EXP}_cycle${k}
  OUT="$ROOT/logs/ckpt_${RG}_rollout4_steps${RL_VERSIONS_PER_CYCLE}"

  # ---- RL 阶段(与 v2 同配置 + WRONGPOOL_ENABLE) ----
  RUN_GROUP=$RG CKPT=$CUR_HF TRAIN_FILES=$TRAIN VAL_FILES=$VAL \
  NORM_ADV_BY_STD=False KL_LOSS_COEF=0.005 \
  REWARD_FORMAT_BONUS=0.05 REWARD_THINK_LEN_MAX_BONUS=0.0 \
  PPO_MINI_BATCH_SIZE=32 REQUIRE_BATCHES=2 \
  TRIGGER_PARAMETER_SYNC_STEP=4 STALENESS_THRESHOLD=0.5 \
  ROLLOUT_N=16 ROLLOUT_TEMPERATURE=1.2 REF_SERVICE=True \
  TEST_FREQ=10 SAVE_FREQ=${RL_SAVE_FREQ:-40} \
  VAL_BEFORE_TRAIN=$([ $k -eq 1 ] && echo True || echo False) \
  bash RL/run_grpo_fully_async_split_a800_ref3_a4000.sh 4 $RL_VERSIONS_PER_CYCLE
  rl_rc=$?
  log "RL cycle $k exit=$rl_rc"

  STEP=$(last_ckpt "$OUT")
  [ -z "$STEP" ] && { log "ERROR: no RL ckpt in $OUT; abort"; exit 1; }
  RL_HF="$WORKDIR/cycle${k}_rl_hf"
  merge_to_hf "$OUT/global_step_$STEP" "$RL_HF" || exit 1

  # ---- 重放SFT 阶段(组件③) ----
  RP="$DATA/replay_sft_cycle${k}.parquet"
  if python3 RL/build_replay_sft_parquet.py --pool-dir "$WRONGPOOL_DIR" --out "$RP" \
        --max "$REPLAY_MAX" --min "$REPLAY_MIN"; then
    SFT_OUT="$WORKDIR/cycle${k}_sft"
    log "replay SFT: $RL_HF + $RP -> $SFT_OUT"
    MODEL_PATH=$RL_HF DATA=$RP SAVE_PATH=$SFT_OUT EXP_NAME=${RG}_replaysft \
        bash SFT/run_replay_sft.sh
    SSTEP=$(last_ckpt "$SFT_OUT")
    if [ -n "$SSTEP" ]; then
      SFT_HF="$WORKDIR/cycle${k}_sft_hf"
      merge_to_hf "$SFT_OUT/global_step_$SSTEP" "$SFT_HF" || exit 1
      CUR_HF=$SFT_HF
      log "cycle $k done: next RL from SFT'd $SFT_HF"
    else
      log "WARN: SFT produced no ckpt; continue from RL_HF"
      CUR_HF=$RL_HF
    fi
  else
    log "cycle $k: pool too small (<$REPLAY_MIN); skip SFT; continue from RL_HF"
    CUR_HF=$RL_HF
  fi
done

log "experiment done. final HF: $CUR_HF"
