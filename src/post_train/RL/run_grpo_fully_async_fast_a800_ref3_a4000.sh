#!/bin/bash
# Fast fully-async GRPO runner:
#   actor/update: physical GPU 1 (A800)
#   ref service:  physical GPU 3 (A4000)
#   rollout:      first N GPUs from physical 4,5,6,7 (A4000)
#
# Defaults are tuned for throughput rather than sync-equivalent smoke:
#   - sync rollout weights every 4 actor updates
#   - allow deeper stale buffer
#   - enable micro-batch ref service and ready trajectory queue
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-4}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.5}
export REF_SERVICE=${REF_SERVICE:-True}
export REF_MICRO_BATCH_SIZE=${REF_MICRO_BATCH_SIZE:-16}
export REF_MICRO_BATCH_TIMEOUT_S=${REF_MICRO_BATCH_TIMEOUT_S:-0.2}
export REQUIRE_BATCHES=${REQUIRE_BATCHES:-2}
export RUN_GROUP=${RUN_GROUP:-grpo_fully_async_fast_a800_ref3_a4000_$(date +%Y%m%d_%H%M%S)}

exec bash "$SCRIPT_DIR/run_grpo_fully_async_split_a800_ref3_a4000.sh" "$@"
