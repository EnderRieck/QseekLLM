#!/bin/bash
# Sweep rollout GPU counts 1,2,3,4 for synchronous GRPO timing.
set -xeuo pipefail
cd "$(dirname "$0")/.."

NSTEP=${1:-5}
REF_CARD_FOR_NAME=${REF_CARDS_CSV:-3}
export RUN_GROUP=${RUN_GROUP:-grpo_sync_a800_refcard${REF_CARD_FOR_NAME}_a4000_sweep_$(date +%Y%m%d_%H%M%S)}

for rollout_count in 1 2 3 4; do
    bash RL/run_grpo_sync_a800_ref3_a4000_one.sh "$rollout_count" "$NSTEP"
done

python3 RL/summarize_grpo_sweep.py "$RUN_GROUP"
