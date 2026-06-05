#!/usr/bin/env bash
# Run the 4 curriculum checkpoints on the same prompts and build a comparison md.
# Run AFTER 16k finishes (needs the GPU free). Uses python -u (not conda run; screen
# is already in the llmtrain env, and standalone runs should `conda activate llmtrain`
# first -- see feedback memory). Single-process infer loads each DCP ckpt via no_dist.
set -uo pipefail
cd /mnt/paper2any/ziyi/llmTrain
export HF_ENDPOINT=https://hf-mirror.com

CONFIG=configs/train/stage2_cosmopedia_cpt_1700m.yaml   # same model/tokenizer for all 4
PROMPTS=inference/compare_prompts.jsonl
OUT=runs/stage_compare
mkdir -p "$OUT"

# name -> checkpoint (numeric prefix keeps md order 50B->CPT->8k->16k)
declare -A CK=(
  [1_stage1_50B]=runs/stage1_general/checkpoints/milestone_050000000000
  [2_CPT_1B]=runs/stage2_cosmopedia_cpt_1700m/checkpoints/milestone_001000000000
  [3_8k_2B]=runs/stage_ext_8k/checkpoints/milestone_002000000000
  [4_16k_3B]=runs/stage_ext_16k/checkpoints/milestone_003000000000
)

for name in $(printf '%s\n' "${!CK[@]}" | sort); do
  ck=${CK[$name]}
  if [ ! -e "$ck/_SUCCESS" ]; then echo "SKIP $name: missing $ck"; continue; fi
  echo "=== infer $name ($ck) ==="
  python -u run.py infer \
    --config "$CONFIG" --checkpoint "$ck" \
    --device cuda --dtype bf16 \
    --input-jsonl "$PROMPTS" --output-jsonl "$OUT/out_$name.jsonl" \
    --max-new-tokens 200 --greedy --no-include-prompt || echo "infer $name FAILED"
done

echo "=== build compare.md ==="
python inference/build_compare_md.py "$OUT" "$PROMPTS" > "$OUT/compare.md"
echo "wrote $OUT/compare.md"
