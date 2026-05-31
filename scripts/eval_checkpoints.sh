#!/usr/bin/env bash
set -uo pipefail

# Edit this JSON list directly.
# Each item must contain:
#   checkpoint: checkpoint directory or alias, e.g. "latest"
#   run_name: output subdirectory name under OUTPUT_DIR
CHECKPOINTS_JSON='[
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_002000000000", "run_name": "eval_02B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_004000000000", "run_name": "eval_04B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_006000000000", "run_name": "eval_06B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_008000000000", "run_name": "eval_08B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_010000000000", "run_name": "eval_10B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_012000000000", "run_name": "eval_12B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_014000000000", "run_name": "eval_14B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_016000000000", "run_name": "eval_16B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_018000000000", "run_name": "eval_18B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_020000000000", "run_name": "eval_20B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_022000000000", "run_name": "eval_22B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_024000000000", "run_name": "eval_24B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_026000000000", "run_name": "eval_26B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_028000000000", "run_name": "eval_28B"},
  {"checkpoint": "runs/stage1_general_300m_v2_wsd_30b/checkpoints/milestone_030000000000", "run_name": "eval_30B"}
]'

CONFIG="configs/eval/default_300m_v2.yaml"
OUTPUT_DIR="runs/stage1_general_300m_v2_wsd_30b"
GPUS="all"
BATCH_SIZE="4"
TASKS=""
LIMIT=""
NUM_FEWSHOT=""
CONTINUE_ON_ERROR=1

usage() {
  cat <<'EOF'
Usage:
  scripts/eval_checkpoints.sh [--dry-run] [--help] [-- extra run.py eval args...]

Configure the checkpoint list by editing CHECKPOINTS_JSON at the top of this file.
Each JSON item must have:
  {
    "checkpoint": "runs/.../checkpoints/latest_step_1000",
    "run_name": "stage1_700m_step1000"
  }

Other defaults are also configured at the top of this file:
  CONFIG, OUTPUT_DIR, GPUS, BATCH_SIZE, TASKS, LIMIT, NUM_FEWSHOT, CONTINUE_ON_ERROR

Options:
  --dry-run      Print commands without running them.
  -h, --help     Show this help.
  --             Extra args forwarded to run.py eval.
EOF
}

dry_run=0
extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mapfile -t checkpoint_entries < <(
  CHECKPOINTS_JSON="$CHECKPOINTS_JSON" python - <<'PY'
import json
import os
import sys

try:
    rows = json.loads(os.environ["CHECKPOINTS_JSON"])
except Exception as exc:
    print(f"Invalid CHECKPOINTS_JSON: {exc}", file=sys.stderr)
    raise SystemExit(2)

if not isinstance(rows, list) or not rows:
    print("CHECKPOINTS_JSON must be a non-empty JSON list", file=sys.stderr)
    raise SystemExit(2)

for index, row in enumerate(rows):
    if not isinstance(row, dict):
        print(f"CHECKPOINTS_JSON item {index} must be an object", file=sys.stderr)
        raise SystemExit(2)
    checkpoint = row.get("checkpoint")
    run_name = row.get("run_name")
    if not checkpoint or not run_name:
        print(f"CHECKPOINTS_JSON item {index} requires checkpoint and run_name", file=sys.stderr)
        raise SystemExit(2)
    if "\t" in checkpoint or "\t" in run_name or "\n" in checkpoint or "\n" in run_name:
        print(f"CHECKPOINTS_JSON item {index} contains unsupported tab/newline", file=sys.stderr)
        raise SystemExit(2)
    print(f"{checkpoint}\t{run_name}")
PY
)

mkdir -p "$OUTPUT_DIR"
summary_path="$OUTPUT_DIR/summary.tsv"
if [[ ! -f "$summary_path" ]]; then
  printf 'checkpoint\trun_name\tstatus\tseconds\tresults_path\tsamples_path\tlog_path\n' > "$summary_path"
fi

for entry in "${checkpoint_entries[@]}"; do
  IFS=$'\t' read -r checkpoint run_name <<< "$entry"
  run_dir="$OUTPUT_DIR/$run_name"
  log_path="$run_dir/eval.log"
  mkdir -p "$run_dir"

  cmd=(python run.py eval
    --config "$CONFIG"
    --checkpoint "$checkpoint"
    --output-dir "$OUTPUT_DIR"
    --run-name "$run_name"
    --gpus "$GPUS"
    --batch-size "$BATCH_SIZE"
  )
  [[ -n "$TASKS" ]] && cmd+=(--tasks "$TASKS")
  [[ -n "$LIMIT" ]] && cmd+=(--limit "$LIMIT")
  [[ -n "$NUM_FEWSHOT" ]] && cmd+=(--num-fewshot "$NUM_FEWSHOT")
  [[ ${#extra_args[@]} -gt 0 ]] && cmd+=("${extra_args[@]}")

  echo "[$(date '+%F %T')] evaluating checkpoint: $checkpoint"
  echo "run_name: $run_name"
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  start_ts="$(date +%s)"
  status="ok"
  if [[ "$dry_run" -eq 1 ]]; then
    echo "dry-run: skipped"
  elif ! "${cmd[@]}" > "$log_path" 2>&1; then
    status="failed"
  fi
  end_ts="$(date +%s)"
  seconds=$((end_ts - start_ts))

  results_path="$run_dir/results.json"
  samples_path="$run_dir/samples.json"
  [[ -f "$results_path" ]] || results_path=""
  [[ -f "$samples_path" ]] || samples_path=""
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$checkpoint" "$run_name" "$status" "$seconds" "$results_path" "$samples_path" "$log_path" >> "$summary_path"

  if [[ "$status" != "ok" ]]; then
    echo "failed: $checkpoint"
    echo "log: $log_path"
    if [[ "$CONTINUE_ON_ERROR" -ne 1 ]]; then
      exit 1
    fi
  else
    echo "done: $checkpoint (${seconds}s)"
  fi
done

echo "summary: $summary_path"
