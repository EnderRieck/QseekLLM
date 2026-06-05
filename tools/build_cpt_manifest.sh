#!/usr/bin/env bash
# Build the cosmopedia-CPT training manifest:
#   chinese_cosmopedia HQ (the CPT style target) + an anti-forgetting REPLAY set
#   pulled from the existing train_340b pool: zh real text (fineweb_edu_chinese_v21
#   + cci3_hq) + EN(fineweb_edu) + code(the_stack_v1) + math(proof_pile_2/openwebmath).
#   No re-preprocessing.
#
# ANTI-DUPLICATION is now AUTOMATIC: the replay sources come from train_340b, which
# stage1 trained on, but we DO NOT pre-filter the manifest here. When the CPT run
# warm-starts with `--init-from <stage1 ckpt>`, run.py auto-excludes the shards that
# checkpoint already consumed (llmtrain.data.dedup.consumed_shard_uris -> every
# ShardReader drops them via exclude_uris). Keeping the manifest as the FULL source
# set means it auto-adapts to whichever base checkpoint you init from (e.g. 45B vs
# the final 50B) with no manifest rebuild. Audit what would be dropped with
# tools/stage1_consumed_shards.py.
#
# Output: runs/cpt_pool/manifest.jsonl  (referenced by configs/data/cosmopedia_cpt_mixture.yaml)
set -euo pipefail

ROOT=/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain
# HQ subset: chinese_cosmopedia filtered to metadata.score >= 0.885 (top ~6%,
# 919k docs, ~1.33B real tok) -- see tools/filter_cosmo_by_score.py.
COSMO="$ROOT/runs/cosmopedia_hq_0885/manifest.jsonl"
BASE="$ROOT/runs/train_340b/manifest.jsonl"
OUT_DIR="$ROOT/runs/cpt_pool"
OUT="$OUT_DIR/manifest.jsonl"

mkdir -p "$OUT_DIR"
[ -f "$COSMO" ] || { echo "MISSING cosmopedia manifest: $COSMO" >&2; exit 1; }
[ -f "$BASE" ]  || { echo "MISSING base manifest: $BASE" >&2; exit 1; }

# 1) cosmopedia HQ (the CPT target; not in train_340b, never consumed by stage1)
cat "$COSMO" > "$OUT"

# 2) replay sources from the 340b pool (exact source-field match). FULL set --
#    stage1-consumed shards are excluded automatically at train time (see header).
grep -E '"source": "(fineweb_edu_chinese_v21|cci3_hq|fineweb_edu|the_stack_v1|proof_pile_2|openwebmath)"' "$BASE" >> "$OUT"

echo "wrote $OUT"; wc -l "$OUT"
echo "(stage1-consumed shards are excluded automatically by --init-from at train time)"

# 3) generate the sidecar manifest.meta.json -- validate_manifest() REQUIRES it and
#    checks manifest_sha256 == sha256(manifest.jsonl) + num_shards. A hand-assembled
#    manifest has no meta (only stream_preprocess emits one), so we build it here.
#    (Uses a script file, not a stdin heredoc -- `conda run python - <<EOF` silently
#    fails to pipe the heredoc, so the meta would never be written.)
echo "=== writing manifest.meta.json + breakdown ==="
conda run -n llmtrain python "$(dirname "$0")/write_manifest_meta.py" "$OUT"
