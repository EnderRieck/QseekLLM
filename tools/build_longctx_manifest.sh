#!/usr/bin/env bash
# Build the unified long-context training manifest:
#   train_340b pool (general sources + the_stack_v1 long code)
#   + newly-preprocessed long sources (pes2o_s2orc, webnovel_cn, longdata_zh)
# Output: runs/longctx_pool/manifest.jsonl  (referenced by configs/data/longctx_mixture.yaml)
#
# The mixture config selects/weights sources via source_filter, so it is safe for
# the merged manifest to carry every source; unused ones are simply not sampled.
set -euo pipefail

ROOT=/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain
BASE_MANIFEST="$ROOT/runs/train_340b/manifest.jsonl"
LONG_MANIFEST="$ROOT/runs/stream_preprocess_longctx/manifest.jsonl"        # peS2o + webnovel (+ old longdata_zh, dropped below)
LONG_ZH_MANIFEST="$ROOT/runs/stream_preprocess_longctx_zh/manifest.jsonl"  # longdata_zh re-run WITHOUT cci + wiki_zh
LONG_EN_MANIFEST="$ROOT/runs/stream_preprocess_longctx_en/manifest.jsonl"  # EN long diversity: redpajama_book + redpajama_cc
OUT_DIR="$ROOT/runs/longctx_pool"
OUT="$OUT_DIR/manifest.jsonl"

mkdir -p "$OUT_DIR"
[ -f "$BASE_MANIFEST" ] || { echo "MISSING base manifest: $BASE_MANIFEST" >&2; exit 1; }
[ -f "$LONG_MANIFEST" ] || { echo "MISSING long manifest (run preprocess first): $LONG_MANIFEST" >&2; exit 1; }
[ -f "$LONG_ZH_MANIFEST" ] || { echo "MISSING longdata_zh re-run manifest: $LONG_ZH_MANIFEST" >&2; exit 1; }
[ -f "$LONG_EN_MANIFEST" ] || { echo "MISSING longdata_en manifest: $LONG_EN_MANIFEST" >&2; exit 1; }

# 1) all base-pool lines (cci3_hq, fineweb_*, dolma, the_stack_v1, codenet, math, wiki, ...)
cat "$BASE_MANIFEST" >  "$OUT"
# 2) peS2o + webnovel from the main long run, but DROP its old longdata_zh (had cci + wiki_zh)
grep -v '"source": "longdata_zh"' "$LONG_MANIFEST" >> "$OUT"
# 3) the cleaned longdata_zh (8 sub-sources, no cci/wiki_zh)
cat "$LONG_ZH_MANIFEST" >> "$OUT"
# 4) EN long diversity (books + long web)
cat "$LONG_EN_MANIFEST" >> "$OUT"

echo "wrote $OUT"
wc -l "$OUT"
echo "=== source breakdown ==="
conda run -n llmtrain python - "$OUT" <<'PY'
import json, sys
from collections import defaultdict
tok=defaultdict(int); cnt=defaultdict(int)
for l in open(sys.argv[1]):
    l=l.strip()
    if not l: continue
    o=json.loads(l); s=o['source']
    tok[s]+=o.get('estimated_tokens',0); cnt[s]+=1
tot=sum(tok.values())
for s in sorted(tok,key=lambda x:-tok[x]):
    print(f'  {s:26s} {cnt[s]:6d} shards  {tok[s]/1e9:7.2f}B  {tok[s]/tot:5.1%}')
print(f'  TOTAL {tot/1e9:.1f}B tokens')
PY
