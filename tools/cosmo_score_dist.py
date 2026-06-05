#!/usr/bin/env python
"""Cosmopedia score distribution + per-threshold token budget.

Scans the preprocessed chinese_cosmopedia shards, reads metadata.score and text
length, and reports how many docs / estimated tokens survive each quality cutoff.
est_tokens = bytes/4 ; real ~= 0.772 * est (audited ratio for hf_byte_bpe_150k).
"""
import json, glob, sys, bisect

RUN = "/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/stream_preprocess_chinese_cosmopedia"
REAL_RATIO = 0.772
THRESHOLDS = [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.90]

stride = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # 1 = all shards
shards = sorted(glob.glob(f"{RUN}/shards/part_*/clean_*.jsonl"))[::stride]

scores = []          # for percentiles
# per-threshold accumulators
n_docs = 0
tot_est = 0
th_docs = [0] * len(THRESHOLDS)
th_est = [0] * len(THRESHOLDS)

for si, path in enumerate(shards):
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            sc = d.get("metadata", {}).get("score")
            if sc is None:
                continue
            est = len(d.get("text", "").encode("utf-8")) // 4
            n_docs += 1
            tot_est += est
            scores.append(sc)
            for i, t in enumerate(THRESHOLDS):
                if sc >= t:
                    th_docs[i] += 1
                    th_est[i] += est
    if (si + 1) % 100 == 0:
        print(f"  ...{si+1}/{len(shards)} shards", file=sys.stderr)

scale = stride  # extrapolate to full corpus when sampling
scores.sort()
def pct(p):
    return scores[min(len(scores) - 1, int(p / 100 * len(scores)))]

print(f"\nshards scanned: {len(shards)} (stride={stride})  docs sampled: {n_docs:,}")
print(f"score: min={scores[0]:.3f} p10={pct(10):.3f} p25={pct(25):.3f} "
      f"p50={pct(50):.3f} p75={pct(75):.3f} p90={pct(90):.3f} "
      f"p95={pct(95):.3f} p99={pct(99):.3f} max={scores[-1]:.3f} "
      f"mean={sum(scores)/len(scores):.4f}")

print(f"\nTotal (scaled x{scale}): docs={n_docs*scale:,}  "
      f"est={tot_est*scale/1e9:.2f}B  real~={tot_est*scale*REAL_RATIO/1e9:.2f}B")
print(f"\n{'thresh':>7} {'%docs':>7} {'docs':>14} {'est_tok':>10} {'real_tok':>10}")
for i, t in enumerate(THRESHOLDS):
    print(f"{t:>7.2f} {100*th_docs[i]/n_docs:>6.1f}% {th_docs[i]*scale:>14,} "
          f"{th_est[i]*scale/1e9:>8.2f}B {th_est[i]*scale*REAL_RATIO/1e9:>8.2f}B")
