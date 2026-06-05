#!/usr/bin/env python
"""Audit which train_340b shards a stage's checkpoint already consumed.

Thin CLI around llmtrain.data.dedup.consumed_shard_uris -- the SAME function the
training pipeline uses for automatic --init-from dedup, so this just shows you what
would be excluded. Prints a per-source breakdown; optionally writes the uri list
(e.g. for a manual manifest pre-filter, though that's no longer required).

  python tools/stage1_consumed_shards.py <milestone_dir> [--out uris.txt]
"""
import sys, argparse
from collections import defaultdict

sys.path.insert(0, "src")
from llmtrain.data.dedup import consumed_shard_uris
from llmtrain.data.manifest import load_manifest

TRAIN_MANIFEST = "/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/train_340b/manifest.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="milestone dir containing meta.pt")
    ap.add_argument("--out", default=None, help="write consumed shard uris (one per line)")
    args = ap.parse_args()

    consumed = consumed_shard_uris(args.ckpt)
    all_shards = load_manifest(TRAIN_MANIFEST)
    uri2shard = {s.uri: s for s in all_shards}

    total = defaultdict(int)
    used = defaultdict(int)
    for s in all_shards:
        total[s.source] += 1
    for u in consumed:
        used[uri2shard[u].source] += 1

    print(f"consumed shard uris: {len(consumed):,}\n")
    print(f"{'source':30s} {'consumed':>9} {'total':>9} {'%':>6}")
    for src in sorted(total, key=lambda x: -used.get(x, 0)):
        c = used.get(src, 0)
        if c == 0:
            continue
        print(f"{src:30s} {c:>9,} {total[src]:>9,} {100*c/total[src]:>5.1f}%")

    if args.out:
        with open(args.out, "w") as f:
            for u in sorted(consumed):
                f.write(u + "\n")
        print(f"\nwrote {len(consumed):,} consumed shard uris -> {args.out}")


if __name__ == "__main__":
    main()
