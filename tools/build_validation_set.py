"""Pick a held-out validation manifest from the training manifest.

For each source:
  1. Replay training shard assignment + per-slot shuffle
  2. Pick one shard from the END of slot 0 (smallest collision risk)
  3. Slice [0, K] of that shard for val (K = records covering ~target_tokens_per_source real)
  4. Remainder [K, num_records] stays in the training pool

Outputs:
  <out_dir>/manifest.jsonl                — val: 1 entry per source, record_start=0, record_end=K
  <out_dir>/train_minus_val/manifest.jsonl — intermediate: original master manifest minus
                                              val [0,K) slices, with [K,end) slices substituted.
                                              Feed this to prune_manifest.py to get the final
                                              training manifest.
  <out_dir>/selection.json                — audit log

Recommended workflow:
  python tools/build_validation_set.py \
      --manifest .../stream_preprocess_parquet_zstd/manifest.jsonl \
      --out-dir .../val_60m
  python tools/prune_manifest.py \
      --manifest .../val_60m/train_minus_val/manifest.jsonl \
      --force-keep-val-manifest .../val_60m/manifest.jsonl \
      --out-dir .../train_340b
  # Then keep .../val_60m/{manifest,manifest.meta,selection}.json,
  # delete .../val_60m/train_minus_val/ (consumed by prune).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmtrain.data.manifest import (  # noqa: E402
    ShardInfo,
    deterministic_assignment,
    load_manifest,
    write_manifest,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument(
        "--target-tokens-per-source",
        type=float,
        default=6e6,
        help="Approximate real tokens per source for val (slice the chosen shard to match).",
    )
    p.add_argument(
        "--bytes-to-real-tokens",
        type=float,
        default=0.4,
        help="Real-token-per-byte conversion (real ≈ bytes * this).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = load_manifest(args.manifest)

    # Group by source
    by_source: dict[str, list[ShardInfo]] = defaultdict(list)
    for s in shards:
        by_source[s.source].append(s)

    val_entries: list[ShardInfo] = []
    train_replacements: dict[str, ShardInfo] = {}  # sha256 -> replacement train slice
    audit: list[dict] = []

    target_real = args.target_tokens_per_source

    for src in sorted(by_source):
        slist = by_source[src]
        # Replay slot assignment + shuffle
        slot_shards: dict[int, list[ShardInfo]] = defaultdict(list)
        for s in slist:
            slot_shards[deterministic_assignment(s, args.world_size, args.num_workers)].append(s)
        for slot in slot_shards:
            rng = random.Random(args.shuffle_seed)
            rng.shuffle(slot_shards[slot])

        # Pick the LAST shard of the FIRST slot (held farthest from training reads)
        first_slot = sorted(slot_shards.keys())[0]
        chosen = slot_shards[first_slot][-1]

        # Determine how many records to take for val
        num_records = chosen.num_records
        avg_real_per_record = (chosen.bytes * args.bytes_to_real_tokens) / max(1, num_records)
        K = max(1, min(num_records, int(math.ceil(target_real / max(1.0, avg_real_per_record)))))
        val_real_estimate = int(K * avg_real_per_record)

        # Build val entry: same shard, sliced [0, K)
        val_entry = chosen.model_copy(update={"record_start": 0, "record_end": K})
        # Build train remainder [K, num_records); skip if K == num_records (val took everything)
        if K < num_records:
            train_entry = chosen.model_copy(update={"record_start": K, "record_end": None})
            train_replacements[chosen.sha256] = train_entry
        else:
            train_replacements[chosen.sha256] = None  # signal: drop this shard from train

        val_entries.append(val_entry)
        audit.append({
            "source": src,
            "shard_id": chosen.id,
            "shard_uri": chosen.uri,
            "shard_sha256": chosen.sha256,
            "shard_num_records": num_records,
            "shard_bytes": chosen.bytes,
            "shard_estimated_tokens": chosen.estimated_tokens,
            "avg_real_tokens_per_record": avg_real_per_record,
            "val_records_taken": K,
            "val_real_tokens_estimate": val_real_estimate,
            "train_remainder_records": num_records - K,
        })
        print(
            f"{src:<28} shard={chosen.id:<35} N={num_records:>7} "
            f"K={K:>6} (val ~{val_real_estimate/1e6:.1f}M real)  "
            f"train_remainder={num_records - K:>6}"
        )

    # 1) Write val manifest
    write_manifest(val_entries, out_dir, manifest_version="0.1.0-val")

    # 2) Write train manifest = original − chosen shards + replacement slices
    train_shards: list[ShardInfo] = []
    for s in shards:
        if s.sha256 in train_replacements:
            replacement = train_replacements[s.sha256]
            if replacement is not None:
                train_shards.append(replacement)
            # else: drop entirely (val took everything; rare for sane targets)
        else:
            train_shards.append(s)
    train_out = out_dir / "train_minus_val"
    write_manifest(train_shards, train_out, manifest_version="0.1.0-train-minus-val")

    (out_dir / "selection.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "summary": audit,
                "total_val_entries": len(val_entries),
                "total_val_real_tokens_estimate": sum(a["val_real_tokens_estimate"] for a in audit),
                "total_train_entries": len(train_shards),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nval manifest: {out_dir/'manifest.jsonl'} ({len(val_entries)} entries)")
    print(f"train-minus-val manifest: {train_out/'manifest.jsonl'} ({len(train_shards)} entries)")
    print(f"audit: {out_dir/'selection.json'}")


if __name__ == "__main__":
    main()
