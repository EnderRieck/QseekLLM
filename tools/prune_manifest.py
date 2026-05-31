"""Prune a training manifest to a target size, by source.

Inputs:
  --manifest          The training manifest to prune (typically train_minus_val).
  --out-dir           Output directory.
  --targets           JSON file mapping source name -> est_tokens budget (or null = keep all).

For each source, group shards by sha256 % (world_size * num_workers) slot,
shuffle each slot deterministically with shuffle_seed, then keep shards
from the HEAD of each slot until the per-slot budget is hit. Shards
already containing only a record-slice (record_start>0 or record_end set)
are kept verbatim — pruning never further trims a slice.

Outputs:
  <out_dir>/manifest.jsonl              — pruned shards
  <out_dir>/manifest.meta.json
  <out_dir>/shards_to_delete.txt        — URIs of shards safe to rm (only those with no surviving slice)
  <out_dir>/prune.json                  — per-source audit
"""
from __future__ import annotations

import argparse
import json
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


DEFAULT_TARGETS_EST = {
    "dolma": 100_000_000_000,
    "fineweb_edu_chinese_v21": 90_000_000_000,
    "the_stack_v1": 50_000_000_000,
    "cci3_hq": 28_000_000_000,
    "chinese_webtext_2": 11_000_000_000,
    # All wiki/edu/math/code-small sources kept entirely
    "fineweb_edu": None,
    "proof_pile_2": None,
    "openwebmath": None,
    "enwiki": None,
    "zhwiki": None,
    "codenet": None,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Training manifest to prune (already minus val).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument(
        "--targets-json",
        default=None,
        help="Optional JSON file overriding DEFAULT_TARGETS_EST per source.",
    )
    p.add_argument(
        "--force-keep-val-manifest",
        default=None,
        help="If set, any train shard whose (uri, sha256) matches a val shard URI "
             "will be unconditionally kept (so the [K,end) remainder slice survives "
             "even if its slot position would otherwise be pruned).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = dict(DEFAULT_TARGETS_EST)
    if args.targets_json:
        targets.update(json.loads(Path(args.targets_json).read_text()))

    all_shards = load_manifest(args.manifest)

    # URIs of shards that have a val counterpart — the train-side slice MUST survive.
    force_keep_uris: set[str] = set()
    if args.force_keep_val_manifest:
        val_shards = load_manifest(args.force_keep_val_manifest)
        force_keep_uris = {v.uri for v in val_shards}

    by_source: dict[str, list[ShardInfo]] = defaultdict(list)
    for s in all_shards:
        by_source[s.source].append(s)

    keep: list[ShardInfo] = []
    audit = []

    for src in sorted(by_source):
        slist = by_source[src]
        target = targets.get(src, None)
        total_est = sum(s.estimated_tokens for s in slist)
        total_bytes = sum(s.bytes for s in slist)

        slot_shards: dict[int, list[ShardInfo]] = defaultdict(list)
        for s in slist:
            slot_shards[deterministic_assignment(s, args.world_size, args.num_workers)].append(s)
        for slot in slot_shards:
            rng = random.Random(args.shuffle_seed)
            rng.shuffle(slot_shards[slot])

        kept_for_source: list[ShardInfo] = []
        # Force-keep any train-side slice whose URI is in force_keep_uris.
        # These are the [K,end) remainders complementing val shards.
        forced: list[ShardInfo] = []
        if force_keep_uris:
            forced = [s for s in slist if s.uri in force_keep_uris]
            kept_for_source.extend(forced)

        if target is None:
            for slot in slot_shards:
                for s in slot_shards[slot]:
                    if s not in forced:
                        kept_for_source.append(s)
        else:
            n_slots = len(slot_shards)
            per_slot_budget = target / max(1, n_slots)
            forced_est_by_slot: dict[int, int] = defaultdict(int)
            for s in forced:
                forced_est_by_slot[deterministic_assignment(s, args.world_size, args.num_workers)] += s.estimated_tokens
            for slot in slot_shards:
                cumulative = forced_est_by_slot.get(slot, 0)
                for shard in slot_shards[slot]:
                    if shard in forced:
                        continue
                    if cumulative >= per_slot_budget:
                        break
                    kept_for_source.append(shard)
                    cumulative += shard.estimated_tokens

        keep.extend(kept_for_source)
        kept_est = sum(s.estimated_tokens for s in kept_for_source)
        kept_bytes = sum(s.bytes for s in kept_for_source)
        audit.append({
            "source": src,
            "target_est_tokens": target,
            "total_entries": len(slist),
            "total_est_tokens": total_est,
            "total_bytes": total_bytes,
            "kept_entries": len(kept_for_source),
            "kept_est_tokens": kept_est,
            "kept_bytes": kept_bytes,
            "kept_real_tokens_est": int(kept_est * 1.6),
        })
        target_str = f"{target/1e9:.0f}B" if target is not None else "ALL"
        print(
            f"{src:<32} target={target_str:<8} kept={len(kept_for_source):>6}/{len(slist):<6} "
            f"est={kept_est/1e9:>6.1f}B real≈{kept_est*1.6/1e9:>6.1f}B  "
            f"bytes={kept_bytes/1e9:>7.1f}GB"
        )

    write_manifest(keep, out_dir, manifest_version="0.1.0-pruned")

    # Build delete list: URIs whose original shards are entirely dropped
    keep_uris = {s.uri for s in keep}
    drop_uris = sorted({s.uri for s in all_shards if s.uri not in keep_uris})
    (out_dir / "shards_to_delete.txt").write_text(
        "\n".join(drop_uris) + "\n", encoding="utf-8"
    )

    total_kept_bytes = sum(s.bytes for s in keep)
    total_kept_est = sum(s.estimated_tokens for s in keep)
    summary = {
        "args": vars(args),
        "per_source": audit,
        "totals": {
            "original_entries": len(all_shards),
            "kept_entries": len(keep),
            "kept_est_tokens": total_kept_est,
            "kept_real_tokens_est": int(total_kept_est * 1.6),
            "kept_bytes": total_kept_bytes,
            "kept_GB": total_kept_bytes / 1e9,
            "drop_uris_count": len(drop_uris),
        },
    }
    (out_dir / "prune.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Totals ===")
    print(f"  Kept    : {len(keep):>6} entries / {total_kept_bytes/1e9:>7.1f} GB / {total_kept_est/1e9:.1f}B est ≈ {total_kept_est*1.6/1e9:.0f}B real")
    print(f"  Drop URIs: {len(drop_uris)}")
    print(f"\nManifest written: {out_dir/'manifest.jsonl'}")
    print(f"Drop list:        {out_dir/'shards_to_delete.txt'}")


if __name__ == "__main__":
    main()
