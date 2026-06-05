#!/usr/bin/env python
"""Backfill `init_from` into a checkpoint's meta.pt.

Checkpoints written before meta carried `init_from` (e.g. the current CPT run, whose
live process loaded the old trainer code) don't record their warm-start source, so
the recursive dedup chain (llmtrain.data.dedup.consumed_shard_uris) breaks at them.
Patch them once so downstream stages (e.g. long-context --init-from CPT) recurse
correctly through stage1 too.

  # after the CPT finishes, point every CPT checkpoint at the 50B base:
  python tools/patch_checkpoint_init_from.py \
    runs/stage1_general/checkpoints/milestone_050000000000 \
    runs/stage2_cosmopedia_cpt_1700m/checkpoints/milestone_* \
    runs/stage2_cosmopedia_cpt_1700m/checkpoints/latest

Only meta.pt is rewritten (load -> set init_from -> save); dcp/ weight shards are
untouched.
"""
import sys
from pathlib import Path

import torch


def main():
    if len(sys.argv) < 3:
        print("usage: patch_checkpoint_init_from.py <init_from_dir> <ckpt_dir> [<ckpt_dir> ...]", file=sys.stderr)
        sys.exit(1)
    init_from = sys.argv[1]
    for ckpt in sys.argv[2:]:
        meta_path = Path(ckpt) / "meta.pt"
        if not meta_path.exists():
            print(f"skip (no meta.pt): {ckpt}", file=sys.stderr)
            continue
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        old = meta.get("init_from")
        meta["init_from"] = init_from
        torch.save(meta, meta_path)
        print(f"patched {meta_path}: init_from {old!r} -> {init_from!r}")


if __name__ == "__main__":
    main()
