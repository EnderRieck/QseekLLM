#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from llmtrain.data.manifest import assigned_shards, load_manifest, validate_manifest


def reshard_data_state(
    state: dict[str, Any],
    *,
    manifest_path: str | Path,
    world_size: int,
    num_workers: int,
    reset_positions: bool = False,
) -> dict[str, Any]:
    meta = validate_manifest(manifest_path, validate_shards=False)
    if state.get("manifest_hash") and state["manifest_hash"] != meta.manifest_sha256:
        raise ValueError(f"manifest_hash mismatch: state={state['manifest_hash']} manifest={meta.manifest_sha256}")
    old_world = int(state.get("world_size", world_size))
    old_workers = int(state.get("num_workers", num_workers))
    if (old_world, old_workers) == (world_size, num_workers):
        out = dict(state)
        out["manifest_hash"] = meta.manifest_sha256
        return out
    if not reset_positions:
        raise ValueError(
            "Exact data cursor resharding is not supported; pass --reset-positions to create a "
            "new rank/worker cursor layout at shard starts while preserving mixer/packing metadata."
        )
    shards = load_manifest(manifest_path)
    rank_states = []
    for rank in range(world_size):
        worker_states = []
        for worker_id in range(num_workers):
            shard_list = assigned_shards(shards, world_size, rank, num_workers, worker_id)
            worker_states.append(
                {
                    "worker_id": worker_id,
                    "current_shard_id": shard_list[0].id if shard_list else "",
                    "shard_index": 0,
                    "shard_byte_offset": 0,
                    "consumed_records": 0,
                }
            )
        rank_states.append({"rank": rank, "worker_states": worker_states})
    out = dict(state)
    out.update(
        {
            "manifest_hash": meta.manifest_sha256,
            "world_size": world_size,
            "num_workers": num_workers,
            "rank_states": rank_states,
            "resharded_from": {"world_size": old_world, "num_workers": old_workers},
            "reshard_reset_positions": True,
        }
    )
    return out


def main() -> None:
    import _bootstrap  # noqa: F401

    parser = argparse.ArgumentParser(description="Reshard llmtrain data iterator metadata across world_size/worker changes.")
    parser.add_argument("--input", required=True, help="Input data_state JSON or checkpoint directory containing meta.pt.")
    parser.add_argument("--output", required=True, help="Output data_state JSON path.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--reset-positions", action="store_true")
    args = parser.parse_args()
    state = _load_state(Path(args.input))
    out = reshard_data_state(
        state,
        manifest_path=args.manifest,
        world_size=args.world_size,
        num_workers=args.num_workers,
        reset_positions=args.reset_positions,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "world_size": args.world_size, "num_workers": args.num_workers}, indent=2))


def _load_state(path: Path) -> dict[str, Any]:
    if path.is_dir():
        meta_path = path / "meta.pt"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing meta.pt under checkpoint directory: {path}")
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        return dict(meta.get("data_state", {}))
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
