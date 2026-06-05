"""Reconstruct which shards a previous run already consumed, from its checkpoint.

Used to auto-deduplicate when warm-starting a NEW stage with --init-from: the new
run should not re-train on data the source run already saw. A reader's byte/record
offset is NOT a portable cursor (it depends on manifest contents x parallelism x
shuffle seed), so instead we rebuild the exact set of consumed shard URIs and
exclude them when building the new run's readers. URI is the stable identifier
(shard `id` is NOT unique across manifest parts).

The reconstruction replays the reader build recorded in the checkpoint's
data_state: load_manifest(reader.manifest_path) -> filter(source/domain) ->
assigned_shards(ws,rank,nw,wid) -> Random(shuffle_seed).shuffle, then takes
shards[0 .. shard_index] (the current shard is partially consumed -> excluded too).
"""
from __future__ import annotations

import random
from pathlib import Path

from llmtrain.data.manifest import assigned_shards, load_manifest


def checkpoint_init_from(checkpoint_dir: str | Path) -> str | None:
    """Return the warm-start source (--init-from) recorded in a checkpoint's meta,
    or None if absent (from-scratch run, or a checkpoint written before meta carried
    init_from). Lets a --resume-from recover its dedup source without re-passing it."""
    import torch

    meta_path = Path(checkpoint_dir) / "meta.pt"
    if not meta_path.exists():
        return None
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    return meta.get("init_from")


def consumed_shard_uris(checkpoint_dir: str | Path, *, _seen: set[str] | None = None) -> set[str]:
    """Return the shard URIs the run at `checkpoint_dir` consumed, INCLUDING its whole
    warm-start chain (recurses over meta.init_from).

    Why recurse: a warm-started run built its readers with the upstream run's shards
    EXCLUDED (ShardReader(exclude_uris=...)), so its saved shard_index counts positions
    in the EXCLUDED+shuffled list. To replay that correctly we must apply the same
    exclusion here -- which we obtain by recursing into init_from. Without it the
    rebuilt list (and its shuffle order) wouldn't match and we'd collect wrong uris.
    The result is this run's consumed shards UNION the entire upstream chain's, so a
    single consumed_shard_uris(init_from) in run.py covers every layer (stage1 -> CPT
    -> 8k -> 16k -> ...), not just one.

    Returns an empty set if the checkpoint has no usable distributed data_state.
    """
    import torch

    meta_path = Path(checkpoint_dir) / "meta.pt"
    if not meta_path.exists():
        return set()
    key = str(meta_path.resolve())
    _seen = set() if _seen is None else _seen
    if key in _seen:  # guard against a malformed/cyclic init_from chain
        return set()
    _seen = _seen | {key}

    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    # 1) recurse the warm-start chain: everything the upstream already consumed.
    init_from = meta.get("init_from")
    prior = consumed_shard_uris(init_from, _seen=_seen) if init_from else set()

    data_state = meta.get("data_state") or {}
    if data_state.get("mode") != "distributed_data_state":
        return prior

    manifest_cache: dict[str, list] = {}
    consumed: set[str] = set(prior)

    for _, rank_state in data_state.get("rank_states", {}).items():
        seed = rank_state.get("shuffle_seed")
        mpath = rank_state.get("manifest_path")  # per-rank, not per-reader
        for _, worker_state in rank_state.get("committed_worker_states", {}).items():
            readers = worker_state.get("upstream", {}).get("readers", {})
            for _, rd in readers.items():
                if mpath not in manifest_cache:
                    manifest_cache[mpath] = load_manifest(mpath)
                shards = manifest_cache[mpath]
                if rd.get("source_filter") is not None:
                    shards = [s for s in shards if s.source == rd["source_filter"]]
                if rd.get("domain_filter") is not None:
                    shards = [s for s in shards if s.domain == rd["domain_filter"]]
                # 2) apply the SAME upstream exclusion the run used at train time,
                #    before assigned_shards/shuffle -- mirrors ShardReader(exclude_uris).
                if prior:
                    shards = [s for s in shards if s.uri not in prior]
                shards = assigned_shards(
                    shards, rd["world_size"], rd["rank"], rd["num_workers"], rd["worker_id"]
                )
                if seed is not None:
                    shards = list(shards)
                    random.Random(seed).shuffle(shards)
                for i in range(min(rd["shard_index"] + 1, len(shards))):
                    consumed.add(shards[i].uri)
    return consumed
