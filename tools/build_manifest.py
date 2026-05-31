#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from llmtrain.data.manifest import inspect_shard, write_manifest
from llmtrain.utils.config import dump_resolved, load_config


def expand_paths(items: list[Path]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        text = str(item)
        if any(ch in text for ch in "*?[]"):
            out.extend(Path(p) for p in sorted(glob.glob(text)))
        elif item.is_dir():
            out.extend(sorted(p for p in item.iterdir() if p.suffix.lower() in {".jsonl", ".json", ".parquet", ".pq"}))
        else:
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg, chain = load_config(args.config, args.override)
    dump_resolved(cfg, cfg.run.output_dir, chain)
    shards = []
    for source in cfg.data.sources:
        for path in expand_paths(source.paths):
            shards.append(
                inspect_shard(
                    path,
                    source=source.name,
                    domain=source.domain,
                    language=source.language,
                    weight=source.weight,
                )
            )
    paths = write_manifest(shards, Path(cfg.data.manifest_path).parent)
    print(json.dumps({"manifest": str(paths.manifest), "meta": str(paths.meta), "num_shards": len(shards)}, indent=2))


if __name__ == "__main__":
    main()
