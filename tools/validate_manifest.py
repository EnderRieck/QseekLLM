#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from llmtrain.data.manifest import validate_manifest
from llmtrain.utils.config import dump_resolved, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg, chain = load_config(args.config, args.override)
    dump_resolved(cfg, cfg.run.output_dir, chain)
    meta = validate_manifest(cfg.data.manifest_path, validate_shards=cfg.data.validate_hashes)
    print(meta.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
