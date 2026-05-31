#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from llmtrain.tokenizer.inspector import inspect_tokenizer
from llmtrain.utils.config import dump_resolved, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument(
        "--max-records-per-domain",
        type=int,
        default=None,
        help="Inspect up to this many records for each domain instead of only the first max records globally.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed used to shuffle shards for per-domain inspection.",
    )
    args = parser.parse_args()
    cfg, chain = load_config(args.config, args.override)
    dump_resolved(cfg, cfg.run.output_dir, chain)
    model_path = cfg.tokenizer.model_path
    if model_path is None:
        raise SystemExit("tokenizer.model_path is required")
    report_path = Path(cfg.run.output_dir) / "tokenizer_inspection.json"
    report = inspect_tokenizer(
        model_path,
        cfg.data.manifest_path,
        special_tokens=cfg.tokenizer.special_tokens,
        max_records=args.max_records,
        max_records_per_domain=args.max_records_per_domain,
        sample_seed=args.sample_seed,
        validate_hashes=cfg.data.validate_hashes,
        output_path=report_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
