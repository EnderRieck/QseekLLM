#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from llmtrain.tokenizer.sampler import sample_tokenizer_corpus
from llmtrain.utils.config import dump_resolved, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg, chain = load_config(args.config, args.override)
    dump_resolved(cfg, cfg.run.output_dir, chain)
    output = (
        cfg.data.tokenizer_sampling.output_dir
        or cfg.tokenizer.corpus_path
        or Path(cfg.run.output_dir) / "tokenizer_corpus"
    )
    stats = sample_tokenizer_corpus(
        cfg.data.manifest_path,
        output,
        byte_budget=cfg.data.tokenizer_sampling.byte_budget,
        ratios=cfg.data.tokenizer_sampling.ratios,
        seed=cfg.data.tokenizer_sampling.seed,
        validate_hashes=cfg.data.validate_hashes,
        output_shard_bytes=cfg.data.tokenizer_sampling.output_shard_bytes,
        num_workers=cfg.data.tokenizer_sampling.num_workers,
    )
    stats_path = Path(cfg.run.output_dir) / "tokenizer_corpus.stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
