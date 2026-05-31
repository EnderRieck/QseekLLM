#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from llmtrain.tokenizer.trainer import train_tokenizer
from llmtrain.utils.config import dump_resolved, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg, chain = load_config(args.config, args.override)
    dump_resolved(cfg, cfg.run.output_dir, chain)
    metadata = train_tokenizer(cfg.tokenizer)
    summary = {
        key: value
        for key, value in metadata.items()
        if key not in {"corpus_inputs", "special_token_ids"}
    }
    summary["num_corpus_inputs"] = len(metadata.get("corpus_inputs", []))
    summary["num_special_tokens"] = len(metadata.get("special_token_ids", {}))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
