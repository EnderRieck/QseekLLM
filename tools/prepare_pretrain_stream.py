#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from llmtrain.checkpointing.io import save_metadata_checkpoint
from llmtrain.data.manifest import validate_manifest
from llmtrain.data.packing import PackedDataIterator
from llmtrain.data.pipeline import RecordPipeline
from llmtrain.data.readers import ShardReader
from llmtrain.observability.callbacks import Phase1RunLogger
from llmtrain.tokenizer.adapter import load_tokenizer
from llmtrain.tokenizer.trainer import tokenizer_metadata
from llmtrain.utils.config import dump_resolved, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg, chain = load_config(args.config, args.override)
    dump_resolved(cfg, cfg.run.output_dir, chain)
    logger = Phase1RunLogger(cfg.run.output_dir)
    manifest_meta = validate_manifest(
        cfg.data.manifest_path,
        validate_shards=cfg.data.validate_hashes,
    ).model_dump(mode="json")
    if cfg.tokenizer.model_path is None:
        raise SystemExit("tokenizer.model_path is required")
    tokenizer = load_tokenizer(cfg.tokenizer)
    reader = ShardReader(
        cfg.data.manifest_path,
        world_size=cfg.data.reader.world_size,
        rank=cfg.data.reader.rank,
        num_workers=cfg.data.reader.num_workers,
        worker_id=cfg.data.reader.worker_id,
        validate_hashes=cfg.data.validate_hashes,
    )
    pipeline = RecordPipeline.from_config(cfg.data.pipeline)
    records = pipeline.apply(reader)
    packed = PackedDataIterator(
        records,
        tokenizer,
        seq_len=cfg.data.packing.seq_len,
        batch_size=cfg.data.packing.batch_size,
        prefetch_records=cfg.data.packing.prefetch_records,
        upstream_state_getter=reader.state_dict,
        upstream_state_loader=reader.load_state_dict,
    )
    emitted = 0
    consumed = 0
    for batch in packed:
        emitted += 1
        consumed += batch.consumed_tokens
        logger.metric(step=emitted, consumed_tokens=consumed, batch_tokens=batch.consumed_tokens)
        logger.beat(step=emitted, consumed_tokens=consumed, status="running")
        if consumed >= cfg.trainer.max_tokens:
            break
    ckpt_dir = Path(cfg.run.output_dir) / "checkpoints" / "phase1_meta"
    save_metadata_checkpoint(
        ckpt_dir,
        cfg=cfg,
        chain=chain,
        manifest_metadata=manifest_meta,
        tokenizer_metadata=tokenizer_metadata(cfg.tokenizer.model_path, cfg.tokenizer.special_tokens),
        data_state=packed.state_dict(),
        extra={"emitted_batches": emitted, "consumed_tokens": consumed},
    )
    logger.event("phase1_stream_prepared", emitted_batches=emitted, consumed_tokens=consumed, checkpoint=str(ckpt_dir))
    logger.beat(step=emitted, consumed_tokens=consumed, status="done")
    print(json.dumps({"batches": emitted, "consumed_tokens": consumed, "checkpoint": str(ckpt_dir)}, indent=2))


if __name__ == "__main__":
    main()
