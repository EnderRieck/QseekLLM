#!/usr/bin/env python
from __future__ import annotations

import argparse
import random

import _bootstrap  # noqa: F401
import numpy as np
import torch

from llmtrain.checkpointing.manager import CheckpointManager
from llmtrain.data.async_packing import AsyncPackedDataIterator
from llmtrain.data.manifest import validate_manifest
from llmtrain.data.packing import PackedDataIterator
from llmtrain.data.pipeline import RecordPipeline
from llmtrain.data.readers import ShardReader
from llmtrain.distributed import cleanup_distributed, init_distributed, wrap_model
from llmtrain.models import build_model
from llmtrain.observability.callbacks import Phase1RunLogger
from llmtrain.tokenizer.adapter import load_tokenizer
from llmtrain.training.optim import build_optimizer
from llmtrain.training.schedule import TokenCosineScheduler
from llmtrain.training.trainer import Trainer
from llmtrain.utils.config import dump_resolved, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoints/latest even if it exists.")
    args = parser.parse_args()
    cfg, chain = load_config(args.config, args.override)
    _seed_everything(cfg.run.seed)
    resolved = dump_resolved(cfg, cfg.run.output_dir, chain)
    dist_ctx = init_distributed(cfg.distributed)
    device = dist_ctx.device
    tokenizer = load_tokenizer(cfg.tokenizer)
    manifest_meta = validate_manifest(cfg.data.manifest_path, validate_shards=cfg.data.validate_hashes).model_dump(mode="json")
    reader = ShardReader(
        cfg.data.manifest_path,
        world_size=dist_ctx.world_size if dist_ctx.enabled else cfg.data.reader.world_size,
        rank=dist_ctx.rank if dist_ctx.enabled else cfg.data.reader.rank,
        num_workers=cfg.data.reader.num_workers,
        worker_id=cfg.data.reader.worker_id,
        validate_hashes=cfg.data.validate_hashes,
        shuffle_seed=cfg.data.mixer.seed,
        parquet_batch_size=cfg.data.reader.parquet_batch_size,
    )
    records = RecordPipeline.from_config(cfg.data.pipeline).apply(reader)
    if cfg.data.packing.async_tokenization:
        data_iterator = AsyncPackedDataIterator(
            manifest_path=cfg.data.manifest_path,
            tokenizer_cfg=cfg.tokenizer,
            pipeline_cfg=cfg.data.pipeline,
            seq_len=cfg.data.packing.seq_len,
            batch_size=cfg.trainer.micro_batch_size,
            producer_workers=cfg.data.packing.producer_workers,
            queue_max_batches=cfg.data.packing.queue_max_batches,
            world_size=dist_ctx.world_size if dist_ctx.enabled else cfg.data.reader.world_size,
            rank=dist_ctx.rank if dist_ctx.enabled else cfg.data.reader.rank,
            validate_hashes=cfg.data.validate_hashes,
            shuffle_seed=cfg.data.mixer.seed,
            parquet_batch_size=cfg.data.reader.parquet_batch_size,
        )
    else:
        data_iterator = PackedDataIterator(
            records,
            tokenizer,
            seq_len=cfg.data.packing.seq_len,
            batch_size=cfg.trainer.micro_batch_size,
            upstream_state_getter=reader.state_dict,
            upstream_state_loader=reader.load_state_dict,
            prefetch_records=0,
        )
    model = build_model(cfg.model).to(device)
    model = wrap_model(model, cfg.distributed, dist_ctx)
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    scheduler = TokenCosineScheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
    ckpt_manager = CheckpointManager(cfg.run.output_dir, keep_latest=cfg.checkpoint.keep_latest)
    logger = Phase1RunLogger(cfg.run.output_dir)
    resume_from = None if args.no_resume else args.resume_from
    if resume_from is None and not args.no_resume:
        latest = ckpt_manager.latest_checkpoint()
        resume_from = str(latest) if latest else None
    trainer = Trainer(
        cfg=cfg,
        chain=chain,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_iterator=data_iterator,
        checkpoint_manager=ckpt_manager,
        manifest_metadata=manifest_meta,
        tokenizer_metadata=tokenizer.metadata(),
        device=device,
        logger=logger,
        distributed=dist_ctx,
    )
    try:
        state = trainer.fit(resume_from=resume_from)
    finally:
        cleanup_distributed()
    if dist_ctx.is_main:
        print(f"training complete: resolved={resolved} step={state.global_step} tokens={state.consumed_tokens}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
