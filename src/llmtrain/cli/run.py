from __future__ import annotations

import argparse
import json
import random
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from llmtrain.checkpointing.manager import CheckpointManager
from llmtrain.data.async_packing import AsyncPackedDataIterator
from llmtrain.data.manifest import validate_manifest
from llmtrain.data.packing import PackedDataIterator
from llmtrain.data.stream import TrainingRecordStream
from llmtrain.distributed import cleanup_distributed, init_distributed, wrap_model
from llmtrain.evaluation import load_eval_config, run_harness_eval
from llmtrain.inference import GenerationConfig, InferenceConfig, InferenceEngine
from llmtrain.models import build_model
from llmtrain.observability.callbacks import Phase1RunLogger
from llmtrain.tokenizer.adapter import load_tokenizer
from llmtrain.training.optim import build_optimizer
from llmtrain.training.schedule import TokenCosineScheduler, build_scheduler
from llmtrain.training.trainer import Trainer
from llmtrain.utils.config import Config, dump_resolved, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run.py", description="Unified llmtrain entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_train_parser(subparsers)
    _add_infer_parser(subparsers)
    _add_eval_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def _add_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Run training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoints/latest even if it exists.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training launcher device mode; cuda/auto may auto-launch torchrun.")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated visible GPU ids for training, or 'all'. Implies CUDA_VISIBLE_DEVICES selection.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backend", choices=["none", "ddp", "fsdp", "zero"], default=None)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--activation-checkpointing-interval", type=int, default=None)
    parser.add_argument("--gradient-accumulation-no-sync", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default=None)
    parser.add_argument("--fsdp-auto-wrap-policy", choices=["none", "transformer_block"], default=None)
    parser.add_argument("--fsdp-use-orig-params", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fsdp-sharding-strategy", choices=["full_shard", "shard_grad_op", "no_shard"], default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=None)
    parser.add_argument("--checkpoint-format", choices=["auto", "torch", "dcp"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-tokens", type=int, default=None)
    parser.add_argument("--decay-tokens", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--async-tokenization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--producer-workers", type=int, default=None)
    parser.add_argument("--queue-max-batches", type=int, default=None)
    parser.add_argument("--parquet-batch-size", type=int, default=None)
    parser.add_argument("--validate-hashes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mixer-temperature", type=float, default=None)
    parser.add_argument("--save-interval-minutes", type=int, default=None)
    parser.add_argument("--milestone-interval-tokens", type=int, default=None)
    parser.add_argument("--keep-latest", type=int, default=None)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=None)
    parser.set_defaults(func=run_train)


def _add_infer_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("infer", help="Run text generation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--kv-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-prompt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stop-on-eot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--output-field", default="completion")
    parser.add_argument("--include-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(func=run_infer)


def _add_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval", help="Run lm-evaluation-harness evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated harness task names. Defaults to config.tasks.")
    parser.add_argument("--limit", type=float, default=None, help="Limit examples per task; integers mean count, floats <1 mean fraction.")
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpus", default=None, help="Comma-separated visible GPU ids for eval, or 'all'.")
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log-samples", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cache-requests", action=argparse.BooleanOptionalAction, default=None)
    parser.set_defaults(func=run_eval)


def run_train(args: argparse.Namespace) -> None:
    gpu_ids = _resolve_gpu_selection(args.gpus)
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    if _should_launch_torchrun(args):
        _launch_torchrun(args)
        return
    _install_launcher_watchdog()
    cfg, chain = load_config(args.config)
    cfg = _apply_train_overrides(cfg, args)
    _seed_everything(cfg.run.seed)
    resolved = dump_resolved(cfg, cfg.run.output_dir, chain)
    dist_ctx = init_distributed(cfg.distributed)
    device = dist_ctx.device
    tokenizer = load_tokenizer(cfg.tokenizer)
    manifest_meta = validate_manifest(cfg.data.manifest_path, validate_shards=cfg.data.validate_hashes).model_dump(mode="json")
    manifest_meta_model = validate_manifest(cfg.data.manifest_path, validate_shards=cfg.data.validate_hashes)
    stream = TrainingRecordStream(
        manifest_path=cfg.data.manifest_path,
        sources=cfg.data.sources,
        pipeline_cfg=cfg.data.pipeline,
        world_size=dist_ctx.world_size if dist_ctx.enabled else cfg.data.reader.world_size,
        rank=dist_ctx.rank if dist_ctx.enabled else cfg.data.reader.rank,
        num_workers=cfg.data.reader.num_workers,
        worker_id=cfg.data.reader.worker_id,
        validate_hashes=cfg.data.validate_hashes,
        shuffle_seed=cfg.data.mixer.seed,
        parquet_batch_size=cfg.data.reader.parquet_batch_size,
        mixer_temperature=cfg.data.mixer.temperature,
        manifest_meta=manifest_meta_model,
    )
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
            sources=[source.model_dump(mode="python") for source in cfg.data.sources],
            manifest_meta=manifest_meta,
            mixer_temperature=cfg.data.mixer.temperature,
            metrics_path=Path(cfg.run.output_dir) / f"data_metrics_rank{dist_ctx.rank}.jsonl",
            metrics_interval_seconds=cfg.observability.data_metrics_interval_seconds,
            emit_metrics=cfg.observability.data_metrics_jsonl,
        )
    else:
        data_iterator = PackedDataIterator(
            iter(stream),
            tokenizer,
            seq_len=cfg.data.packing.seq_len,
            batch_size=cfg.trainer.micro_batch_size,
            upstream_state_getter=stream.state_dict,
            upstream_state_loader=stream.load_state_dict,
            prefetch_records=0,
        )
    model = build_model(cfg.model).to(device)
    model = wrap_model(model, cfg.distributed, dist_ctx)
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    scheduler = build_scheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
    ckpt_manager = CheckpointManager(
        cfg.run.output_dir,
        keep_latest=cfg.checkpoint.keep_latest,
        checkpoint_format=cfg.checkpoint.format,
    )
    logger = Phase1RunLogger(cfg.run.output_dir)
    validation_callback = None
    if cfg.validation.enabled and cfg.validation.val_manifest is not None:
        from llmtrain.evaluation.callback import ValidationCallback
        validation_callback = ValidationCallback(
            cfg=cfg.validation,
            tokenizer=tokenizer,
            device=device,
            logger=logger,
            distributed=dist_ctx,
        )
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
        validation_callback=validation_callback,
    )
    try:
        state = trainer.fit(resume_from=resume_from)
    finally:
        cleanup_distributed()
    if dist_ctx.is_main:
        print(f"training complete: resolved={resolved} step={state.global_step} tokens={state.consumed_tokens}")


def run_infer(args: argparse.Namespace) -> None:
    infer_cfg = _build_inference_config(args)
    engine = InferenceEngine.from_config_path(args.config, checkpoint_path=args.checkpoint, runtime=infer_cfg.runtime)

    if args.input_jsonl:
        _run_generation_batch(engine, args.input_jsonl, args.output_jsonl, infer_cfg, prompt_field=args.prompt_field, output_field=args.output_field, include_metadata=args.include_metadata)
        return

    prompt = _read_prompt(args.prompt, args.prompt_file)
    if args.stream:
        for step in engine.iter_generate(prompt, infer_cfg.generation):
            print(step.text, end="", flush=True)
        print()
        return

    result = engine.generate(prompt, infer_cfg.generation)
    print(result.text)


def run_eval(args: argparse.Namespace) -> None:
    gpu_ids = _resolve_gpu_selection(args.gpus)
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    if _should_launch_eval_torchrun(args):
        _launch_eval_torchrun(args)
        return
    _install_launcher_watchdog()
    cfg = _apply_eval_overrides(load_eval_config(args.config), args)
    result = run_harness_eval(cfg, checkpoint_override=args.checkpoint)
    if result.rank == 0:
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "results_path": str(result.results_path) if result.results_path else None,
                    "samples_path": str(result.samples_path) if result.samples_path else None,
                    "metadata_path": str(result.metadata_path) if result.metadata_path else None,
                    "world_size": result.world_size,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def _apply_train_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    data = cfg.model_dump(mode="python")
    if args.run_name is not None:
        data["run"]["name"] = args.run_name
    if args.output_dir is not None:
        data["run"]["output_dir"] = Path(args.output_dir)
    if args.seed is not None:
        data["run"]["seed"] = args.seed
    if args.backend is not None:
        data["distributed"]["backend"] = args.backend
    if args.activation_checkpointing is not None:
        data["distributed"]["activation_checkpointing"] = args.activation_checkpointing
    if args.activation_checkpointing_interval is not None:
        data["distributed"]["activation_checkpointing_interval"] = args.activation_checkpointing_interval
    if args.gradient_accumulation_no_sync is not None:
        data["distributed"]["gradient_accumulation_no_sync"] = args.gradient_accumulation_no_sync
    if args.compile_model is not None:
        data["distributed"]["compile_model"] = args.compile_model
    if args.compile_mode is not None:
        data["distributed"]["compile_mode"] = args.compile_mode
    if args.fsdp_auto_wrap_policy is not None:
        data["distributed"]["fsdp_auto_wrap_policy"] = args.fsdp_auto_wrap_policy
    if args.fsdp_use_orig_params is not None:
        data["distributed"]["fsdp_use_orig_params"] = args.fsdp_use_orig_params
    if args.fsdp_sharding_strategy is not None:
        data["distributed"]["fsdp_sharding_strategy"] = args.fsdp_sharding_strategy
    if args.micro_batch_size is not None:
        data["trainer"]["micro_batch_size"] = args.micro_batch_size
    if args.global_batch_size is not None:
        data["trainer"]["global_batch_size"] = args.global_batch_size
    if args.max_tokens is not None:
        data["trainer"]["max_tokens"] = args.max_tokens
    if args.max_steps is not None:
        data["trainer"]["max_steps"] = args.max_steps
    if args.checkpoint_interval_steps is not None:
        data["trainer"]["checkpoint_interval_steps"] = args.checkpoint_interval_steps
    if args.checkpoint_format is not None:
        data["checkpoint"]["format"] = args.checkpoint_format
    if args.lr is not None:
        data["trainer"]["optimizer"]["lr"] = args.lr
    if args.weight_decay is not None:
        data["trainer"]["optimizer"]["weight_decay"] = args.weight_decay
    if args.warmup_tokens is not None:
        data["trainer"]["scheduler"]["warmup_tokens"] = args.warmup_tokens
    if args.decay_tokens is not None:
        data["trainer"]["scheduler"]["decay_tokens"] = args.decay_tokens
    if args.min_lr_ratio is not None:
        data["trainer"]["scheduler"]["min_lr_ratio"] = args.min_lr_ratio
    if args.precision is not None:
        data["trainer"]["precision"] = args.precision
    if args.grad_clip is not None:
        data["trainer"]["grad_clip"] = args.grad_clip
    if args.seq_len is not None:
        data["data"]["packing"]["seq_len"] = args.seq_len
    if args.async_tokenization is not None:
        data["data"]["packing"]["async_tokenization"] = args.async_tokenization
    if args.producer_workers is not None:
        data["data"]["packing"]["producer_workers"] = args.producer_workers
    if args.queue_max_batches is not None:
        data["data"]["packing"]["queue_max_batches"] = args.queue_max_batches
    if args.parquet_batch_size is not None:
        data["data"]["reader"]["parquet_batch_size"] = args.parquet_batch_size
    if args.validate_hashes is not None:
        data["data"]["validate_hashes"] = args.validate_hashes
    if args.mixer_temperature is not None:
        data["data"]["mixer"]["temperature"] = args.mixer_temperature
    if args.save_interval_minutes is not None:
        data["checkpoint"]["save_interval_minutes"] = args.save_interval_minutes
    if args.milestone_interval_tokens is not None:
        data["checkpoint"]["milestone_interval_tokens"] = args.milestone_interval_tokens
    if args.keep_latest is not None:
        data["checkpoint"]["keep_latest"] = args.keep_latest
    if args.save_best is not None:
        data["checkpoint"]["save_best"] = args.save_best
    data["data"]["packing"]["batch_size"] = data["trainer"]["micro_batch_size"]
    return Config.model_validate(data)


def _build_inference_config(args: argparse.Namespace) -> InferenceConfig:
    cfg = InferenceConfig()
    data = cfg.model_dump(mode="python")
    data["runtime"]["device"] = args.device
    data["runtime"]["dtype"] = args.dtype
    if args.compile_model is not None:
        data["runtime"]["compile_model"] = args.compile_model
    data["generation"]["max_new_tokens"] = _maybe(args.max_new_tokens, data["generation"]["max_new_tokens"])
    data["generation"]["temperature"] = _maybe(args.temperature, data["generation"]["temperature"])
    data["generation"]["top_p"] = _maybe(args.top_p, data["generation"]["top_p"])
    data["generation"]["top_k"] = _maybe(args.top_k, data["generation"]["top_k"])
    data["generation"]["repetition_penalty"] = _maybe(args.repetition_penalty, data["generation"]["repetition_penalty"])
    data["generation"]["max_input_tokens"] = args.max_input_tokens if args.max_input_tokens is not None else data["generation"]["max_input_tokens"]
    if args.greedy:
        data["generation"]["do_sample"] = False
        data["generation"]["temperature"] = 0.0
    if args.kv_cache is not None:
        data["generation"]["use_kv_cache"] = args.kv_cache
    if args.include_prompt is not None:
        data["generation"]["include_prompt"] = args.include_prompt
    if args.stop_on_eot is not None:
        data["generation"]["stop_on_eot"] = args.stop_on_eot
    data["batch"]["prompt_field"] = args.prompt_field
    data["batch"]["output_field"] = args.output_field
    data["batch"]["include_metadata"] = args.include_metadata
    return InferenceConfig.model_validate(data)


def _apply_eval_overrides(cfg: Any, args: argparse.Namespace) -> Any:
    data = cfg.model_dump(mode="python")
    if args.tasks is not None:
        data["tasks"] = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if args.limit is not None:
        data["harness"]["limit"] = int(args.limit) if float(args.limit).is_integer() else args.limit
    if args.batch_size is not None:
        data["harness"]["batch_size"] = int(args.batch_size) if str(args.batch_size).isdigit() else args.batch_size
    if args.num_fewshot is not None:
        data["harness"]["num_fewshot"] = args.num_fewshot
    if args.output_dir is not None:
        data["run"]["output_dir"] = Path(args.output_dir)
    if args.run_name is not None:
        data["run"]["name"] = args.run_name
    data["runtime"]["device"] = args.device
    data["runtime"]["dtype"] = args.dtype
    if args.compile_model is not None:
        data["runtime"]["compile_model"] = args.compile_model
    if args.log_samples is not None:
        data["harness"]["log_samples"] = args.log_samples
    if args.cache_requests is not None:
        data["harness"]["cache_requests"] = args.cache_requests
    return type(cfg).model_validate(data)


def _run_generation_batch(
    engine: InferenceEngine,
    input_jsonl: str,
    output_jsonl: str | None,
    infer_cfg: InferenceConfig,
    *,
    prompt_field: str,
    output_field: str,
    include_metadata: bool,
) -> None:
    out_f = Path(output_jsonl).open("w", encoding="utf-8") if output_jsonl else sys.stdout
    close = output_jsonl is not None
    try:
        with Path(input_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = str(row[prompt_field])
                result = engine.generate(prompt, infer_cfg.generation)
                if include_metadata:
                    row[output_field] = result.text
                    row["_generation"] = {
                        "stop_reason": result.stop_reason,
                        "input_tokens": result.input_tokens,
                        "generated_tokens": result.generated_tokens,
                    }
                else:
                    row = {output_field: result.text}
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if close:
            out_f.close()


def _run_evaluation(engine: InferenceEngine, args: argparse.Namespace, infer_cfg: InferenceConfig) -> dict[str, Any]:
    output_handle = Path(args.output_jsonl).open("w", encoding="utf-8") if args.output_jsonl else None
    metrics: dict[str, Any] = {
        "samples": 0,
        "exact_match": None,
        "avg_input_tokens": 0.0,
        "avg_generated_tokens": 0.0,
        "wall_seconds": 0.0,
    }
    exact_matches = 0
    input_tokens = 0
    generated_tokens = 0
    started = perf_counter()
    try:
        with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                if args.max_samples is not None and index >= args.max_samples:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = str(row[args.prompt_field])
                result = engine.generate(prompt, infer_cfg.generation)
                input_tokens += result.input_tokens
                generated_tokens += result.generated_tokens
                metrics["samples"] += 1
                if args.reference_field and args.reference_field in row:
                    reference = str(row[args.reference_field])
                    predicted = result.text
                    if args.normalize_whitespace:
                        reference = " ".join(reference.split())
                        predicted = " ".join(predicted.split())
                    exact_matches += int(predicted == reference)
                if output_handle is not None:
                    row[args.output_field] = result.text
                    row["_generation"] = {
                        "stop_reason": result.stop_reason,
                        "input_tokens": result.input_tokens,
                        "generated_tokens": result.generated_tokens,
                    }
                    output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics["avg_input_tokens"] = input_tokens / metrics["samples"] if metrics["samples"] else 0.0
        metrics["avg_generated_tokens"] = generated_tokens / metrics["samples"] if metrics["samples"] else 0.0
        if args.reference_field:
            metrics["exact_match"] = exact_matches / metrics["samples"] if metrics["samples"] else 0.0
        metrics["wall_seconds"] = perf_counter() - started
        return metrics
    finally:
        if output_handle is not None:
            output_handle.close()


def _read_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if prompt is not None and prompt_file is not None:
        raise ValueError("Use only one of --prompt or --prompt-file")
    if prompt_file is not None:
        return Path(prompt_file).read_text(encoding="utf-8")
    if prompt is not None:
        return prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("Provide --prompt, --prompt-file, or stdin")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _should_launch_torchrun(args: argparse.Namespace) -> bool:
    if args.device == "cpu":
        return False
    if os.environ.get("RANK") is not None or os.environ.get("WORLD_SIZE") not in (None, "1"):
        return False
    return _resolve_train_nproc_per_node(args.device) > 1


def _should_launch_eval_torchrun(args: argparse.Namespace) -> bool:
    if args.device == "cpu":
        return False
    if os.environ.get("RANK") is not None or os.environ.get("WORLD_SIZE") not in (None, "1"):
        return False
    return _resolve_train_nproc_per_node(args.device) > 1


def _resolve_train_nproc_per_node(device_mode: str) -> int:
    if device_mode == "cpu":
        return 1
    if device_mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Training device 'cuda' was requested but CUDA is not available")
        return max(1, torch.cuda.device_count())
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def _resolve_gpu_selection(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    value = spec.strip()
    if not value:
        return None
    if value == "all":
        if not torch.cuda.is_available():
            raise RuntimeError("--gpus all was requested but CUDA is not available")
        return [str(i) for i in range(torch.cuda.device_count())]
    ids = [part.strip() for part in value.split(",") if part.strip()]
    if not ids:
        raise ValueError("--gpus must be 'all' or a comma-separated list like 0,1,3")
    for gpu_id in ids:
        if not gpu_id.isdigit():
            raise ValueError(f"Invalid GPU id in --gpus: {gpu_id}")
    return ids


def _launch_torchrun(args: argparse.Namespace) -> None:
    nproc = _resolve_train_nproc_per_node(args.device)
    if nproc <= 1:
        return
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "run.py"
    env = os.environ.copy()
    env["LLMTRAIN_LAUNCHER_PID"] = str(os.getpid())
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(nproc),
        str(script),
        "train",
        "--config",
        args.config,
        "--device",
        args.device,
    ]
    passthrough = [
        "--resume-from",
        args.resume_from,
        "--run-name",
        args.run_name,
        "--output-dir",
        args.output_dir,
        "--seed",
        args.seed,
        "--backend",
        args.backend,
        "--activation-checkpointing",
        args.activation_checkpointing,
        "--activation-checkpointing-interval",
        args.activation_checkpointing_interval,
        "--gradient-accumulation-no-sync",
        args.gradient_accumulation_no_sync,
        "--compile-model",
        args.compile_model,
        "--compile-mode",
        args.compile_mode,
        "--fsdp-auto-wrap-policy",
        args.fsdp_auto_wrap_policy,
        "--fsdp-use-orig-params",
        args.fsdp_use_orig_params,
        "--fsdp-sharding-strategy",
        args.fsdp_sharding_strategy,
        "--micro-batch-size",
        args.micro_batch_size,
        "--global-batch-size",
        args.global_batch_size,
        "--max-tokens",
        args.max_tokens,
        "--max-steps",
        args.max_steps,
        "--checkpoint-interval-steps",
        args.checkpoint_interval_steps,
        "--checkpoint-format",
        args.checkpoint_format,
        "--lr",
        args.lr,
        "--weight-decay",
        args.weight_decay,
        "--warmup-tokens",
        args.warmup_tokens,
        "--decay-tokens",
        args.decay_tokens,
        "--min-lr-ratio",
        args.min_lr_ratio,
        "--precision",
        args.precision,
        "--grad-clip",
        args.grad_clip,
        "--seq-len",
        args.seq_len,
        "--async-tokenization",
        args.async_tokenization,
        "--producer-workers",
        args.producer_workers,
        "--queue-max-batches",
        args.queue_max_batches,
        "--parquet-batch-size",
        args.parquet_batch_size,
        "--validate-hashes",
        args.validate_hashes,
        "--mixer-temperature",
        args.mixer_temperature,
        "--save-interval-minutes",
        args.save_interval_minutes,
        "--milestone-interval-tokens",
        args.milestone_interval_tokens,
        "--keep-latest",
        args.keep_latest,
        "--save-best",
        args.save_best,
    ]
    for i in range(0, len(passthrough), 2):
        value = passthrough[i + 1]
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(passthrough[i])
            else:
                cmd.append("--no-" + passthrough[i][2:])
            continue
        cmd.extend([passthrough[i], str(value)])
    if args.no_resume:
        cmd.append("--no-resume")
    if torch.cuda.is_available():
        print(f"[run.py] launching torchrun with nproc-per-node={nproc}", file=sys.stderr)
    raise SystemExit(_run_process_group(cmd, env=env, label="torchrun"))


def _launch_eval_torchrun(args: argparse.Namespace) -> None:
    nproc = _resolve_train_nproc_per_node(args.device)
    if nproc <= 1:
        return
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "run.py"
    env = os.environ.copy()
    env["LLMTRAIN_LAUNCHER_PID"] = str(os.getpid())
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(nproc),
        str(script),
        "eval",
        "--config",
        args.config,
        "--device",
        args.device,
    ]
    passthrough = [
        "--checkpoint",
        args.checkpoint,
        "--tasks",
        args.tasks,
        "--limit",
        args.limit,
        "--batch-size",
        args.batch_size,
        "--num-fewshot",
        args.num_fewshot,
        "--output-dir",
        args.output_dir,
        "--run-name",
        args.run_name,
        "--dtype",
        args.dtype,
        "--compile-model",
        args.compile_model,
        "--log-samples",
        args.log_samples,
        "--cache-requests",
        args.cache_requests,
    ]
    for i in range(0, len(passthrough), 2):
        value = passthrough[i + 1]
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(passthrough[i])
            else:
                cmd.append("--no-" + passthrough[i][2:])
            continue
        cmd.extend([passthrough[i], str(value)])
    if torch.cuda.is_available():
        print(f"[run.py] launching eval torchrun with nproc-per-node={nproc}", file=sys.stderr)
    raise SystemExit(_run_process_group(cmd, env=env, label="eval torchrun"))


def _run_process_group(cmd: list[str], *, env: dict[str, str], label: str) -> int:
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    terminating = False
    terminate_signum: int | None = None
    previous_handlers: dict[int, Any] = {}

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal terminating, terminate_signum
        if not terminating:
            terminating = True
            terminate_signum = signum
            signame = signal.Signals(signum).name
            print(f"[run.py] received {signame}; terminating {label} process group", file=sys.stderr, flush=True)
            _terminate_process_group(proc, signum=signum)

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    for signum in handled_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_signal)
    try:
        returncode = proc.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if proc.poll() is None:
            _terminate_process_group(proc, signum=signal.SIGTERM)
    if terminating and returncode == 0:
        return 128 + (terminate_signum or signal.SIGTERM)
    if returncode < 0:
        return 128 + abs(returncode)
    return returncode


def _terminate_process_group(proc: subprocess.Popen[Any], *, signum: int) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signum)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if proc.poll() is None:
        print("[run.py] process group did not exit; sending SIGKILL", file=sys.stderr, flush=True)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def _install_launcher_watchdog() -> None:
    launcher_pid_text = os.environ.get("LLMTRAIN_LAUNCHER_PID")
    if not launcher_pid_text:
        return
    if os.environ.get("RANK") is None and os.environ.get("WORLD_SIZE") in (None, "1"):
        return
    try:
        launcher_pid = int(launcher_pid_text)
    except ValueError:
        return
    if launcher_pid <= 1 or launcher_pid == os.getpid():
        return

    def _watch() -> None:
        while True:
            time.sleep(5.0)
            if os.getppid() == 1 or not _pid_exists(launcher_pid):
                print(
                    "[run.py] launcher process disappeared; exiting distributed worker",
                    file=sys.stderr,
                    flush=True,
                )
                os._exit(143)

    thread = threading.Thread(target=_watch, name="llmtrain-launcher-watchdog", daemon=True)
    thread.start()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
