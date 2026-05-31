from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from llmtrain.evaluation.config import EvalConfig
from llmtrain.evaluation.model import LLMTrainHarnessLM, resolve_eval_checkpoint
from llmtrain.utils.config import load_config


@dataclass(frozen=True)
class EvalRunResult:
    output_dir: Path
    results_path: Path | None
    samples_path: Path | None
    metadata_path: Path | None
    rank: int
    world_size: int


def run_harness_eval(cfg: EvalConfig, *, checkpoint_override: str | Path | None = None) -> EvalRunResult:
    dist_ctx = _eval_dist_env()
    rank = dist_ctx["rank"]
    world_size = dist_ctx["world_size"]
    local_rank = dist_ctx["local_rank"]
    output_dir = Path(cfg.run.output_dir) / cfg.run.name
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_dataset_cache_defaults(cfg.datasets.root_dir, offline=cfg.datasets.offline)
    try:
        if "chinese_wplc" in cfg.tasks and rank == 0:
            _require_file(cfg.datasets.wplc_prepared_path, "prepared Chinese WPLC dataset")
        if "chinese_wplc" in cfg.tasks and rank != 0:
            _wait_for_file(cfg.datasets.wplc_prepared_path)
        evaluator, task_manager_cls = _import_harness()
        train_cfg, _ = load_config(cfg.model.train_config)
        checkpoint_value = checkpoint_override if checkpoint_override is not None else cfg.model.checkpoint
        checkpoint_path = resolve_eval_checkpoint(train_cfg, checkpoint_value)
        lm = LLMTrainHarnessLM.from_train_config(
            train_cfg,
            checkpoint_path=checkpoint_path,
            runtime=cfg.runtime,
            batch_size=_int_batch_size(cfg.harness.batch_size),
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
        _init_eval_process_group(cfg.runtime.device, local_rank=local_rank, world_size=world_size)
        task_manager = task_manager_cls(include_path=str(cfg.harness.include_path))
        use_cache = str(cfg.harness.use_cache) if cfg.harness.use_cache is not None else None
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=cfg.tasks,
            num_fewshot=cfg.harness.num_fewshot,
            batch_size=cfg.harness.batch_size,
            max_batch_size=cfg.harness.max_batch_size,
            device=str(lm.device),
            use_cache=use_cache,
            cache_requests=cfg.harness.cache_requests,
            limit=cfg.harness.limit,
            bootstrap_iters=cfg.harness.bootstrap_iters,
            log_samples=cfg.harness.log_samples,
            task_manager=task_manager,
            random_seed=cfg.run.seed,
            numpy_random_seed=cfg.run.seed,
            torch_random_seed=cfg.run.seed,
            fewshot_random_seed=cfg.run.seed,
            verbosity=cfg.harness.verbosity,
        )
        if rank != 0 or results is None:
            return EvalRunResult(output_dir=output_dir, results_path=None, samples_path=None, metadata_path=None, rank=rank, world_size=world_size)
        metrics, samples = _split_eval_results(results)
        results_path = output_dir / "results.json"
        samples_path = output_dir / "samples.json" if samples is not None else None
        metadata_path = output_dir / "metadata.json"
        _write_json(results_path, metrics)
        if samples_path is not None:
            _write_json(samples_path, samples)
        _write_json(
            metadata_path,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "eval_config": cfg.model_dump(mode="json"),
                "train_config": str(cfg.model.train_config),
                "checkpoint": str(checkpoint_path),
                "world_size": world_size,
                "tasks": cfg.tasks,
                "results_path": str(results_path),
                "samples_path": str(samples_path) if samples_path is not None else None,
            },
        )
        return EvalRunResult(output_dir=output_dir, results_path=results_path, samples_path=samples_path, metadata_path=metadata_path, rank=rank, world_size=world_size)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _import_harness() -> tuple[Any, Any]:
    try:
        from lm_eval import evaluator
        from lm_eval.tasks import TaskManager
    except Exception as exc:
        raise ImportError("lm-evaluation-harness is required for eval. Install with: pip install -e '.[eval]'") from exc
    return evaluator, TaskManager


def _eval_dist_env() -> dict[str, int]:
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    }


def _init_eval_process_group(device: str, *, local_rank: int, world_size: int) -> None:
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device != "cpu" and torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)


def _wait_for_file(path: str | Path, *, timeout_seconds: int = 300) -> None:
    target = Path(path).expanduser()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if target.exists():
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for prepared eval dataset: {target}")


def _require_file(path: str | Path, description: str) -> None:
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Missing {description}: {target}")


def _int_batch_size(value: int | str) -> int:
    if isinstance(value, int):
        return value
    if value.isdigit():
        return int(value)
    if value.startswith("auto"):
        return 1
    raise ValueError(f"llmtrain eval adapter requires an integer batch_size, got {value!r}")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _split_eval_results(results: dict[str, Any]) -> tuple[dict[str, Any], Any | None]:
    metrics = dict(results)
    samples = metrics.pop("samples", None)
    return metrics, samples


def _set_dataset_cache_defaults(root_dir: str | Path, *, offline: bool) -> None:
    root = Path(root_dir).expanduser()
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "huggingface" / "datasets"))
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
