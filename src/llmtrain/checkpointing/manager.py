from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist

try:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_state_dict,
        set_model_state_dict,
        set_state_dict,
    )
except Exception:  # pragma: no cover - depends on torch build
    dcp = None
    StateDictOptions = None
    get_state_dict = None
    set_state_dict = None
    get_model_state_dict = None
    set_model_state_dict = None

from llmtrain.checkpointing.io import load_rng_state, rng_state
from llmtrain.distributed.env import checkpoint_barrier, checkpoint_process_group
from llmtrain.distributed.wrap import unwrap_model
from llmtrain.utils.config import SCHEMA_VERSION, Config

CheckpointFormat = Literal["auto", "torch", "dcp"]
ResolvedCheckpointFormat = Literal["torch", "dcp"]


class CheckpointManager:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        keep_latest: int = 3,
        checkpoint_format: CheckpointFormat = "auto",
    ) -> None:
        self.root = Path(output_dir) / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_latest = keep_latest
        self.checkpoint_format = checkpoint_format

    def latest_checkpoint(self) -> Path | None:
        latest = self.root / "latest"
        if (latest / "_SUCCESS").exists():
            return latest
        candidates = sorted(
            (p for p in self.root.glob("latest_step_*") if (p / "_SUCCESS").exists()),
            key=lambda p: p.name,
        )
        return candidates[-1] if candidates else None

    def save(
        self,
        name: str,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        trainer_state: dict[str, Any],
        data_state: dict[str, Any],
        cfg: Config,
        chain: list[dict[str, str]],
        manifest_metadata: dict[str, Any],
        tokenizer_metadata: dict[str, Any],
        metrics: dict[str, Any] | None = None,
        distributed: bool = False,
        is_main: bool = True,
        init_from: str | None = None,
    ) -> Path:
        fmt = self.resolve_format(distributed=distributed)
        target = self.root / name
        tmp = self.root / f".{name}.tmp"
        if is_main:
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
        if distributed:
            checkpoint_barrier(barrier_dir=self.root / ".barriers", tag=f"{name}.prepare")
        if fmt == "dcp":
            self._save_dcp_state(tmp, model, optimizer)
        elif is_main:
            self._save_torch_state(tmp, model, optimizer)
        else:
            raise RuntimeError("torch checkpoint save should only be called by the main rank")
        if distributed:
            checkpoint_barrier(barrier_dir=self.root / ".barriers", tag=f"{name}.state_saved")
        meta = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_format": fmt,
            "scheduler": scheduler.state_dict(),
            "trainer_state": trainer_state,
            "data_state": data_state,
            "rng_state": rng_state(),
            "resolved_config": cfg.model_dump(mode="python"),
            "chain": chain,
            "tokenizer": tokenizer_metadata,
            "manifest": manifest_metadata,
            "metrics": metrics or {},
            # Warm-start source (--init-from), if any. Lets dedup recurse the init-from
            # chain (consumed_shard_uris) and lets resume recover the dedup source
            # without re-passing --init-from. None for a from-scratch run.
            "init_from": init_from,
        }
        if is_main:
            torch.save(meta, tmp / "meta.pt")
            (tmp / "_SUCCESS").write_text("ok\n", encoding="utf-8")
            if target.exists():
                shutil.rmtree(target)
            tmp.rename(target)
            if name.startswith("latest_step_"):
                self.cleanup_latest()
        if distributed:
            checkpoint_barrier(barrier_dir=self.root / ".barriers", tag=f"{name}.complete")
        return target

    def load(
        self,
        checkpoint_dir: str | Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        load_rng: bool = True,
    ) -> dict[str, Any]:
        path = Path(checkpoint_dir)
        meta = self._load_meta(path)
        fmt = self._format_from_meta(meta)
        param_group_options = _snapshot_optimizer_param_group_options(optimizer)
        if fmt == "dcp":
            self._load_dcp(path, model, optimizer=optimizer)
        else:
            self._load_torch_state(path, model, optimizer)
        _restore_optimizer_param_group_options(optimizer, param_group_options)
        scheduler.load_state_dict(meta.get("scheduler", {}))
        if load_rng and meta.get("rng_state"):
            load_rng_state(meta["rng_state"])
        return meta

    def load_model(
        self,
        checkpoint_dir: str | Path,
        *,
        model: torch.nn.Module,
        strict: bool = True,
    ) -> dict[str, Any]:
        path = Path(checkpoint_dir)
        meta = self._load_meta(path)
        fmt = self._format_from_meta(meta)
        if fmt == "dcp":
            if (path / "state.pt").exists():
                self._load_torch_model_state(path, model, strict=strict)
            else:
                self._load_dcp(path, model, strict=strict)
        else:
            self._load_torch_model_state(path, model, strict=strict)
        return meta

    def cleanup_latest(self) -> None:
        latest = sorted(
            (p for p in self.root.glob("latest_step_*") if p.is_dir() and (p / "_SUCCESS").exists()),
            key=lambda p: p.name,
        )
        for path in latest[:-self.keep_latest]:
            shutil.rmtree(path)

    def resolve_format(self, *, distributed: bool) -> ResolvedCheckpointFormat:
        fmt = "dcp" if self.checkpoint_format == "auto" and distributed else self.checkpoint_format
        if fmt == "auto":
            fmt = "torch"
        if fmt == "dcp" and dcp is None:
            raise RuntimeError("checkpoint.format=dcp requires torch.distributed.checkpoint, but it is unavailable")
        return fmt

    @staticmethod
    def _load_meta(path: Path) -> dict[str, Any]:
        if not (path / "_SUCCESS").exists():
            raise FileNotFoundError(f"Checkpoint is incomplete, missing _SUCCESS: {path}")
        meta = torch.load(path / "meta.pt", map_location="cpu", weights_only=False)
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported checkpoint schema_version: {meta.get('schema_version')}")
        return meta

    @staticmethod
    def _format_from_meta(meta: dict[str, Any]) -> ResolvedCheckpointFormat:
        fmt = meta.get("checkpoint_format", "torch")
        if fmt not in {"torch", "dcp"}:
            raise ValueError(f"Unsupported checkpoint_format: {fmt}")
        return fmt

    @staticmethod
    def _save_torch_state(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        model_state = unwrap_model(model).state_dict()
        torch.save({"model": model_state, "optimizer": optimizer.state_dict()}, path / "state.pt")

    @staticmethod
    def _save_dcp_state(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        if dcp is None or get_state_dict is None or StateDictOptions is None:
            raise RuntimeError("DCP checkpoint save requested, but torch.distributed.checkpoint is unavailable")
        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        dcp.save(
            {"model": model_state, "optimizer": optimizer_state},
            checkpoint_id=str(path / "dcp"),
            process_group=checkpoint_process_group(),
        )

    @staticmethod
    def _load_torch_state(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        state_path = path / "state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing checkpoint model/optimizer state under {path}")
        loaded = torch.load(state_path, map_location="cpu", weights_only=False)
        unwrap_model(model).load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])

    @staticmethod
    def _load_dcp(
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        strict: bool = True,
    ) -> None:
        """Load model (and optionally optimizer) from a DCP checkpoint.

        One code path for both entry points so their state-dict handling cannot drift:
          - optimizer given -> full restore (--resume-from): model + optimizer.
          - optimizer None  -> weights-only (--init-from warm start): model only,
            fresh optimizer.
        Both use the distributed-checkpoint state-dict API (get/set_state_dict and its
        model-only variant get/set_model_state_dict), which returns the DTensor layout
        matching how _save_dcp_state wrote it. The raw FSDP module.state_dict() would
        instead give 1D flat LOCAL shards under shard_grad_op + use_orig_params and
        fail with a size mismatch against the saved DTensor shapes.
        """
        if dcp is None or get_model_state_dict is None or set_model_state_dict is None or StateDictOptions is None:
            raise RuntimeError("DCP checkpoint load requested, but torch.distributed.checkpoint is unavailable")
        if optimizer is not None and (get_state_dict is None or set_state_dict is None):
            raise RuntimeError("DCP checkpoint load requested, but torch.distributed.checkpoint is unavailable")
        dcp_path = path / "dcp"
        if not dcp_path.exists():
            raise FileNotFoundError(f"Missing DCP checkpoint state under {path}")
        initialized = dist.is_available() and dist.is_initialized()
        if optimizer is not None and not initialized:
            raise RuntimeError("DCP training checkpoint load requires an initialized distributed process group")
        options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=strict)
        if optimizer is None:
            state = {"model": get_model_state_dict(model, options=options)}
        else:
            model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
            state = {"model": model_state, "optimizer": optimizer_state}
        if initialized:
            dcp.load(state, checkpoint_id=str(dcp_path), process_group=checkpoint_process_group())
        else:
            dcp.load(state, checkpoint_id=str(dcp_path), no_dist=True)
        if optimizer is None:
            set_model_state_dict(model, model_state_dict=state["model"], options=options)
        else:
            set_state_dict(
                model,
                optimizer,
                model_state_dict=state["model"],
                optim_state_dict=state["optimizer"],
                options=options,
            )

    @staticmethod
    def _load_torch_model_state(path: Path, model: torch.nn.Module, *, strict: bool = True) -> None:
        model_obj = unwrap_model(model)
        state_path = path / "state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing checkpoint model state under {path}")
        loaded = torch.load(state_path, map_location="cpu", weights_only=False)
        model_obj.load_state_dict(loaded["model"], strict=strict)


def _snapshot_optimizer_param_group_options(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]


def _restore_optimizer_param_group_options(
    optimizer: torch.optim.Optimizer,
    saved_options: list[dict[str, Any]],
) -> None:
    for index, group in enumerate(optimizer.param_groups):
        if index < len(saved_options):
            for key, value in saved_options[index].items():
                group.setdefault(key, value)
        for key, value in optimizer.defaults.items():
            group.setdefault(key, value)
