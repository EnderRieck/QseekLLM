from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist

try:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
except Exception:  # pragma: no cover - depends on torch build
    dcp = None
    StateDictOptions = None
    get_state_dict = None
    set_state_dict = None

from llmtrain.checkpointing.io import load_rng_state, rng_state
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
    ) -> Path:
        fmt = self.resolve_format(distributed=distributed)
        target = self.root / name
        tmp = self.root / f".{name}.tmp"
        if is_main:
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
        if distributed:
            dist.barrier()
        if fmt == "dcp":
            self._save_dcp_state(tmp, model, optimizer)
        elif is_main:
            self._save_torch_state(tmp, model, optimizer)
        else:
            raise RuntimeError("torch checkpoint save should only be called by the main rank")
        if distributed:
            dist.barrier()
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
            dist.barrier()
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
            self._load_dcp_state(path, model, optimizer)
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
                self._load_dcp_model_state(path, model, strict=strict)
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
    def _load_dcp_state(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        if dcp is None or get_state_dict is None or set_state_dict is None or StateDictOptions is None:
            raise RuntimeError("DCP checkpoint load requested, but torch.distributed.checkpoint is unavailable")
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("DCP training checkpoint load requires an initialized distributed process group")
        dcp_path = path / "dcp"
        if not dcp_path.exists():
            raise FileNotFoundError(f"Missing DCP checkpoint state under {path}")
        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        state = {"model": model_state, "optimizer": optimizer_state}
        dcp.load(state, checkpoint_id=str(dcp_path))
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

    @staticmethod
    def _load_dcp_model_state(path: Path, model: torch.nn.Module, *, strict: bool = True) -> None:
        if dcp is None:
            raise RuntimeError("DCP checkpoint load requested, but torch.distributed.checkpoint is unavailable")
        dcp_path = path / "dcp"
        if not dcp_path.exists():
            raise FileNotFoundError(f"Missing DCP checkpoint state under {path}")
        state = {"model": unwrap_model(model).state_dict()}
        if dist.is_available() and dist.is_initialized():
            dcp.load(state, checkpoint_id=str(dcp_path))
        else:
            dcp.load(state, checkpoint_id=str(dcp_path), no_dist=True)
        unwrap_model(model).load_state_dict(state["model"], strict=strict)


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
