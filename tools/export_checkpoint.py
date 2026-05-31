from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from llmtrain.checkpointing.manager import CheckpointManager
from llmtrain.distributed import cleanup_distributed, init_distributed, wrap_model
from llmtrain.distributed.wrap import unwrap_model
from llmtrain.models import build_model
from llmtrain.training.optim import build_optimizer
from llmtrain.training.schedule import TokenCosineScheduler
from llmtrain.utils.config import SCHEMA_VERSION, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a training checkpoint to a single-process inference checkpoint.")
    parser.add_argument("--config", required=True, help="Training config used for the checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Source checkpoint directory.")
    parser.add_argument("--output", required=True, help="Output checkpoint directory.")
    args = parser.parse_args()

    cfg, _ = load_config(args.config)
    dist_ctx = init_distributed(cfg.distributed)
    try:
        device = dist_ctx.device
        model = build_model(cfg.model).to(device)
        model = wrap_model(model, cfg.distributed, dist_ctx)
        optimizer = build_optimizer(model, cfg.trainer.optimizer)
        scheduler = TokenCosineScheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
        source = Path(args.checkpoint)
        meta = CheckpointManager(cfg.run.output_dir, checkpoint_format=cfg.checkpoint.format).load(
            source,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            load_rng=False,
        )
        state = _full_model_state_dict(model)
        if dist_ctx.is_main:
            output = Path(args.output)
            tmp = output.with_name(f".{output.name}.tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            torch.save({"model": state}, tmp / "state.pt")
            exported_meta = dict(meta)
            exported_meta["schema_version"] = SCHEMA_VERSION
            exported_meta["checkpoint_format"] = "torch"
            exported_meta["exported_from"] = str(source)
            torch.save(exported_meta, tmp / "meta.pt")
            (tmp / "_SUCCESS").write_text("ok\n", encoding="utf-8")
            if output.exists():
                shutil.rmtree(output)
            tmp.rename(output)
            print(f"exported checkpoint: {output}")
    finally:
        cleanup_distributed()


def _full_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    try:
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel, StateDictType
    except Exception:
        FullStateDictConfig = None
        FullyShardedDataParallel = None
        StateDictType = None

    if FullyShardedDataParallel is not None and isinstance(model, FullyShardedDataParallel):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    return {key: value.detach().cpu() for key, value in unwrap_model(model).state_dict().items()}


if __name__ == "__main__":
    main()
