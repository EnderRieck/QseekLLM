from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from llmtrain.utils.config import SCHEMA_VERSION, Config


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def load_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def save_metadata_checkpoint(
    checkpoint_dir: str | Path,
    *,
    cfg: Config,
    chain: list[dict[str, str]],
    manifest_metadata: dict[str, Any],
    tokenizer_metadata: dict[str, Any],
    data_state: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    success = path / "_SUCCESS"
    if success.exists():
        success.unlink()
    meta = {
        "schema_version": SCHEMA_VERSION,
        "resolved_config": cfg.model_dump(mode="python"),
        "chain": chain,
        "manifest": manifest_metadata,
        "tokenizer": tokenizer_metadata,
        "data_state": data_state,
        "rng_state": rng_state(),
        "extra": extra or {},
    }
    torch.save(meta, path / "meta.pt")
    success.write_text("ok\n", encoding="utf-8")
    return path


def load_metadata_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_dir)
    if not (path / "_SUCCESS").exists():
        raise FileNotFoundError(f"Checkpoint is incomplete, missing _SUCCESS: {path}")
    meta = torch.load(path / "meta.pt", map_location="cpu", weights_only=False)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema_version: {meta.get('schema_version')}")
    return meta
