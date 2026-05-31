from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # dotenv is optional

from llmtrain.checkpointing.config import CheckpointConfig
from llmtrain.data.config import DataConfig
from llmtrain.distributed.config import DistributedConfig
from llmtrain.models.config import ModelConfig
from llmtrain.observability.config import ObservabilityConfig
from llmtrain.preprocessing.config import PreprocessConfig
from llmtrain.tokenizer.config import TokenizerConfig
from llmtrain.training.config import TrainerConfig
from llmtrain.training.validation_config import ValidationConfig


SCHEMA_VERSION = "1.0"


class RunConfig(BaseModel):
    name: str
    output_dir: Path
    seed: int = 42

    model_config = {"extra": "forbid"}


class Config(BaseModel):
    run: RunConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    trainer: TrainerConfig
    checkpoint: CheckpointConfig
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    preprocess: PreprocessConfig | None = None

    model_config = {"extra": "forbid"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return _expand_env_vars(data)


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in strings."""
    if isinstance(obj, str):
        # Pattern: ${VAR} or ${VAR:-default}
        def replacer(match):
            var_expr = match.group(1)
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return os.environ.get(var_name.strip(), default.strip())
            else:
                var_name = var_expr.strip()
                value = os.environ.get(var_name)
                if value is None:
                    raise ValueError(f"Environment variable ${{{var_name}}} is not set")
                return value
        return re.sub(r'\$\{([^}]+)\}', replacer, obj)
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    else:
        return obj


def load_yaml_with_extends(path: str | Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    root = Path(path).expanduser().resolve()
    seen: set[Path] = set()
    chain: list[dict[str, str]] = []

    def visit(p: Path) -> dict[str, Any]:
        p = p.resolve()
        if p in seen:
            raise ValueError(f"Circular config extends detected at {p}")
        seen.add(p)
        raw = _load_yaml(p)
        merged: dict[str, Any] = {}
        for ext in raw.get("extends", []) or []:
            merged = deep_merge(merged, visit((p.parent / ext).resolve()))
        merged = deep_merge(merged, raw)
        chain.append({"path": os.path.relpath(p, root.parent), "sha256": sha256_file(p)})
        seen.remove(p)
        return merged

    return visit(root), chain


def _parse_override_value(value: str) -> Any:
    return yaml.safe_load(value)


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    out = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use k=v syntax: {item}")
        key, raw_value = item.split("=", 1)
        parts = key.split(".")
        cur = out
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
            if not isinstance(cur, dict):
                raise ValueError(f"Override path crosses non-mapping key: {key}")
        cur[parts[-1]] = _parse_override_value(raw_value)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> tuple[Config, list[dict[str, str]]]:
    data, chain = load_yaml_with_extends(path)
    data = apply_overrides(data, overrides)
    _apply_run_output_dir_defaults(data)
    return Config.model_validate(data), chain


def _apply_run_output_dir_defaults(data: dict[str, Any]) -> None:
    preprocess = data.get("preprocess")
    run = data.get("run")
    if not isinstance(preprocess, dict) or not isinstance(run, dict) or not run.get("output_dir"):
        return
    root = Path(run["output_dir"])
    data.setdefault("data", {})["manifest_path"] = root / "manifest.jsonl"
    writer = preprocess.setdefault("writer", {})
    writer["output_dir"] = root
    writer["rejected_path"] = root / "rejected.jsonl"
    dedup = preprocess.setdefault("dedup", {})
    dedup["state_dir"] = root / "dedup_state"


def config_to_yaml(cfg: Config, chain: list[dict[str, str]] | None = None) -> str:
    body = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    header = [
        f"# schema_version: {SCHEMA_VERSION}",
        f"# generated_at: {datetime.now(timezone.utc).isoformat()}",
    ]
    if chain:
        header.append("# chain:")
        for entry in chain:
            header.append(f"#   - {entry['path']} sha256:{entry['sha256']}")
    return "\n".join(header) + "\n" + body


def dump_resolved(cfg: Config, output_dir: str | Path | None = None, chain: list[dict[str, str]] | None = None) -> Path:
    out_dir = Path(output_dir or cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "config.resolved.yaml"
    content = config_to_yaml(cfg, chain)
    if target.exists():
        old = target.read_text(encoding="utf-8")
        if old == content or _resolved_body(old) == _resolved_body(content):
            return target
        backup = target.with_name(f"config.resolved.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.yaml")
        try:
            target.rename(backup)
        except FileNotFoundError:
            return target
    try:
        with target.open("x", encoding="utf-8") as fh:
            fh.write(content)
    except FileExistsError:
        pass
    return target


def _resolved_body(content: str) -> str:
    return "\n".join(line for line in content.splitlines() if not line.startswith("# generated_at:"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and optionally dump a resolved llmtrain config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--dump-resolved",
        action="store_true",
        help="Write config.resolved.yaml under run.output_dir after validation.",
    )
    args = parser.parse_args()
    cfg, chain = load_config(args.config, args.override)
    resolved_path = None
    if args.dump_resolved:
        resolved_path = dump_resolved(cfg, cfg.run.output_dir, chain)
    print(
        json.dumps(
            {
                "ok": True,
                "run": cfg.run.model_dump(mode="json"),
                "preprocess": cfg.preprocess is not None,
                "chain": chain,
                "resolved_path": str(resolved_path) if resolved_path else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
