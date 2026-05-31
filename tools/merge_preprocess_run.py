#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: F401
from llmtrain.data.manifest import ShardInfo, load_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move one completed preprocess run's shards into another run and merge manifests."
    )
    parser.add_argument("--base-run", required=True, help="Existing preprocess run to merge into.")
    parser.add_argument("--incoming-run", required=True, help="Completed preprocess run to merge from.")
    parser.add_argument("--subdir", required=True, help="Subdirectory under <base-run>/shards for incoming shards.")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move incoming shard files instead of copying them. The incoming manifest is left untouched.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow an existing destination shard subdirectory if all target files are already present.",
    )
    parser.add_argument(
        "--skip-shard-copy",
        action="store_true",
        help="Only rewrite and merge manifests. Target shard files must already exist.",
    )
    args = parser.parse_args()

    result = merge_preprocess_run(
        base_run=Path(args.base_run),
        incoming_run=Path(args.incoming_run),
        subdir=args.subdir,
        move=args.move,
        force=args.force,
        skip_shard_copy=args.skip_shard_copy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def merge_preprocess_run(
    *,
    base_run: Path,
    incoming_run: Path,
    subdir: str,
    move: bool = False,
    force: bool = False,
    skip_shard_copy: bool = False,
) -> dict[str, str | int]:
    base_run = base_run.expanduser().resolve()
    incoming_run = incoming_run.expanduser().resolve()
    _validate_subdir(subdir)

    base_manifest = base_run / "manifest.jsonl"
    incoming_manifest = incoming_run / "manifest.jsonl"
    if not base_manifest.exists():
        raise FileNotFoundError(f"base manifest not found: {base_manifest}")
    if not incoming_manifest.exists():
        raise FileNotFoundError(f"incoming manifest not found: {incoming_manifest}")
    if not (incoming_run / "manifest.meta.json").exists():
        raise FileNotFoundError(f"incoming manifest meta not found: {incoming_run / 'manifest.meta.json'}")

    base_shards = load_manifest(base_manifest)
    incoming_shards = load_manifest(incoming_manifest)
    dest_root = base_run / "shards" / subdir
    if dest_root.exists() and any(dest_root.iterdir()) and not force:
        raise FileExistsError(f"destination is not empty; pass --force if this is a resume: {dest_root}")
    dest_root.mkdir(parents=True, exist_ok=True)

    rewritten = []
    for shard in incoming_shards:
        src = Path(shard.uri)
        if not src.exists():
            raise FileNotFoundError(f"incoming shard not found: {src}")
        rel = _relative_to_run_shards(src, incoming_run)
        dest = dest_root / rel
        if not skip_shard_copy:
            _copy_or_move(src, dest, move=move, force=force)
        elif not dest.exists():
            raise FileNotFoundError(f"target shard not found with --skip-shard-copy: {dest}")
        rewritten.append(shard.model_copy(update={"uri": str(dest.resolve())}))

    old_ids = {shard.id for shard in base_shards}
    old_uris = {shard.uri for shard in base_shards}
    for shard in rewritten:
        if shard.id in old_ids:
            raise ValueError(f"duplicate shard id after merge: {shard.id}")
        if shard.uri in old_uris:
            raise ValueError(f"duplicate shard uri after merge: {shard.uri}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_manifest = base_run / f"manifest.before_{subdir}.{timestamp}.jsonl"
    backup_meta = base_run / f"manifest.before_{subdir}.{timestamp}.meta.json"
    shutil.copy2(base_manifest, backup_manifest)
    if (base_run / "manifest.meta.json").exists():
        shutil.copy2(base_run / "manifest.meta.json", backup_meta)

    paths = write_manifest([*base_shards, *rewritten], base_run)
    rewritten_manifest = base_run / f"manifest.{subdir}.{timestamp}.jsonl"
    with rewritten_manifest.open("w", encoding="utf-8") as f:
        for shard in rewritten:
            f.write(json.dumps(shard.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "base_run": str(base_run),
        "incoming_run": str(incoming_run),
        "dest_root": str(dest_root),
        "manifest": str(paths.manifest),
        "manifest_meta": str(paths.meta),
        "backup_manifest": str(backup_manifest),
        "backup_manifest_meta": str(backup_meta),
        "rewritten_incoming_manifest": str(rewritten_manifest),
        "base_shards": len(base_shards),
        "incoming_shards": len(incoming_shards),
        "total_shards": len(base_shards) + len(incoming_shards),
    }


def _validate_subdir(subdir: str) -> None:
    path = Path(subdir)
    if path.is_absolute() or ".." in path.parts or not subdir.strip():
        raise ValueError(f"--subdir must be a relative directory name under shards/: {subdir!r}")


def _relative_to_run_shards(path: Path, run_dir: Path) -> Path:
    shards_dir = run_dir / "shards"
    try:
        return path.resolve().relative_to(shards_dir.resolve())
    except ValueError:
        return Path(path.name)


def _copy_or_move(src: Path, dest: Path, *, move: bool, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if force and dest.stat().st_size == src.stat().st_size:
            return
        raise FileExistsError(f"target shard already exists and differs or --force was not set: {dest}")
    if move:
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(src, dest)


if __name__ == "__main__":
    main()
