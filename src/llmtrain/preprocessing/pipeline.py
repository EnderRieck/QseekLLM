from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llmtrain.data.manifest import ShardInfo, inspect_shard, write_manifest
from llmtrain.preprocessing.cleaners import TextCleaner
from llmtrain.preprocessing.config import DedupConfig, PreprocessConfig, PreprocessSourceConfig
from llmtrain.preprocessing.dedup import StreamingDeduper
from llmtrain.preprocessing.parsers import iter_documents
from llmtrain.preprocessing.quality import HeuristicQualityScorer
from llmtrain.preprocessing.writers import RejectedWriter, RollingShardWriter
from tqdm import tqdm


STATE_SCHEMA_VERSION = "1.0"


def run_stream_preprocess(cfg: PreprocessConfig, *, resume: bool = False) -> dict[str, Any]:
    if cfg.num_workers > 1:
        from llmtrain.preprocessing.parallel import run_parallel_preprocess

        return run_parallel_preprocess(cfg, resume=resume)
    cfg.writer.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = cfg.writer.output_dir / "preprocess.state.json"
    state = _load_state(state_path) if resume and state_path.exists() else _new_state(cfg)
    cleaner = TextCleaner(cfg.cleaning)
    dedup_cfg = cfg.dedup.model_copy(update={"load_existing_state": True}) if resume else cfg.dedup
    deduper = StreamingDeduper(dedup_cfg)
    scorer = HeuristicQualityScorer(cfg.quality)
    rejected = RejectedWriter(cfg.writer.rejected_path or cfg.writer.output_dir / "rejected.jsonl")
    all_shards: list[ShardInfo] = _scan_existing_shards(cfg) if resume else []
    totals = Counter()
    per_source: dict[str, Counter] = {}

    for source in cfg.sources:
        if not source.enabled:
            continue
        source_state = state["sources"].setdefault(source.name, {"seen": 0, "stats": {}, "completed": False})
        stats = Counter(source_state.get("stats", {}))
        if resume and source_state.get("completed"):
            per_source[source.name] = stats
            totals.update(stats)
            continue
        start_seen = int(source_state.get("seen", 0)) if resume else 0
        writer = RollingShardWriter(
            cfg.writer,
            source=source.name,
            domain=source.domain,
            language=source.language,
            weight=source.weight,
            license=source.license,
            start_index=_next_shard_index(cfg, source.name) if resume else 0,
        )
        records_since_checkpoint = 0
        progress = _source_progress_bar(source, start_seen)
        try:
            for doc in iter_documents(
                source,
                skip=start_seen,
                resume_state=source_state.get("remote_parquet") if resume and source.type == "remote_parquet" else None,
            ):
                stats["seen"] += 1
                _update_source_progress(source_state, source, doc)
                records_since_checkpoint += 1
                if progress is not None:
                    progress.update(1)
                excluded_field = _excluded_metadata_field(source, doc.metadata)
                if excluded_field is not None:
                    stats[f"rejected_excluded_metadata_{excluded_field}"] += 1
                    rejected.write(doc.id, f"excluded_metadata_{excluded_field}", doc.metadata)
                    _maybe_checkpoint(cfg, state_path, state, source, stats, deduper, records_since_checkpoint)
                    _refresh_progress_postfix(progress, stats)
                    continue
                cleaned, reason = cleaner.clean(doc)
                if cleaned is None:
                    stats[f"rejected_{reason}"] += 1
                    rejected.write(doc.id, reason or "cleaning", doc.metadata)
                    _maybe_checkpoint(cfg, state_path, state, source, stats, deduper, records_since_checkpoint)
                    _refresh_progress_postfix(progress, stats)
                    continue
                decision = deduper.check(cleaned)
                if decision.duplicate:
                    stats[f"rejected_{decision.reason}"] += 1
                    rejected.write(cleaned.id, decision.reason or "duplicate", decision.metadata)
                    _maybe_checkpoint(cfg, state_path, state, source, stats, deduper, records_since_checkpoint)
                    _refresh_progress_postfix(progress, stats)
                    continue
                quality = scorer.score(cleaned)
                if quality.score < cfg.quality.min_score:
                    stats["rejected_low_quality"] += 1
                    rejected.write(cleaned.id, "low_quality", {"score": quality.score, "signals": quality.signals})
                    _maybe_checkpoint(cfg, state_path, state, source, stats, deduper, records_since_checkpoint)
                    _refresh_progress_postfix(progress, stats)
                    continue
                metadata = dict(cleaned.metadata)
                metadata.update(decision.metadata)
                metadata["quality_score"] = quality.score
                metadata["quality_signals"] = quality.signals
                writer.write(cleaned.__class__(cleaned.id, cleaned.text, cleaned.source, cleaned.domain, cleaned.language, metadata).to_record())
                stats["written"] += 1
                _maybe_checkpoint(cfg, state_path, state, source, stats, deduper, records_since_checkpoint)
                _refresh_progress_postfix(progress, stats)
        finally:
            if progress is not None:
                progress.close()
        new_shards = writer.close()
        all_shards.extend(new_shards)
        source_state["stats"] = dict(stats)
        if source.type == "remote_parquet" and source.limit is None:
            _mark_current_remote_url_completed(source_state)
        source_state["completed"] = True
        source_state["completed_at"] = datetime.now(timezone.utc).isoformat()
        state["completed_shards"].extend([s.model_dump(mode="json") for s in new_shards])
        _save_state(state_path, state)
        per_source[source.name] = stats
        totals.update(stats)

    deduper.save_state()
    manifest = write_manifest(all_shards, cfg.writer.output_dir)
    summary = {
        "output_dir": str(cfg.writer.output_dir),
        "manifest": str(manifest.manifest),
        "manifest_meta": str(manifest.meta),
        "num_shards": len(all_shards),
        "totals": dict(totals),
        "sources": {k: dict(v) for k, v in per_source.items()},
    }
    (cfg.writer.output_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _excluded_metadata_field(source: PreprocessSourceConfig, metadata: dict[str, Any]) -> str | None:
    for field, excluded_values in source.exclude_metadata_values.items():
        value = metadata.get(field)
        if value is None:
            continue
        value_text = str(value).strip().casefold()
        if value_text in {str(item).strip().casefold() for item in excluded_values}:
            return field
    return None


def _new_state(cfg: PreprocessConfig) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "writer_output_dir": str(cfg.writer.output_dir),
        "sources": {},
        "completed_shards": [],
    }


def _progress_enabled() -> bool:
    flag = os.environ.get("LLMTRAIN_PROGRESS")
    if flag is not None:
        return flag.lower() in {"1", "true", "yes", "on"}
    return sys.stderr.isatty()


def _source_progress_bar(source: PreprocessSourceConfig, initial: int) -> Any | None:
    if not _progress_enabled():
        return None
    return tqdm(
        total=source.limit,
        initial=initial,
        desc=source.name,
        unit="doc",
        dynamic_ncols=True,
        leave=True,
    )


def _refresh_progress_postfix(progress: Any | None, stats: Counter) -> None:
    if progress is None:
        return
    progress.set_postfix(
        seen=int(stats.get("seen", 0)),
        written=int(stats.get("written", 0)),
        rejected=int(sum(v for key, v in stats.items() if key.startswith("rejected_"))),
        refresh=False,
    )


def _load_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported preprocess state schema: {state.get('schema_version')}")
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _maybe_checkpoint(
    cfg: PreprocessConfig,
    state_path: Path,
    state: dict[str, Any],
    source: PreprocessSourceConfig,
    stats: Counter,
    deduper: StreamingDeduper,
    records_since_checkpoint: int,
) -> None:
    if records_since_checkpoint % cfg.checkpoint_interval_records != 0:
        return
    state["sources"].setdefault(source.name, {})["stats"] = dict(stats)
    deduper.save_state()
    _save_state(state_path, state)


def _update_source_progress(source_state: dict[str, Any], source: PreprocessSourceConfig, doc: Any) -> None:
    source_state["seen"] = int(source_state.get("seen", 0)) + 1
    if source.type != "remote_parquet":
        return
    url = doc.metadata.get("remote_url")
    if not url:
        return
    remote = source_state.setdefault("remote_parquet", {"completed_urls": []})
    completed = remote.setdefault("completed_urls", [])
    previous = remote.get("current_url")
    if previous != url:
        if previous and previous not in completed:
            completed.append(previous)
        remote["current_url"] = url
        remote["current_url_seen"] = 0
    row_index = doc.metadata.get("remote_row_index")
    if row_index is not None:
        remote["current_url_seen"] = int(row_index) + 1
    else:
        remote["current_url_seen"] = int(remote.get("current_url_seen", 0)) + 1


def _mark_current_remote_url_completed(source_state: dict[str, Any]) -> None:
    remote = source_state.get("remote_parquet") or {}
    current = remote.get("current_url")
    if not current:
        return
    completed = remote.setdefault("completed_urls", [])
    if current not in completed:
        completed.append(current)


def _scan_existing_shards(cfg: PreprocessConfig) -> list[ShardInfo]:
    shard_dir = cfg.writer.output_dir / "shards"
    if not shard_dir.exists():
        return []
    shards: list[ShardInfo] = []
    source_by_name = {source.name: source for source in cfg.sources}
    suffixes = {"jsonl", "parquet"}
    for path in sorted(p for p in shard_dir.iterdir() if p.is_file() and p.suffix.lstrip(".") in suffixes and p.stat().st_size > 0):
        source_name = _source_name_from_shard(cfg, path)
        source = source_by_name.get(source_name)
        if source is None:
            continue
        shards.append(
            inspect_shard(
                path,
                source=source.name,
                domain=source.domain,
                language=source.language,
                weight=source.weight,
                license=source.license,
            )
        )
    return shards


def _next_shard_index(cfg: PreprocessConfig, source_name: str) -> int:
    shard_dir = cfg.writer.output_dir / "shards"
    if not shard_dir.exists():
        return 0
    prefix = f"{cfg.writer.shard_prefix}_{source_name}_"
    indexes = []
    for path in shard_dir.iterdir():
        if not path.name.startswith(prefix):
            continue
        stem = path.stem
        try:
            indexes.append(int(stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return max(indexes, default=-1) + 1


def _source_name_from_shard(cfg: PreprocessConfig, path: Path) -> str:
    prefix = f"{cfg.writer.shard_prefix}_"
    stem = path.stem
    if not stem.startswith(prefix):
        return ""
    rest = stem[len(prefix):]
    return rest.rsplit("_", 1)[0]
