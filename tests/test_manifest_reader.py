import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from llmtrain.data.manifest import inspect_shard, load_manifest, validate_manifest, write_manifest
from llmtrain.data.readers import IncompatibleDataState, ShardReader
from llmtrain.data.stream import TrainingRecordStream
from tools.merge_preprocess_run import merge_preprocess_run


def write_jsonl(path: Path, source: str, domain: str, n: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "id": f"{source}/{i}",
                "text": f"{source} text {i}",
                "source": source,
                "domain": domain,
                "language": "en",
                "metadata": {"quality_score": 1.0},
            }) + "\n")


def write_parquet(path: Path, source: str, domain: str, n: int) -> None:
    rows = [{
        "id": f"{source}/{i}",
        "text": f"{source} text {i}",
        "source": source,
        "domain": domain,
        "language": "en",
        "metadata": {"quality_score": 1.0},
    } for i in range(n)]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_manifest_jsonl_parquet_hash_and_assignment(tmp_path):
    jsonl = tmp_path / "a.jsonl"
    parquet = tmp_path / "b.parquet"
    write_jsonl(jsonl, "a", "en", 3)
    write_parquet(parquet, "b", "paper", 4)
    paths = write_manifest([
        inspect_shard(jsonl, source="a", domain="en", language="en"),
        inspect_shard(parquet, source="b", domain="paper", language="en"),
    ], tmp_path)

    meta = validate_manifest(paths.manifest)
    assert meta.num_shards == 2
    assert sum(s.num_records for s in load_manifest(paths.manifest)) == 7

    ids = set()
    for worker in range(4):
        reader = ShardReader(paths.manifest, world_size=2, rank=worker // 2, num_workers=2, worker_id=worker % 2)
        for record in reader:
            assert record.id not in ids
            ids.add(record.id)
    assert len(ids) == 7

    jsonl.write_text(jsonl.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Shard hash mismatch"):
        validate_manifest(paths.manifest)


def test_reader_rejects_incompatible_state(tmp_path):
    jsonl = tmp_path / "a.jsonl"
    write_jsonl(jsonl, "a", "en", 3)
    paths = write_manifest([inspect_shard(jsonl, source="a", domain="en", language="en")], tmp_path)
    reader = ShardReader(paths.manifest, world_size=1, num_workers=1)
    state = reader.state_dict()
    state["world_size"] = 2
    with pytest.raises(IncompatibleDataState, match="world_size mismatch"):
        reader.load_state_dict(state)
    state = reader.state_dict()
    state["rank"] = 1
    with pytest.raises(IncompatibleDataState, match="rank mismatch"):
        reader.load_state_dict(state)
    state = reader.state_dict()
    state["worker_id"] = 1
    with pytest.raises(IncompatibleDataState, match="worker_id mismatch"):
        reader.load_state_dict(state)


def test_reader_allows_hash_validation_toggle(tmp_path):
    jsonl = tmp_path / "a.jsonl"
    write_jsonl(jsonl, "a", "en", 3)
    paths = write_manifest([inspect_shard(jsonl, source="a", domain="en", language="en")], tmp_path)
    reader = ShardReader(paths.manifest, validate_hashes=False)
    assert len(list(reader)) == 3


def test_reader_supports_domain_filter(tmp_path):
    zh = tmp_path / "zh.jsonl"
    en = tmp_path / "en.jsonl"
    write_jsonl(zh, "zh_source", "zh", 2)
    write_jsonl(en, "en_source", "en", 3)
    paths = write_manifest(
        [
            inspect_shard(zh, source="zh_source", domain="zh", language="zh"),
            inspect_shard(en, source="en_source", domain="en", language="en"),
        ],
        tmp_path,
    )
    reader = ShardReader(paths.manifest, domain_filter="zh", validate_hashes=False)
    assert all(record.domain == "zh" for record in reader)
    assert len(list(ShardReader(paths.manifest, domain_filter="en", validate_hashes=False))) == 3


def test_reader_supports_source_filter_with_domain_filter(tmp_path):
    zhwiki = tmp_path / "zhwiki.jsonl"
    enwiki = tmp_path / "enwiki.jsonl"
    write_jsonl(zhwiki, "zhwiki", "wiki", 2)
    write_jsonl(enwiki, "enwiki", "wiki", 3)
    paths = write_manifest(
        [
            inspect_shard(zhwiki, source="zhwiki", domain="wiki", language="zh"),
            inspect_shard(enwiki, source="enwiki", domain="wiki", language="en"),
        ],
        tmp_path,
    )
    reader = ShardReader(paths.manifest, source_filter="zhwiki", domain_filter="wiki", validate_hashes=False)
    records = list(reader)
    assert len(records) == 2
    assert {record.source for record in records} == {"zhwiki"}


def test_training_stream_passes_source_filter(tmp_path):
    zhwiki = tmp_path / "zhwiki.jsonl"
    enwiki = tmp_path / "enwiki.jsonl"
    write_jsonl(zhwiki, "zhwiki", "wiki", 2)
    write_jsonl(enwiki, "enwiki", "wiki", 3)
    paths = write_manifest(
        [
            inspect_shard(zhwiki, source="zhwiki", domain="wiki", language="zh"),
            inspect_shard(enwiki, source="enwiki", domain="wiki", language="en"),
        ],
        tmp_path,
    )
    stream = TrainingRecordStream(
        manifest_path=paths.manifest,
        sources=[
            {"name": "zhwiki", "domain": "wiki", "language": "zh", "source_filter": "zhwiki", "weight": 1.0}
        ],
        pipeline_cfg={"min_chars": 1, "normalize_whitespace": True},
        validate_hashes=False,
        shuffle_seed=42,
    )
    records = list(stream)
    assert len(records) == 2
    assert {record.source for record in records} == {"zhwiki"}


def test_merge_preprocess_run_copies_shards_and_rewrites_manifest(tmp_path):
    base = tmp_path / "base"
    incoming = tmp_path / "incoming"
    base_shards = base / "shards" / "part_00000"
    incoming_shards = incoming / "shards" / "part_00000"
    base_shards.mkdir(parents=True)
    incoming_shards.mkdir(parents=True)
    base_jsonl = base_shards / "base.jsonl"
    incoming_jsonl = incoming_shards / "incoming.jsonl"
    write_jsonl(base_jsonl, "base", "en", 2)
    write_jsonl(incoming_jsonl, "stack", "code", 3)
    write_manifest([inspect_shard(base_jsonl, source="base", domain="en", language="en")], base)
    write_manifest([inspect_shard(incoming_jsonl, source="stack", domain="code", language="multi")], incoming)

    result = merge_preprocess_run(base_run=base, incoming_run=incoming, subdir="stack_v1")

    merged = load_manifest(base / "manifest.jsonl")
    assert len(merged) == 2
    copied = base / "shards" / "stack_v1" / "part_00000" / "incoming.jsonl"
    assert copied.exists()
    assert any(shard.uri == str(copied.resolve()) for shard in merged)
    assert Path(result["backup_manifest"]).exists()
    assert Path(result["rewritten_incoming_manifest"]).exists()
    assert validate_manifest(base / "manifest.jsonl", validate_shards=True).num_shards == 2
