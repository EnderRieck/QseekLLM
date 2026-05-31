import json
from pathlib import Path

from llmtrain.data.manifest import inspect_shard, write_manifest
from llmtrain.tokenizer.sampler import sample_tokenizer_corpus


def _write_jsonl(path: Path, source: str, domain: str, n: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "id": f"{source}/{i}",
                        "text": f"{source} text {i}",
                        "source": source,
                        "domain": domain,
                        "language": "en",
                        "metadata": {},
                    }
                )
                + "\n"
            )


def test_tokenizer_sampler_uses_seeded_random_shard_order(tmp_path):
    shard_a = tmp_path / "a.jsonl"
    shard_b = tmp_path / "b.jsonl"
    _write_jsonl(shard_a, "a", "en", 4)
    _write_jsonl(shard_b, "b", "en", 4)
    manifest = write_manifest(
        [
            inspect_shard(shard_a, source="a", domain="en", language="en"),
            inspect_shard(shard_b, source="b", domain="en", language="en"),
        ],
        tmp_path,
    )

    first = tmp_path / "first.txt"
    repeat = tmp_path / "repeat.txt"
    changed = tmp_path / "changed.txt"
    stats = sample_tokenizer_corpus(
        manifest.manifest,
        first,
        byte_budget=10_000,
        ratios={"en": 1.0},
        seed=1,
    )
    sample_tokenizer_corpus(
        manifest.manifest,
        repeat,
        byte_budget=10_000,
        ratios={"en": 1.0},
        seed=1,
    )
    sample_tokenizer_corpus(
        manifest.manifest,
        changed,
        byte_budget=10_000,
        ratios={"en": 1.0},
        seed=5,
    )

    assert stats["seed"] == 1
    assert stats["sampling_strategy"] == "seeded_domain_shard_shuffle"
    assert first.read_text(encoding="utf-8") == repeat.read_text(encoding="utf-8")
    assert first.read_text(encoding="utf-8") != changed.read_text(encoding="utf-8")


def test_tokenizer_sampler_writes_sharded_corpus_directory(tmp_path):
    shard = tmp_path / "a.jsonl"
    _write_jsonl(shard, "a", "en", 20)
    manifest = write_manifest([inspect_shard(shard, source="a", domain="en", language="en")], tmp_path)

    output_dir = tmp_path / "corpus"
    stats = sample_tokenizer_corpus(
        manifest.manifest,
        output_dir,
        byte_budget=10_000,
        ratios={"en": 1.0},
        seed=1,
        output_shard_bytes=64,
    )

    paths = sorted(output_dir.glob("*.txt"))
    assert len(paths) > 1
    assert stats["output_files"] == [str(path) for path in paths]
    assert stats["output_shard_bytes"] == 64
    assert all(path.stat().st_size > 0 for path in paths)


def test_tokenizer_sampler_parallel_workers_write_separate_files(tmp_path):
    shard_a = tmp_path / "a.jsonl"
    shard_b = tmp_path / "b.jsonl"
    _write_jsonl(shard_a, "a", "en", 20)
    _write_jsonl(shard_b, "b", "en", 20)
    manifest = write_manifest(
        [
            inspect_shard(shard_a, source="a", domain="en", language="en"),
            inspect_shard(shard_b, source="b", domain="en", language="en"),
        ],
        tmp_path,
    )

    output_dir = tmp_path / "parallel_corpus"
    stats = sample_tokenizer_corpus(
        manifest.manifest,
        output_dir,
        byte_budget=10_000,
        ratios={"en": 1.0},
        seed=1,
        output_shard_bytes=128,
        num_workers=2,
    )

    paths = sorted(output_dir.glob("*.txt"))
    assert stats["num_workers"] == 2
    assert {worker["worker_id"] for worker in stats["workers"]} == {0, 1}
    assert len(paths) >= 2
    assert any(path.name.startswith("part-000-") for path in paths)
    assert any(path.name.startswith("part-001-") for path in paths)
    assert stats["bytes_written"] == sum(worker["bytes_written"] for worker in stats["workers"])
