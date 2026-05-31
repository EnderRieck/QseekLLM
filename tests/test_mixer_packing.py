import json
from pathlib import Path

import torch

from llmtrain.data.manifest import inspect_shard, write_manifest
from llmtrain.data.mixer import WeightedMixer
from llmtrain.data.stream import TrainingRecordStream
from llmtrain.data.packing import PackedDataIterator, block_diagonal_attention_mask, unpack_document_segments
from llmtrain.data.pipeline import RecordPipeline
from llmtrain.data.readers import ShardReader
from llmtrain.interfaces import Record


class CharTokenizer:
    eot_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if i)

    def metadata(self) -> dict:
        return {"type": "char"}


def test_mixer_ratio_tracks_weights():
    streams = {
        "big": [Record(str(i), "xxxx", "big", "en", "en", {}) for i in range(10000)],
        "small": [Record(str(i), "xxxx", "small", "en", "en", {}) for i in range(10000)],
    }
    mixer = WeightedMixer(streams, {"big": 80.0, "small": 20.0}, seed=7)
    for _, _record in zip(range(5000), mixer):
        pass
    stats = mixer.ratio_stats()
    assert abs(stats["big"] - 0.8) < 0.03
    assert abs(stats["small"] - 0.2) < 0.03


def test_training_record_stream_uses_domain_matching(tmp_path):
    zh = tmp_path / "zh.jsonl"
    rows = [
        {"id": "zh/0", "text": "你好世界", "source": "manifest_zh", "domain": "zh", "language": "zh", "metadata": {}},
        {"id": "zh/1", "text": "再见世界", "source": "manifest_zh", "domain": "zh", "language": "zh", "metadata": {}},
    ]
    zh.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    manifest = write_manifest([inspect_shard(zh, source="manifest_zh", domain="zh", language="zh")], tmp_path)

    stream = TrainingRecordStream(
        manifest_path=manifest.manifest,
        sources=[{"name": "chinese_general", "domain": "zh", "language": "zh", "weight": 1.0}],
        pipeline_cfg={"min_chars": 1, "max_chars": None, "min_quality_score": None, "normalize_whitespace": True},
        validate_hashes=False,
    )
    records = list(stream)
    assert [record.id for record in records] == ["zh/0", "zh/1"]


def test_training_record_stream_skips_zero_weight_sources(tmp_path):
    zh = tmp_path / "zh.jsonl"
    rows = [
        {"id": "zh/0", "text": "你好世界", "source": "manifest_zh", "domain": "zh", "language": "zh", "metadata": {}},
    ]
    zh.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    manifest = write_manifest([inspect_shard(zh, source="manifest_zh", domain="zh", language="zh")], tmp_path)

    stream = TrainingRecordStream(
        manifest_path=manifest.manifest,
        sources=[
            {"name": "chinese_general", "domain": "zh", "language": "zh", "weight": 1.0},
            {"name": "paper", "domain": "paper", "language": "en", "weight": 0.0},
        ],
        pipeline_cfg={"min_chars": 1, "max_chars": None, "min_quality_score": None, "normalize_whitespace": True},
        validate_hashes=False,
    )
    records = list(stream)
    assert [record.id for record in records] == ["zh/0"]


def test_packing_boundary_mask_and_resume(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [
        {"id": "r1", "text": "abcd", "source": "s", "domain": "d", "language": "en", "metadata": {}},
        {"id": "r2", "text": "efgh", "source": "s", "domain": "d", "language": "en", "metadata": {}},
        {"id": "r3", "text": "ijkl", "source": "s", "domain": "d", "language": "en", "metadata": {}},
        {"id": "r4", "text": "mnop", "source": "s", "domain": "d", "language": "en", "metadata": {}},
        {"id": "r5", "text": "qrst", "source": "s", "domain": "d", "language": "en", "metadata": {}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = write_manifest([inspect_shard(path, source="s", domain="d", language="en")], tmp_path)

    def make_iter():
        reader = ShardReader(manifest.manifest)
        records = RecordPipeline([]).apply(reader)
        packed = PackedDataIterator(
            records,
            CharTokenizer(),
            seq_len=5,
            batch_size=1,
            upstream_state_getter=reader.state_dict,
            upstream_state_loader=reader.load_state_dict,
        )
        return packed

    baseline = list(make_iter())
    first = make_iter()
    first_iter = iter(first)
    assert next(first_iter).input_ids.tolist() == baseline[0].input_ids.tolist()
    state = first.state_dict()
    resumed = make_iter()
    resumed.load_state_dict(state)
    assert [b.input_ids.tolist() for b in resumed] == [b.input_ids.tolist() for b in baseline[1:]]

    batch = baseline[0]
    mask = block_diagonal_attention_mask(batch.document_ids)
    assert mask.shape == (1, 5, 5)
    assert torch.equal(mask[0], torch.tril(mask[0]))
    segments = unpack_document_segments(batch.input_ids[0].tolist(), batch.document_ids[0].tolist(), 0)
    assert segments == [[ord(c) for c in "abcd"]]


def test_packing_prefetch_records_matches_baseline(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [
        {"id": "r1", "text": "abcd", "source": "s", "domain": "d", "language": "en", "metadata": {}},
        {"id": "r2", "text": "efgh", "source": "s", "domain": "d", "language": "en", "metadata": {}},
        {"id": "r3", "text": "ijkl", "source": "s", "domain": "d", "language": "en", "metadata": {}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = write_manifest([inspect_shard(path, source="s", domain="d", language="en")], tmp_path)

    def make_iter(prefetch_records: int):
        reader = ShardReader(manifest.manifest)
        records = RecordPipeline([]).apply(reader)
        return PackedDataIterator(
            records,
            CharTokenizer(),
            seq_len=4,
            batch_size=1,
            prefetch_records=prefetch_records,
        )

    baseline = [b.input_ids.tolist() for b in make_iter(0)]
    prefetched = [b.input_ids.tolist() for b in make_iter(8)]
    assert prefetched == baseline
