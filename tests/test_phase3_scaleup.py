from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from llmtrain.data.manifest import inspect_shard, write_manifest
from llmtrain.data.readers import ShardReader
from llmtrain.distributed.config import DistributedConfig
from llmtrain.distributed.env import init_distributed
from llmtrain.distributed.wrap import configure_model_for_training, wrap_model
from llmtrain.models import build_model
from llmtrain.models.config import ModelConfig
from tools.reshard_data_state import reshard_data_state


def test_activation_checkpointing_flag_is_applied():
    model = build_model(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
    )
    configure_model_for_training(model, DistributedConfig(activation_checkpointing=True, activation_checkpointing_interval=2))
    assert model.use_activation_checkpointing is True
    assert model.activation_checkpointing_interval == 2
    out = model(torch.tensor([[1, 2, 3, 4]]), document_ids=torch.tensor([[0, 0, 0, 0]]))
    assert out.loss is not None
    out.loss.backward()


def test_ddp_backend_does_not_wrap_without_torchrun():
    ctx = init_distributed(DistributedConfig(backend="ddp"))
    model = build_model(
        ModelConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=8,
        )
    )
    wrapped = wrap_model(model, DistributedConfig(backend="ddp"), ctx)
    assert wrapped is model
    assert ctx.world_size == 1


def test_parquet_reader_uses_configurable_batch_size(tmp_path):
    path = tmp_path / "records.parquet"
    rows = [
        {"id": f"r{i}", "text": f"text {i}", "source": "s", "domain": "d", "language": "en", "metadata": {"quality_score": 1.0}}
        for i in range(5)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    manifest = write_manifest([inspect_shard(path, source="s", domain="d", language="en")], tmp_path)
    reader = ShardReader(manifest.manifest, validate_hashes=False, parquet_batch_size=2)
    assert reader.parquet_batch_size == 2
    assert [record.id for record in reader] == [f"r{i}" for i in range(5)]


def test_reshard_data_state_requires_explicit_reset(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [
        {"id": f"r{i}", "text": f"text {i}", "source": "s", "domain": "d", "language": "en", "metadata": {}}
        for i in range(4)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = write_manifest([inspect_shard(path, source="s", domain="d", language="en")], tmp_path)
    state = {
        "manifest_hash": manifest.meta.read_text(encoding="utf-8"),
        "world_size": 1,
        "num_workers": 1,
        "packing": {"buffer_ids": [1], "buffer_doc_ids": [0]},
    }
    meta = json.loads(manifest.meta.read_text(encoding="utf-8"))
    state["manifest_hash"] = meta["manifest_sha256"]
    with pytest.raises(ValueError, match="reset-positions"):
        reshard_data_state(state, manifest_path=manifest.manifest, world_size=2, num_workers=1)
    out = reshard_data_state(state, manifest_path=manifest.manifest, world_size=2, num_workers=1, reset_positions=True)
    assert out["world_size"] == 2
    assert len(out["rank_states"]) == 2
    assert out["packing"] == state["packing"]
