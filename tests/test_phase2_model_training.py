from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from llmtrain.checkpointing.manager import CheckpointManager, _restore_optimizer_param_group_options
from llmtrain.data.manifest import inspect_shard, write_manifest
from llmtrain.data.async_packing import AsyncPackedDataIterator
from llmtrain.data.packing import PackedDataIterator
from llmtrain.data.pipeline import RecordPipeline
from llmtrain.data.readers import ShardReader
from llmtrain.interfaces import Batch, Record
from llmtrain.models import build_model
from llmtrain.training.config import SchedulerConfig, TrainerConfig
from llmtrain.training.optim import build_optimizer
from llmtrain.training.schedule import TokenCosineScheduler, TokenWSDScheduler
from llmtrain.training.trainer import Trainer
from llmtrain.utils.config import Config, RunConfig
from llmtrain.checkpointing.config import CheckpointConfig
from llmtrain.data.config import DataConfig
from llmtrain.tokenizer.config import TokenizerConfig
from llmtrain.models.config import ModelConfig
from llmtrain.distributed.config import DistributedConfig
from llmtrain.observability.config import ObservabilityConfig


class TinyTokenizer:
    eot_id = 0

    def encode(self, text: str) -> list[int]:
        return [1 + (ord(ch) % 7) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr((i - 1) % 7 + 65) for i in ids if i)

    def metadata(self) -> dict:
        return {"type": "tiny"}


class CaptureLogger:
    def __init__(self) -> None:
        self.metrics: list[dict] = []
        self.beats: list[dict] = []
        self.events: list[dict] = []

    def metric(self, **record):
        self.metrics.append(record)

    def beat(self, **record):
        self.beats.append(record)

    def event(self, event: str, **record):
        self.events.append({"event": event, **record})


class FakeDistributedContext:
    enabled = True
    rank = 0
    local_rank = 0
    world_size = 8
    backend = "ddp"
    device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def make_cfg(tmp_path: Path) -> Config:
    return Config(
        run=RunConfig(name="phase2", output_dir=tmp_path, seed=7),
        data=DataConfig(manifest_path=tmp_path / "manifest.jsonl", validate_hashes=False),
        tokenizer=TokenizerConfig(algorithm="hf_byte_bpe", model_path=tmp_path / "tok.json"),
        model=ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=16),
        trainer=TrainerConfig(micro_batch_size=1, global_batch_size=1, max_tokens=8, max_steps=4, checkpoint_interval_steps=1, save_final_checkpoint=True),
        checkpoint=CheckpointConfig(save_interval_minutes=999, milestone_interval_tokens=4, keep_latest=2),
        distributed=DistributedConfig(),
        observability=ObservabilityConfig(heartbeat=False, metrics_jsonl=False, events_jsonl=False, console_interval_steps=1),
    )


def test_model_forward_backward():
    cfg = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=16)
    model = build_model(cfg)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    document_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    out = model(input_ids, document_ids=document_ids)
    assert out.logits.shape == (1, 4, 32)
    assert out.loss is not None
    out.loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_model_loss_masks_shifted_pad_labels():
    logits = torch.zeros((1, 3, 5), dtype=torch.float32)
    logits[0, 0, 4] = 10.0
    logits[0, 1, 2] = 10.0
    labels = torch.tensor([[1, 3, 2]], dtype=torch.long)
    loss = build_model(
        ModelConfig(vocab_size=8, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
    ).loss(logits, labels, pad_token_id=3)
    assert loss.item() < 1.0e-3


def test_model_fused_linear_cross_entropy_matches_standard_path():
    pytest.importorskip("liger_kernel")
    if not torch.cuda.is_available():
        pytest.skip("Liger fused linear cross entropy requires CUDA tensors")
    torch.manual_seed(0)
    device = torch.device("cuda")
    base_cfg = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    standard = build_model(ModelConfig(**base_cfg, fused_linear_cross_entropy=False)).to(device).train()
    fused = build_model(ModelConfig(**base_cfg, fused_linear_cross_entropy=True)).to(device).train()
    fused.load_state_dict(standard.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long, device=device)
    document_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long, device=device)
    standard_loss = standard(input_ids, document_ids=document_ids, pad_token_id=3).loss
    fused_out = fused(input_ids, document_ids=document_ids, pad_token_id=3)
    assert standard_loss is not None
    assert fused_out.loss is not None
    assert fused_out.logits.shape[-1] == 0
    assert torch.allclose(fused_out.loss, standard_loss, atol=1.0e-4, rtol=1.0e-4)


def test_model_liger_rms_norm_and_swiglu_match_standard_path():
    pytest.importorskip("liger_kernel")
    if not torch.cuda.is_available():
        pytest.skip("Liger kernels require CUDA tensors")
    torch.manual_seed(0)
    device = torch.device("cuda")
    base_cfg = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        fused_linear_cross_entropy=False,
    )
    standard = build_model(ModelConfig(**base_cfg, liger_rms_norm=False, liger_swiglu=False)).to(device).train()
    fused = build_model(ModelConfig(**base_cfg, liger_rms_norm=True, liger_swiglu=True)).to(device).train()
    fused.load_state_dict(standard.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long, device=device)
    document_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long, device=device)
    standard_loss = standard(input_ids, document_ids=document_ids, pad_token_id=3).loss
    fused_loss = fused(input_ids, document_ids=document_ids, pad_token_id=3).loss
    assert standard_loss is not None
    assert fused_loss is not None
    assert torch.allclose(fused_loss, standard_loss, atol=2.0e-4, rtol=2.0e-4)
    fused_loss.backward()
    assert any(p.grad is not None for p in fused.parameters())


def test_model_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=16)
    model = build_model(cfg).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.inference_mode():
        full = model(input_ids).logits[:, -1, :]
        prefill = model(input_ids[:, :3], use_cache=True)
        cached = model(input_ids[:, 3:], past_key_values=prefill.past_key_values, use_cache=True).logits[:, -1, :]
    assert prefill.past_key_values is not None
    assert len(prefill.past_key_values) == cfg.num_hidden_layers
    assert torch.allclose(full, cached, atol=1.0e-5)


def test_checkpoint_manager_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path)
    model = build_model(cfg.model)
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    scheduler = TokenCosineScheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
    manager = CheckpointManager(tmp_path, keep_latest=2)
    ckpt = manager.save(
        "latest",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={"global_step": 3, "consumed_tokens": 7},
        data_state={"foo": "bar"},
        cfg=cfg,
        chain=[{"path": "x", "sha256": "y"}],
        manifest_metadata={"manifest_sha256": "abc"},
        tokenizer_metadata={"type": "tiny"},
        metrics={"loss": 1.0},
    )
    meta = manager.load(ckpt, model=model, optimizer=optimizer, scheduler=scheduler)
    assert meta["checkpoint_format"] == "torch"
    assert meta["trainer_state"]["global_step"] == 3
    assert meta["data_state"] == {"foo": "bar"}
    assert (ckpt / "_SUCCESS").exists()
    assert (ckpt / "state.pt").exists()
    fresh = build_model(cfg.model)
    model_meta = manager.load_model(ckpt, model=fresh)
    assert model_meta["trainer_state"]["global_step"] == 3


def test_checkpoint_manager_loads_dcp_model_without_process_group(tmp_path):
    pytest.importorskip("torch.distributed.checkpoint")
    cfg = make_cfg(tmp_path)
    model = build_model(cfg.model)
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    scheduler = TokenCosineScheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
    manager = CheckpointManager(tmp_path, keep_latest=2, checkpoint_format="dcp")
    ckpt = manager.save(
        "dcp",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={"global_step": 3, "consumed_tokens": 7},
        data_state={},
        cfg=cfg,
        chain=[],
        manifest_metadata={},
        tokenizer_metadata={},
    )
    fresh = build_model(cfg.model)
    meta = manager.load_model(ckpt, model=fresh)
    assert meta["checkpoint_format"] == "dcp"
    for name, tensor in model.state_dict().items():
        assert torch.equal(fresh.state_dict()[name], tensor)


def test_checkpoint_manager_loads_dcp_model_with_fsdp_metadata_without_process_group(tmp_path):
    pytest.importorskip("torch.distributed.checkpoint")
    cfg = make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"distributed": cfg.distributed.model_copy(update={"backend": "fsdp"})})
    model = build_model(cfg.model)
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    scheduler = TokenCosineScheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
    manager = CheckpointManager(tmp_path, keep_latest=2, checkpoint_format="dcp")
    ckpt = manager.save(
        "fsdp_meta_dcp",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={"global_step": 3, "consumed_tokens": 7},
        data_state={},
        cfg=cfg,
        chain=[],
        manifest_metadata={},
        tokenizer_metadata={},
    )
    fresh = build_model(cfg.model)
    meta = manager.load_model(ckpt, model=fresh)
    assert meta["resolved_config"]["distributed"]["backend"] == "fsdp"
    for name, tensor in model.state_dict().items():
        assert torch.equal(fresh.state_dict()[name], tensor)


def test_optimizer_param_group_options_are_restored_after_checkpoint_load(tmp_path):
    cfg = make_cfg(tmp_path)
    model = build_model(cfg.model)
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    saved_options = [{key: value for key, value in group.items() if key != "params"} for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group.pop("betas", None)
    _restore_optimizer_param_group_options(optimizer, saved_options)
    for group in optimizer.param_groups:
        assert group["betas"] == cfg.trainer.optimizer.betas
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    loss = model(input_ids).loss
    assert loss is not None
    loss.backward()
    optimizer.step()


def test_trainer_smoke_and_resume(tmp_path):
    rows = [
        {"id": f"r{i}", "text": f"hello {i}", "source": "s", "domain": "d", "language": "en", "metadata": {}}
        for i in range(6)
    ]
    shard = tmp_path / "data.jsonl"
    shard.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = write_manifest([inspect_shard(shard, source="s", domain="d", language="en")], tmp_path)
    cfg = make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"manifest_path": manifest.manifest})})
    reader = ShardReader(manifest.manifest, validate_hashes=False)
    packed = PackedDataIterator(
        RecordPipeline([]).apply(reader),
        TinyTokenizer(),
        seq_len=4,
        batch_size=1,
        upstream_state_getter=reader.state_dict,
        upstream_state_loader=reader.load_state_dict,
    )
    model = build_model(cfg.model)
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    scheduler = TokenCosineScheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
    manager = CheckpointManager(tmp_path, keep_latest=2)
    trainer = Trainer(
        cfg=cfg,
        chain=[{"path": "a", "sha256": "b"}],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_iterator=packed,
        checkpoint_manager=manager,
        manifest_metadata={"manifest_sha256": "abc"},
        tokenizer_metadata={"type": "tiny"},
        device=torch.device("cpu"),
        logger=None,
    )
    state = trainer.fit()
    assert state.global_step > 0
    assert (tmp_path / "checkpoints" / "latest" / "_SUCCESS").exists()


def test_trainer_saves_final_milestone_when_training_stops_on_tokens(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "trainer": cfg.trainer.model_copy(
                update={
                    "global_batch_size": 2,
                    "max_tokens": 4,
                    "checkpoint_interval_steps": None,
                }
            ),
            "checkpoint": cfg.checkpoint.model_copy(update={"milestone_interval_tokens": 4}),
        }
    )
    model = build_model(cfg.model)
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    scheduler = TokenCosineScheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
    trainer = Trainer(
        cfg=cfg,
        chain=[],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_iterator=[
            Batch(
                input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                document_ids=torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
                consumed_tokens=4,
            )
        ],
        checkpoint_manager=CheckpointManager(tmp_path, keep_latest=2),
        manifest_metadata={},
        tokenizer_metadata={"type": "tiny"},
        device=torch.device("cpu"),
        logger=None,
    )
    trainer.fit()
    assert (tmp_path / "checkpoints" / "milestone_000000000004" / "_SUCCESS").exists()
    assert (tmp_path / "checkpoints" / "latest" / "_SUCCESS").exists()


def test_grad_accum_uses_global_batch_across_ranks(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "trainer": cfg.trainer.model_copy(update={"micro_batch_size": 2, "global_batch_size": 384}),
        }
    )
    model = build_model(cfg.model)
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    scheduler = TokenCosineScheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
    trainer = Trainer(
        cfg=cfg,
        chain=[],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_iterator=[],
        checkpoint_manager=CheckpointManager(tmp_path, keep_latest=2),
        manifest_metadata={},
        tokenizer_metadata={},
        device=torch.device("cpu"),
        logger=None,
        distributed=FakeDistributedContext(),
    )
    assert trainer.global_micro_batch_size == 16
    assert trainer.grad_accum_steps == 24


def test_trainer_selects_rank_local_distributed_data_state():
    ctx = FakeDistributedContext()
    ctx.rank = 5
    trainer = object.__new__(Trainer)
    trainer.distributed = ctx
    trainer.data_parallel_world_size = ctx.world_size
    state = {
        "mode": "distributed_data_state",
        "world_size": ctx.world_size,
        "rank_states": {"0": {"rank": 0}, "5": {"rank": 5}},
    }
    assert trainer._data_state_for_current_rank(state) == {"rank": 5}


def test_trainer_skips_legacy_single_rank_data_state_on_non_main_rank():
    ctx = FakeDistributedContext()
    ctx.rank = 3
    trainer = object.__new__(Trainer)
    trainer.distributed = ctx
    trainer.data_parallel_world_size = ctx.world_size
    assert trainer._data_state_for_current_rank({"mode": "async_tokenization"}) == {}


def test_grad_accum_uses_no_sync_before_sync_step(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "trainer": cfg.trainer.model_copy(update={"micro_batch_size": 2, "global_batch_size": 384}),
            "distributed": cfg.distributed.model_copy(update={"gradient_accumulation_no_sync": True}),
        }
    )
    model = build_model(cfg.model)
    calls: list[str] = []

    @contextmanager
    def no_sync():
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    model.no_sync = no_sync  # type: ignore[method-assign]
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    scheduler = TokenCosineScheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
    trainer = Trainer(
        cfg=cfg,
        chain=[],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_iterator=[],
        checkpoint_manager=CheckpointManager(tmp_path, keep_latest=2),
        manifest_metadata={},
        tokenizer_metadata={},
        device=torch.device("cpu"),
        logger=None,
        distributed=FakeDistributedContext(),
    )
    with trainer._gradient_sync_context(0):
        pass
    assert calls == ["enter", "exit"]
    calls.clear()
    with trainer._gradient_sync_context(trainer.grad_accum_steps - 1):
        pass
    assert calls == []


def test_scheduler_decay_tokens_can_exceed_training_tokens(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg = cfg.model_copy(
        update={
            "trainer": cfg.trainer.model_copy(
                update={
                    "max_tokens": 2_000,
                    "optimizer": cfg.trainer.optimizer.model_copy(update={"lr": 1.0e-3}),
                    "scheduler": cfg.trainer.scheduler.model_copy(
                        update={"warmup_tokens": 200, "decay_tokens": 10_000, "min_lr_ratio": 0.1}
                    ),
                }
            ),
        }
    )
    model = build_model(cfg.model)
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    scheduler = TokenCosineScheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
    scheduler.step(2_000)
    progress = (2_000 - 200) / (10_000 - 200)
    expected_scale = 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
    assert math.isclose(scheduler.get_lr(), 1.0e-3 * expected_scale)
    assert scheduler.get_lr() > 1.0e-3 * 0.1


def test_wsd_scheduler_can_warmup_from_resume_lr():
    param = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([param], lr=3.0e-4)
    cfg = SchedulerConfig(
        type="wsd",
        start_tokens=30_000_000_000,
        warmup_tokens=200_000_000,
        warmup_start_ratio=0.8,
        stable_tokens=9_800_000_000,
        decay_tokens=10_000_000_000,
        min_lr_ratio=0.01,
    )
    scheduler = TokenWSDScheduler(optimizer, cfg, max_tokens=50_000_000_000)

    scheduler.step(30_000_000_000)
    assert math.isclose(scheduler.get_lr(), 3.0e-4 * 0.8)
    scheduler.step(30_100_000_000)
    assert math.isclose(scheduler.get_lr(), 3.0e-4 * 0.9)
    scheduler.step(30_200_000_000)
    assert math.isclose(scheduler.get_lr(), 3.0e-4)
    scheduler.step(40_000_000_000)
    assert math.isclose(scheduler.get_lr(), 3.0e-4)
    scheduler.step(50_000_000_000)
    assert math.isclose(scheduler.get_lr(), 3.0e-4 * 0.01)


def test_trainer_heartbeat_is_separate_from_step_logging(tmp_path):
    rows = [
        {"id": f"r{i}", "text": f"hello {i}", "source": "s", "domain": "d", "language": "en", "metadata": {}}
        for i in range(6)
    ]
    shard = tmp_path / "data.jsonl"
    shard.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = write_manifest([inspect_shard(shard, source="s", domain="d", language="en")], tmp_path)
    cfg = make_cfg(tmp_path)
    cfg = cfg.model_copy(update={
        "data": cfg.data.model_copy(update={"manifest_path": manifest.manifest}),
        "observability": cfg.observability.model_copy(update={"heartbeat": True, "heartbeat_interval_seconds": 60}),
    })
    reader = ShardReader(manifest.manifest, validate_hashes=False)
    packed = PackedDataIterator(
        RecordPipeline([]).apply(reader),
        TinyTokenizer(),
        seq_len=4,
        batch_size=1,
        upstream_state_getter=reader.state_dict,
        upstream_state_loader=reader.load_state_dict,
    )
    model = build_model(cfg.model)
    optimizer = build_optimizer(model, cfg.trainer.optimizer)
    scheduler = TokenCosineScheduler(optimizer, cfg.trainer.scheduler, max_tokens=cfg.trainer.max_tokens)
    manager = CheckpointManager(tmp_path, keep_latest=2)
    logger = CaptureLogger()
    trainer = Trainer(
        cfg=cfg,
        chain=[{"path": "a", "sha256": "b"}],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_iterator=packed,
        checkpoint_manager=manager,
        manifest_metadata={"manifest_sha256": "abc"},
        tokenizer_metadata={"type": "tiny"},
        device=torch.device("cpu"),
        logger=logger,
    )
    trainer._last_heartbeat_time -= 60
    trainer._write_heartbeat()
    trainer.fit()
    assert logger.beats
    assert logger.metrics
    assert logger.metrics[-1]["consumed_tokens"] == trainer.state.global_consumed_tokens
    assert "local_consumed_tokens" in logger.metrics[-1]


def test_async_packed_iterator_smoke(tmp_path):
    rows = [
        {"id": f"r{i}", "text": f"hello {i}", "source": "s", "domain": "d", "language": "en", "metadata": {}}
        for i in range(6)
    ]
    shard = tmp_path / "data.jsonl"
    shard.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = write_manifest([inspect_shard(shard, source="s", domain="d", language="en")], tmp_path)
    cfg = make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"manifest_path": manifest.manifest})})
    packed = AsyncPackedDataIterator(
        manifest_path=cfg.data.manifest_path,
        tokenizer_cfg=cfg.tokenizer,
        pipeline_cfg=cfg.data.pipeline,
        seq_len=4,
        batch_size=1,
        producer_workers=1,
        queue_max_batches=2,
        validate_hashes=False,
    )
    batches = list(packed)
    assert batches
    assert all(batch.input_ids.shape[1] == 4 for batch in batches)


def test_async_packed_iterator_writes_rank_data_metrics(tmp_path):
    rows = [
        {"id": f"r{i}", "text": f"hello {i}", "source": "s", "domain": "d", "language": "en", "metadata": {}}
        for i in range(6)
    ]
    shard = tmp_path / "data.jsonl"
    shard.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = write_manifest([inspect_shard(shard, source="s", domain="d", language="en")], tmp_path)
    cfg = make_cfg(tmp_path)
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"manifest_path": manifest.manifest})})
    metrics_path = tmp_path / "data_metrics_rank0.jsonl"
    packed = AsyncPackedDataIterator(
        manifest_path=cfg.data.manifest_path,
        tokenizer_cfg=cfg.tokenizer,
        pipeline_cfg=cfg.data.pipeline,
        seq_len=4,
        batch_size=1,
        producer_workers=1,
        queue_max_batches=2,
        validate_hashes=False,
        metrics_path=metrics_path,
        metrics_interval_seconds=1,
    )
    assert list(packed)
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert records
    last = records[-1]
    assert last["rank"] == 0
    assert last["producer_workers_per_rank"] == 1
    assert last["produced_tokens"] > 0
    assert last["consumed_tokens"] > 0
    assert "queue_depth_batches" in last
    assert "consumer_tokens_per_sec" in last
