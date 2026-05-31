from pathlib import Path

import pytest
from pydantic import ValidationError

from llmtrain.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_config_extends_and_override():
    cfg, chain = load_config(
        ROOT / "configs/train/stage0_data_tokenizer.yaml",
        ["data.packing.seq_len=64", "run.name=override_run"],
    )
    assert cfg.run.name == "override_run"
    assert cfg.data.packing.seq_len == 64
    assert cfg.data.validate_hashes is False
    assert cfg.tokenizer.vocab_size == 150000
    assert cfg.tokenizer.train_num_threads == 64
    assert len(chain) >= 4


def test_smoke_config_keeps_small_settings():
    cfg, _ = load_config(ROOT / "configs/train/stage0_data_tokenizer_smoke.yaml")
    assert cfg.data.manifest_path == Path("examples/data/manifest.jsonl")
    assert cfg.data.validate_hashes is True
    assert cfg.data.packing.seq_len == 32
    assert cfg.data.tokenizer_sampling.output_dir == Path("runs/stage0_data_tokenizer_smoke/tokenizer_corpus")
    assert cfg.data.tokenizer_sampling.output_shard_bytes == 4096
    assert cfg.data.tokenizer_sampling.num_workers == 2
    assert cfg.tokenizer.algorithm == "sentencepiece_bpe"
    assert cfg.tokenizer.vocab_size == 512
    assert cfg.tokenizer.train_num_threads == 2
    assert cfg.model.vocab_size == 512


def test_config_forbids_unknown_fields(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
run: {name: bad, output_dir: runs/bad, seed: 1, typo: true}
data: {manifest_path: examples/data/manifest.jsonl}
tokenizer: {}
model: {}
trainer: {}
checkpoint: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(bad)


def test_preprocess_paths_default_to_run_output_dir(tmp_path):
    root = tmp_path / "preprocess_run"
    cfg, _ = load_config(
        ROOT / "configs/preprocess/stream_preprocess_smoke.yaml",
        [f"run.output_dir={root}"],
    )
    assert cfg.run.output_dir == root
    assert cfg.data.manifest_path == root / "manifest.jsonl"
    assert cfg.preprocess is not None
    assert cfg.preprocess.writer.output_dir == root
    assert cfg.preprocess.writer.rejected_path == root / "rejected.jsonl"
    assert cfg.preprocess.dedup.state_dir == root / "dedup_state"
    assert cfg.preprocess.dedup.index_workers == 64
    assert cfg.preprocess.dedup.exact_shards == 256
    assert cfg.preprocess.dedup.simhash_shards == 256
    assert cfg.preprocess.dedup.simhash_max_group_size == 2048


def test_stage0_data_tokenizer_formal_paths():
    cfg, _ = load_config(ROOT / "configs/train/stage0_data_tokenizer.yaml")
    assert cfg.data.manifest_path == Path("/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/stream_preprocess/manifest.jsonl")
    assert cfg.data.validate_hashes is False
    assert cfg.tokenizer.vocab_size == 150000
    assert cfg.tokenizer.algorithm == "hf_byte_bpe"
    assert cfg.tokenizer.model_prefix == "llmtrain_hf_byte_bpe_150k"
    assert cfg.tokenizer.model_path == Path("runs/stage0_data_tokenizer_hf/tokenizer/llmtrain_hf_byte_bpe_150k.json")
    assert cfg.data.tokenizer_sampling.output_dir == Path("runs/stage0_data_tokenizer/tokenizer_corpus")
    assert cfg.data.tokenizer_sampling.output_shard_bytes == 1073741824
    assert cfg.data.tokenizer_sampling.num_workers == 64
    assert cfg.tokenizer.train_num_threads == 64


def test_stage0_data_tokenizer_hf_config():
    cfg, _ = load_config(ROOT / "configs/train/stage0_data_tokenizer_hf.yaml")
    assert cfg.run.name == "stage0_data_tokenizer_hf"
    assert cfg.tokenizer.algorithm == "hf_byte_bpe"
    assert cfg.tokenizer.model_path == Path("runs/stage0_data_tokenizer_hf/tokenizer/llmtrain_hf_byte_bpe_150k.json")
    assert cfg.tokenizer.corpus_path == Path("runs/stage0_data_tokenizer/tokenizer_corpus")
    assert cfg.tokenizer.output_dir == Path("runs/stage0_data_tokenizer_hf/tokenizer")
    assert cfg.tokenizer.train_log is True


def test_stream_preprocess_cli_cleanup_override(tmp_path):
    root = tmp_path / "preprocess_run"
    cfg, _ = load_config(
        ROOT / "configs/preprocess/stream_preprocess_smoke.yaml",
        [f"run.output_dir={root}"],
    )
    assert cfg.preprocess is not None
    overridden = cfg.model_copy(
        update={
            "preprocess": cfg.preprocess.model_copy(update={"cleanup_dedup_work": False})
        }
    )
    assert overridden.preprocess.cleanup_dedup_work is False
    assert cfg.preprocess.cleanup_dedup_work is True


def test_fineweb_edu_chinese_config_excludes_noisy_overlap_sources():
    cfg, _ = load_config(ROOT / "configs/preprocess/stage0_stream_preprocess.yaml")
    assert cfg.preprocess is not None
    source = next(item for item in cfg.preprocess.sources if item.name == "fineweb_edu_chinese_v21")
    assert source.hf_repo_id == "opencsg/Fineweb-Edu-Chinese-V2.1"
    assert source.hf_include_patterns == ["3_4/*.parquet", "4_5/*.parquet"]
    assert source.metadata_fields == ["score", "source"]
    assert source.exclude_metadata_values == {"source": ["CCI3", "ChineseWebText"]}


def test_fineweb_edu_chinese_dedicated_preprocess_config():
    cfg, _ = load_config(ROOT / "configs/preprocess/stage0_fineweb_edu_chinese_v21_preprocess.yaml")
    assert cfg.run.name == "stream_preprocess_fineweb_edu_chinese_v21"
    assert cfg.run.output_dir == Path(
        "/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/stream_preprocess_fineweb_edu_chinese_v21"
    )
    assert cfg.data.manifest_path == cfg.run.output_dir / "manifest.jsonl"
    assert cfg.preprocess is not None
    assert cfg.preprocess.writer.output_dir == cfg.run.output_dir
    assert cfg.preprocess.dedup.state_dir == cfg.run.output_dir / "dedup_state"
    assert [source.name for source in cfg.preprocess.sources] == ["fineweb_edu_chinese_v21"]
    source = cfg.preprocess.sources[0]
    assert source.type == "remote_parquet"
    assert source.hf_include_patterns == ["3_4/*.parquet", "4_5/*.parquet"]
    assert source.exclude_metadata_values == {"source": ["CCI3", "ChineseWebText"]}
