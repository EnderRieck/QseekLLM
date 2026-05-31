from __future__ import annotations

import json
from pathlib import Path

import torch

from llmtrain.evaluation.config import load_eval_config
from llmtrain.evaluation.model import LogLikelihoodScorer
from llmtrain.evaluation.runner import _require_file, _set_dataset_cache_defaults, _split_eval_results
from llmtrain.evaluation.wplc import normalize_wplc_record, prepare_wplc_dataset
from llmtrain.models import build_model
from llmtrain.models.config import ModelConfig


class TinyTokenizer:
    eot_id = 0
    pad_id = 0

    def encode(self, text: str) -> list[int]:
        return [1 + (ord(ch) % 20) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    def metadata(self) -> dict:
        return {"type": "tiny"}


def test_wplc_normalization_uses_prefix_and_correct_word():
    sample = normalize_wplc_record(
        {"masked_text": "前文<mask><mask>后文", "correct_word": "答案"},
        index=7,
    )
    assert sample.id == "7"
    assert sample.context == "前文"
    assert sample.target == "答案"


def test_prepare_wplc_dataset_writes_jsonl(tmp_path: Path):
    source = tmp_path / "dev.json"
    output = tmp_path / "eval/datasets/chinese_wplc/dev.jsonl"
    source.write_text(
        json.dumps({"masked_text": "公主废了这个<mask><mask>。", "correct_word": "驸马"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    meta = prepare_wplc_dataset(source, output)
    assert meta["samples"] == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["context"] == "公主废了这个"
    assert row["target"] == "驸马"


def test_require_file_accepts_prepared_dataset(tmp_path: Path):
    prepared = tmp_path / "eval/datasets/chinese_wplc/dev.jsonl"
    prepared.parent.mkdir(parents=True)
    prepared.write_text('{"context":"已有","target":"数据"}\n', encoding="utf-8")
    _require_file(prepared, "prepared Chinese WPLC dataset")
    assert prepared.read_text(encoding="utf-8") == '{"context":"已有","target":"数据"}\n'


def test_split_eval_results_removes_samples_from_metrics():
    metrics, samples = _split_eval_results(
        {
            "results": {"chinese_wplc": {"acc": 0.25}},
            "configs": {"chinese_wplc": {}},
            "samples": {"chinese_wplc": [{"doc_id": 1}]},
        }
    )
    assert "samples" not in metrics
    assert metrics["results"]["chinese_wplc"]["acc"] == 0.25
    assert samples == {"chinese_wplc": [{"doc_id": 1}]}


def test_eval_config_loads_default_stage1_config():
    cfg = load_eval_config("configs/eval/stage1_general_700m.yaml")
    assert cfg.run.output_dir == Path("runs/eval")
    assert cfg.datasets.wplc_prepared_path == Path("eval/datasets/chinese_wplc/dev.jsonl")
    assert cfg.datasets.offline is True
    assert cfg.tasks == ["chinese_wplc", "lambada_openai"]


def test_dataset_cache_defaults_can_force_offline(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    _set_dataset_cache_defaults(tmp_path / "datasets", offline=True)
    assert Path(str(tmp_path / "datasets" / "huggingface")) == Path(__import__("os").environ["HF_HOME"])
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
    assert __import__("os").environ["HF_DATASETS_OFFLINE"] == "1"
    assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"


def test_loglikelihood_scorer_returns_one_score_per_request():
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
    ).eval()
    scorer = LogLikelihoodScorer(
        model=model,
        tokenizer=TinyTokenizer(),
        max_context_tokens=16,
        device=torch.device("cpu"),
        batch_size=2,
    )
    results = scorer.loglikelihood([("水的沸点是", "100度"), ("hello", " world")])
    assert len(results) == 2
    assert all(isinstance(score, float) for score, _ in results)
    assert all(isinstance(greedy, bool) for _, greedy in results)
