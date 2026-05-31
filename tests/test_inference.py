from __future__ import annotations

from pathlib import Path

import torch

from llmtrain.inference import GenerationConfig, InferenceConfig, InferenceEngine, load_inference_config
from llmtrain.models import build_model
from llmtrain.models.config import ModelConfig


class ToyTokenizer:
    eot_id = 0

    def encode(self, text: str) -> list[int]:
        return [1 + (ord(ch) % 10) for ch in text] or [self.eot_id]

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    def metadata(self) -> dict:
        return {"type": "toy"}


def test_inference_config_loads_default_file():
    cfg = load_inference_config(Path("configs/inference/default.yaml"))
    assert isinstance(cfg, InferenceConfig)
    assert cfg.generation.use_kv_cache is True


def test_inference_engine_generates_with_kv_cache():
    torch.manual_seed(0)
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
    ).eval()
    engine = InferenceEngine(
        model=model,
        tokenizer=ToyTokenizer(),
        max_context_tokens=16,
        device=torch.device("cpu"),
    )
    result = engine.generate(
        "abc",
        GenerationConfig(max_new_tokens=3, do_sample=False, stop_on_eot=False, use_kv_cache=True),
    )
    assert result.generated_tokens == 3
    assert len(result.new_token_ids) == 3
    assert result.stop_reason == "max_new_tokens"


def test_streaming_generation_yields_text_deltas():
    torch.manual_seed(0)
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
    ).eval()
    engine = InferenceEngine(
        model=model,
        tokenizer=ToyTokenizer(),
        max_context_tokens=16,
        device=torch.device("cpu"),
    )
    cfg = GenerationConfig(max_new_tokens=3, do_sample=False, stop_on_eot=False, use_kv_cache=True)
    result = engine.generate("abc", cfg)
    streamed = "".join(step.text for step in engine.iter_generate("abc", cfg))
    assert streamed == result.text
