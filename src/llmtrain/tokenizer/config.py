from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class TokenizerConfig(BaseModel):
    algorithm: Literal["sentencepiece_bpe", "hf_byte_bpe"] = "sentencepiece_bpe"
    vocab_size: PositiveInt = 150_000
    model_prefix: str = "llmtrain_spm"
    model_path: Path | None = None
    corpus_path: Path | None = None
    output_dir: Path = Path("runs/tokenizer")
    character_coverage: float = Field(0.9995, gt=0.0, le=1.0)
    byte_fallback: bool = True
    normalization_rule_name: str = "identity"
    shuffle_input_sentence: bool = True
    train_num_threads: PositiveInt | None = None
    train_log: bool = False
    special_tokens: list[str] = Field(
        default_factory=lambda: [
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|tool|>",
            "<|endoftext|>",
        ]
    )

    model_config = {"extra": "forbid"}
