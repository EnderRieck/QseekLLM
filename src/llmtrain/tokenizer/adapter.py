from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from llmtrain.tokenizer.config import TokenizerConfig


class ByteFallbackTokenizer:
    def __init__(self, *, vocab_size: int, eot_token: str = "<|endoftext|>") -> None:
        if vocab_size < 3:
            raise ValueError("vocab_size must be at least 3 for fallback tokenizer")
        self.vocab_size = vocab_size
        self.eot_token = eot_token
        self._eot_id = vocab_size - 1
        self._byte_space = vocab_size - 2

    @property
    def eot_id(self) -> int:
        return self._eot_id

    def encode(self, text: str) -> list[int]:
        raw = text.encode("utf-8", errors="ignore")
        return [1 + (b % self._byte_space) for b in raw]

    def decode(self, ids: list[int]) -> str:
        return bytes(((i - 1) % self._byte_space) for i in ids if 0 < i < self._eot_id).decode("utf-8", errors="ignore")

    def metadata(self) -> dict:
        return {
            "type": "byte_fallback",
            "vocab_size": self.vocab_size,
            "eot_token": self.eot_token,
            "eot_id": self.eot_id,
        }


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path, *, eot_token: str = "<|endoftext|>") -> None:
        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.eot_token = eot_token
        eot_id = self.processor.piece_to_id(eot_token)
        if eot_id < 0:
            raise ValueError(f"Missing required EOT token in tokenizer: {eot_token}")
        self._eot_id = eot_id
        pad_id = self.processor.piece_to_id("<pad>")
        self._pad_id = int(pad_id) if pad_id >= 0 else None

    @property
    def eot_id(self) -> int:
        return self._eot_id

    @property
    def pad_id(self) -> int | None:
        return self._pad_id

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def decode(self, ids: list[int]) -> str:
        return str(self.processor.decode(ids))

    def metadata(self) -> dict:
        return {
            "type": "sentencepiece",
            "model_path": str(self.model_path),
            "vocab_size": self.processor.get_piece_size(),
            "eot_token": self.eot_token,
            "eot_id": self.eot_id,
            "pad_id": self.pad_id,
        }


class HFByteBPETokenizer:
    def __init__(self, model_path: str | Path, *, eot_token: str = "<|endoftext|>") -> None:
        self.model_path = Path(model_path)
        self.processor = Tokenizer.from_file(str(self.model_path))
        if self.processor.decoder is None:
            self.processor.decoder = ByteLevelDecoder()
        self.eot_token = eot_token
        eot_id = self.processor.token_to_id(eot_token)
        if eot_id is None:
            raise ValueError(f"Missing required EOT token in tokenizer: {eot_token}")
        self._eot_id = int(eot_id)
        pad_id = self.processor.token_to_id("<pad>")
        self._pad_id = int(pad_id) if pad_id is not None else None

    @property
    def eot_id(self) -> int:
        return self._eot_id

    @property
    def pad_id(self) -> int | None:
        return self._pad_id

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text).ids)

    def decode(self, ids: list[int]) -> str:
        return str(self.processor.decode(ids))

    def metadata(self) -> dict:
        return {
            "type": "hf_byte_bpe",
            "model_path": str(self.model_path),
            "vocab_size": self.processor.get_vocab_size(),
            "eot_token": self.eot_token,
            "eot_id": self.eot_id,
            "pad_id": self.pad_id,
        }


def load_tokenizer(cfg: TokenizerConfig) -> SentencePieceTokenizer | HFByteBPETokenizer:
    if cfg.model_path is None:
        return ByteFallbackTokenizer(vocab_size=cfg.vocab_size)
    eot_token = "<|endoftext|>"
    if eot_token not in cfg.special_tokens:
        raise ValueError(f"Missing required EOT token in tokenizer config: {eot_token}")
    if not Path(cfg.model_path).exists():
        return ByteFallbackTokenizer(vocab_size=cfg.vocab_size, eot_token=eot_token)
    if cfg.algorithm == "sentencepiece_bpe":
        return SentencePieceTokenizer(cfg.model_path, eot_token=eot_token)
    if cfg.algorithm == "hf_byte_bpe":
        return HFByteBPETokenizer(cfg.model_path, eot_token=eot_token)
    raise ValueError(f"Unsupported tokenizer algorithm: {cfg.algorithm}")
