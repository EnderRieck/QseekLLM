from __future__ import annotations

import json
import os
from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from llmtrain.tokenizer.config import TokenizerConfig


def train_tokenizer(cfg: TokenizerConfig, corpus_path: str | Path | None = None) -> dict:
    if cfg.algorithm == "sentencepiece_bpe":
        return train_sentencepiece_bpe(cfg, corpus_path)
    if cfg.algorithm == "hf_byte_bpe":
        return train_hf_byte_bpe(cfg, corpus_path)
    raise ValueError(f"Unsupported tokenizer algorithm: {cfg.algorithm}")


def train_sentencepiece_bpe(cfg: TokenizerConfig, corpus_path: str | Path | None = None) -> dict:
    if cfg.algorithm not in {"sentencepiece_bpe"}:
        raise ValueError(f"Unsupported tokenizer algorithm: {cfg.algorithm}")
    corpus = Path(corpus_path or cfg.corpus_path or "")
    corpus_inputs = _resolve_corpus_inputs(corpus)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.output_dir / cfg.model_prefix
    user_symbols = [tok for tok in cfg.special_tokens if tok not in {"<unk>", "<s>", "</s>", "<pad>"}]
    args = {
        "input": ",".join(str(path) for path in corpus_inputs),
        "model_prefix": str(prefix),
        "model_type": "bpe",
        "vocab_size": cfg.vocab_size,
        "character_coverage": cfg.character_coverage,
        "byte_fallback": cfg.byte_fallback,
        "normalization_rule_name": cfg.normalization_rule_name,
        "shuffle_input_sentence": cfg.shuffle_input_sentence,
        "unk_piece": "<unk>",
        "bos_piece": "<s>",
        "eos_piece": "</s>",
        "pad_piece": "<pad>",
        "pad_id": 3,
        "user_defined_symbols": ",".join(user_symbols),
        "hard_vocab_limit": False,
    }
    if cfg.train_num_threads is not None:
        args["num_threads"] = cfg.train_num_threads
    if cfg.train_log:
        spm.SentencePieceTrainer.Train(**args)
    else:
        with open(os.devnull, "w", encoding="utf-8") as logstream:
            spm.SentencePieceTrainer.Train(logstream=logstream, **args)
    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")
    metadata = tokenizer_metadata(model_path, cfg.special_tokens)
    metadata.update(
        {
            "model_path": str(model_path),
            "vocab_path": str(vocab_path),
            "corpus_path": str(corpus),
            "corpus_inputs": [str(path) for path in corpus_inputs],
            "train_num_threads": cfg.train_num_threads,
            "train_log": cfg.train_log,
        }
    )
    (cfg.output_dir / "tokenizer.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def train_hf_byte_bpe(cfg: TokenizerConfig, corpus_path: str | Path | None = None) -> dict:
    if cfg.algorithm != "hf_byte_bpe":
        raise ValueError(f"Unsupported tokenizer algorithm: {cfg.algorithm}")
    corpus = Path(corpus_path or cfg.corpus_path or "")
    corpus_inputs = _resolve_corpus_inputs(corpus)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=cfg.byte_fallback))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=cfg.special_tokens,
        show_progress=cfg.train_log,
        initial_alphabet=ByteLevel.alphabet(),
    )
    if cfg.train_num_threads is not None:
        os.environ.setdefault("RAYON_NUM_THREADS", str(cfg.train_num_threads))
    tokenizer.train([str(path) for path in corpus_inputs], trainer=trainer)
    tokenizer_path = cfg.output_dir / f"{cfg.model_prefix}.json"
    tokenizer.save(str(tokenizer_path))
    vocab_path, merges_path = tokenizer.model.save(str(cfg.output_dir), cfg.model_prefix)
    metadata = {
        "type": "hf_byte_bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "model_path": str(tokenizer_path),
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
        "corpus_path": str(corpus),
        "corpus_inputs": [str(path) for path in corpus_inputs],
        "train_num_threads": cfg.train_num_threads,
        "train_log": cfg.train_log,
        "special_token_ids": {tok: tokenizer.token_to_id(tok) for tok in cfg.special_tokens},
    }
    missing = [tok for tok, idx in metadata["special_token_ids"].items() if idx is None]
    if missing:
        raise ValueError(f"Tokenizer missing special tokens: {missing}")
    (cfg.output_dir / "tokenizer.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def _resolve_corpus_inputs(corpus: Path) -> list[Path]:
    if corpus.is_dir():
        inputs = sorted(p for p in corpus.glob("*.txt") if p.is_file())
    else:
        inputs = [corpus]
    if not inputs or any(not path.exists() for path in inputs):
        raise FileNotFoundError(f"Tokenizer corpus not found: {corpus}")
    return inputs


def tokenizer_metadata(model_path: str | Path, special_tokens: list[str]) -> dict:
    model = Path(model_path)
    if model.suffix == ".json":
        tokenizer = Tokenizer.from_file(str(model))
        ids = {tok: tokenizer.token_to_id(tok) for tok in special_tokens}
        missing = [tok for tok, idx in ids.items() if idx is None]
        if missing:
            raise ValueError(f"Tokenizer missing special tokens: {missing}")
        return {
            "type": "hf_byte_bpe",
            "vocab_size": tokenizer.get_vocab_size(),
            "special_token_ids": ids,
        }

    sp = spm.SentencePieceProcessor(model_file=str(model))
    ids = {tok: sp.piece_to_id(tok) for tok in special_tokens}
    missing = [tok for tok, idx in ids.items() if idx < 0]
    if missing:
        raise ValueError(f"Tokenizer missing special tokens: {missing}")
    return {"type": "sentencepiece", "vocab_size": sp.get_piece_size(), "special_token_ids": ids}
