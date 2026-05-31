from pathlib import Path

from llmtrain.data.manifest import inspect_shard, write_manifest
from llmtrain.tokenizer.adapter import HFByteBPETokenizer, SentencePieceTokenizer
from llmtrain.tokenizer.config import TokenizerConfig
from llmtrain.tokenizer.inspector import inspect_tokenizer
from llmtrain.tokenizer.trainer import train_hf_byte_bpe, train_sentencepiece_bpe


def _write_corpus(path: Path) -> None:
    path.write_text(
        "\n".join([
            "This tokenizer sample includes English and 中文。",
            "def add(a, b): return a + b",
            "数学公式 x^2 + y^2 = z^2",
        ]),
        encoding="utf-8",
    )


def test_sentencepiece_train_adapter_and_inspection(tmp_path):
    corpus = tmp_path / "corpus.txt"
    _write_corpus(corpus)
    cfg = TokenizerConfig(
        vocab_size=400,
        model_prefix="tok",
        corpus_path=corpus,
        output_dir=tmp_path,
        byte_fallback=True,
        train_num_threads=2,
    )
    metadata = train_sentencepiece_bpe(cfg)
    assert metadata["train_num_threads"] == 2
    assert metadata["train_log"] is False
    tok = SentencePieceTokenizer(metadata["model_path"])
    ids = tok.encode("hello 中文")
    assert ids
    assert tok.eot_id == metadata["special_token_ids"]["<|endoftext|>"]


def test_sentencepiece_train_accepts_corpus_directory(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_corpus(corpus_dir / "part-00000.txt")
    _write_corpus(corpus_dir / "part-00001.txt")
    cfg = TokenizerConfig(
        vocab_size=400,
        model_prefix="tok_dir",
        corpus_path=corpus_dir,
        output_dir=tmp_path,
        byte_fallback=True,
        train_num_threads=2,
    )
    metadata = train_sentencepiece_bpe(cfg)
    assert metadata["corpus_path"] == str(corpus_dir)
    assert metadata["corpus_inputs"] == [
        str(corpus_dir / "part-00000.txt"),
        str(corpus_dir / "part-00001.txt"),
    ]


def test_hf_byte_bpe_train_adapter_and_inspection(tmp_path):
    corpus_dir = tmp_path / "hf_corpus"
    corpus_dir.mkdir()
    _write_corpus(corpus_dir / "part-00000.txt")
    _write_corpus(corpus_dir / "part-00001.txt")
    cfg = TokenizerConfig(
        algorithm="hf_byte_bpe",
        vocab_size=400,
        model_prefix="hf_tok",
        corpus_path=corpus_dir,
        output_dir=tmp_path,
        byte_fallback=True,
        train_num_threads=2,
    )
    metadata = train_hf_byte_bpe(cfg)
    assert metadata["type"] == "hf_byte_bpe"
    assert Path(metadata["model_path"]).exists()
    assert Path(metadata["vocab_path"]).exists()
    assert Path(metadata["merges_path"]).exists()
    tok = HFByteBPETokenizer(metadata["model_path"])
    ids = tok.encode("hello 中文")
    assert ids
    assert tok.decode(ids) == "hello 中文"
    assert tok.eot_id == metadata["special_token_ids"]["<|endoftext|>"]
    assert tok.metadata()["pad_id"] == metadata["special_token_ids"]["<pad>"]


def test_inspect_tokenizer_can_sample_each_domain(tmp_path):
    corpus_dir = tmp_path / "hf_corpus"
    corpus_dir.mkdir()
    _write_corpus(corpus_dir / "part-00000.txt")
    cfg = TokenizerConfig(
        algorithm="hf_byte_bpe",
        vocab_size=400,
        model_prefix="hf_tok_domains",
        corpus_path=corpus_dir,
        output_dir=tmp_path,
        byte_fallback=True,
        train_num_threads=2,
    )
    metadata = train_hf_byte_bpe(cfg)

    zh = tmp_path / "zh.jsonl"
    en = tmp_path / "en.jsonl"
    zh.write_text(
        "\n".join(
            [
                '{"id":"zh/0","text":"中文样本一","source":"zh","domain":"zh","language":"zh","metadata":{}}',
                '{"id":"zh/1","text":"中文样本二","source":"zh","domain":"zh","language":"zh","metadata":{}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    en.write_text(
        '{"id":"en/0","text":"English sample","source":"en","domain":"en","language":"en","metadata":{}}\n',
        encoding="utf-8",
    )
    manifest = write_manifest(
        [
            inspect_shard(zh, source="zh", domain="zh", language="zh"),
            inspect_shard(en, source="en", domain="en", language="en"),
        ],
        tmp_path / "manifest",
    )

    global_report = inspect_tokenizer(
        metadata["model_path"],
        manifest.manifest,
        special_tokens=cfg.special_tokens,
        max_records=1,
    )
    assert set(global_report["domains"]) == {"zh"}

    domain_report = inspect_tokenizer(
        metadata["model_path"],
        manifest.manifest,
        special_tokens=cfg.special_tokens,
        max_records_per_domain=1,
    )
    assert set(domain_report["domains"]) == {"zh", "en"}
    assert domain_report["domains"]["zh"]["records"] == 1
    assert domain_report["domains"]["en"]["records"] == 1
    assert domain_report["records_seen"] == 2
