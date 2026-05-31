from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from collections.abc import Iterator

import pyarrow.parquet as pq
import sentencepiece as spm
from tokenizers import Tokenizer

from llmtrain.data.manifest import ShardInfo, load_manifest, validate_manifest
from llmtrain.data.readers import ShardReader
from llmtrain.data.schemas import validate_record


def inspect_tokenizer(
    model_path: str | Path,
    manifest_path: str | Path,
    *,
    special_tokens: list[str],
    max_records: int = 1000,
    max_records_per_domain: int | None = None,
    sample_seed: int = 42,
    output_path: str | Path | None = None,
    validate_hashes: bool = True,
) -> dict:
    encoder = _TokenizerInspector(model_path)
    stats = defaultdict(lambda: {"records": 0, "chars": 0, "tokens": 0, "byte_fallback_tokens": 0})
    abnormal: list[dict] = []
    records_seen = 0
    records_used = 0
    if max_records_per_domain is None:
        reader = ShardReader(manifest_path, validate_hashes=validate_hashes)
        for i, record in enumerate(reader):
            records_seen = i + 1
            if records_used >= max_records:
                break
            _collect_tokenizer_stats(encoder, record, stats, abnormal)
            records_used += 1
    else:
        validate_manifest(manifest_path, validate_shards=validate_hashes)
        shards = load_manifest(manifest_path)
        rng = random.Random(sample_seed)
        by_domain: dict[str, list[ShardInfo]] = defaultdict(list)
        for shard in shards:
            by_domain[shard.domain].append(shard)
        for domain, domain_shards in by_domain.items():
            rng.shuffle(domain_shards)
            for shard in domain_shards:
                for record in _iter_shard_records(shard):
                    if stats[domain]["records"] >= max_records_per_domain:
                        break
                    records_seen += 1
                    _collect_tokenizer_stats(encoder, record, stats, abnormal)
                    records_used += 1
                if stats[domain]["records"] >= max_records_per_domain:
                    break
    domains = {}
    for domain, item in stats.items():
        chars = max(1, item["chars"])
        tokens = max(1, item["tokens"])
        domains[domain] = {
            **item,
            "tokens_per_char": item["tokens"] / chars,
            "chars_per_token": item["chars"] / tokens,
        }
    report = {
        "model_path": str(model_path),
        "vocab_size": encoder.vocab_size,
        "special_token_ids": {tok: encoder.piece_to_id(tok) for tok in special_tokens},
        "records_seen": records_seen,
        "records_used": records_used,
        "max_records": max_records,
        "max_records_per_domain": max_records_per_domain,
        "sample_seed": sample_seed,
        "domains": domains,
        "abnormal_examples": abnormal[:20],
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


class _TokenizerInspector:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if self.model_path.suffix == ".json":
            self.kind = "hf"
            self.hf = Tokenizer.from_file(str(self.model_path))
            self.sp = None
            self.vocab_size = self.hf.get_vocab_size()
        else:
            self.kind = "sp"
            self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
            self.hf = None
            self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        if self.kind == "hf":
            assert self.hf is not None
            return list(self.hf.encode(text).ids)
        assert self.sp is not None
        return list(self.sp.encode(text, out_type=int))

    def id_to_piece(self, idx: int) -> str:
        if self.kind == "hf":
            assert self.hf is not None
            return self.hf.id_to_token(idx) or ""
        assert self.sp is not None
        return str(self.sp.id_to_piece(idx))

    def piece_to_id(self, piece: str) -> int | None:
        if self.kind == "hf":
            assert self.hf is not None
            return self.hf.token_to_id(piece)
        assert self.sp is not None
        return int(self.sp.piece_to_id(piece))


def _collect_tokenizer_stats(encoder: _TokenizerInspector, record, stats, abnormal: list[dict]) -> None:
    ids = encoder.encode(record.text)
    pieces = [encoder.id_to_piece(x) for x in ids]
    bucket = stats[record.domain]
    bucket["records"] += 1
    bucket["chars"] += len(record.text)
    bucket["tokens"] += len(ids)
    bucket["byte_fallback_tokens"] += sum(1 for p in pieces if p.startswith("<0x"))
    if not ids or len(ids) > max(32, len(record.text) * 4):
        abnormal.append({"id": record.id, "domain": record.domain, "chars": len(record.text), "tokens": len(ids)})


def _iter_shard_records(shard: ShardInfo) -> Iterator:
    if shard.format == "jsonl":
        with Path(shard.uri).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield validate_record(json.loads(line))
        return
    pf = pq.ParquetFile(shard.uri)
    columns = ["id", "text", "source", "domain", "language", "metadata"]
    for batch in pf.iter_batches(columns=columns, batch_size=1024):
        for row in batch.to_pylist():
            if row.get("metadata") is None:
                row["metadata"] = {}
            yield validate_record(row)
