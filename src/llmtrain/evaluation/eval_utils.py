"""Per-source validation tokenization + forward eval.

Shared between the standalone tool (`tools/run_validation.py`) and the
in-training `ValidationCallback`. The model passed in must already be on
`device` and in `.eval()` mode; this module does NOT manage train/eval
mode flips.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def iter_shard_records(shard: Any) -> Iterator[str]:
    """Yield text strings from a single ShardInfo, respecting record_start/record_end."""
    import pyarrow.parquet as pq

    slice_start = getattr(shard, "record_start", 0) or 0
    slice_end = getattr(shard, "record_end", None)
    if slice_end is None:
        slice_end = getattr(shard, "num_records", None)
    fmt = shard.format

    if fmt == "jsonl":
        seen = 0
        with Path(shard.uri).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if seen < slice_start:
                    seen += 1
                    continue
                if slice_end is not None and seen >= slice_end:
                    break
                d = json.loads(line)
                yield d.get("text") or ""
                seen += 1
        return

    seen = 0
    pf = pq.ParquetFile(shard.uri)
    for batch in pf.iter_batches(columns=["text"], batch_size=4096):
        for row in batch.to_pylist():
            if seen < slice_start:
                seen += 1
                continue
            if slice_end is not None and seen >= slice_end:
                return
            yield row.get("text") or ""
            seen += 1


def pack_token_stream(
    token_stream: Iterable[list[int]], seq_len: int, batch_size: int
) -> Iterator[tuple[list[list[int]], list[list[int]]]]:
    """Pack a stream of per-document token lists into (input_ids, document_ids) batches.

    Documents are concatenated (with whatever EOT separator the caller adds)
    and chunked into seq_len-sized windows. Each window inherits a per-token
    document_id so cross-document attention boundaries can be masked.
    """
    buf_ids: list[int] = []
    buf_doc: list[int] = []
    doc_id = 0
    pack_ids: list[list[int]] = []
    pack_doc: list[list[int]] = []
    for ids in token_stream:
        if not ids:
            continue
        buf_ids.extend(ids)
        buf_doc.extend([doc_id] * len(ids))
        doc_id += 1
        while len(buf_ids) >= seq_len:
            pack_ids.append(buf_ids[:seq_len])
            pack_doc.append(buf_doc[:seq_len])
            buf_ids = buf_ids[seq_len:]
            buf_doc = buf_doc[seq_len:]
            if len(pack_ids) == batch_size:
                yield pack_ids, pack_doc
                pack_ids = []
                pack_doc = []


def evaluate_source(
    *,
    model,
    tokenizer,
    shards: list[Any],
    seq_len: int,
    max_tokens: int,
    device: torch.device,
    dtype: torch.dtype | None,
    eot_id: int,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Compute mean CE / PPL on `shards` (one source). Caller owns model.eval()."""
    sum_loss_x_count = 0.0
    sum_count = 0
    n_packs = 0
    target_packs = max(1, max_tokens // seq_len)

    def _tokens() -> Iterator[list[int]]:
        for shard in shards:
            for text in iter_shard_records(shard):
                if not text:
                    continue
                ids = tokenizer.encode(text)
                if not ids:
                    continue
                yield ids + [eot_id]

    for batch_chunk_ids, batch_chunk_doc in pack_token_stream(_tokens(), seq_len, batch_size):
        if n_packs >= target_packs:
            break
        input_ids = torch.tensor(batch_chunk_ids, dtype=torch.long, device=device)
        doc_ids = torch.tensor(batch_chunk_doc, dtype=torch.long, device=device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=dtype if dtype is not None else torch.float32,
                enabled=(dtype is not None and device.type == "cuda"),
            ):
                out = model(input_ids, document_ids=doc_ids)
            logits = out.logits
            if logits.shape[-1] == 0:
                raise RuntimeError("Empty logits in eval — check that model is in .eval() mode")
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = input_ids[:, 1:].contiguous()
            same_doc = doc_ids[:, 1:] == doc_ids[:, :-1]
            shift_labels = shift_labels.masked_fill(~same_doc, -100)
            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            count = (shift_labels != -100).sum()
            sum_loss_x_count += float(losses.item())
            sum_count += int(count.item())
            n_packs += input_ids.shape[0]

    mean_ce = sum_loss_x_count / max(1, sum_count)
    return {
        "n_packs": n_packs,
        "n_tokens_eval": sum_count,
        "mean_ce": mean_ce,
        "ppl": math.exp(mean_ce) if mean_ce < 50 else float("inf"),
        "sum_loss_x_count": sum_loss_x_count,
    }
