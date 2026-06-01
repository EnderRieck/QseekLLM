from __future__ import annotations

from collections.abc import Iterable, Iterator
from queue import Queue
from threading import Thread
from typing import Any

import torch

from llmtrain.interfaces import Batch, Record, Tokenizer


class PackedDataIterator:
    def __init__(
        self,
        records: Iterable[Record],
        tokenizer: Tokenizer,
        *,
        seq_len: int,
        batch_size: int = 1,
        upstream_state_getter=None,
        upstream_state_loader=None,
        prefetch_records: int = 0,
    ) -> None:
        self.records = iter(records)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.prefetch_records = prefetch_records
        self.buffer_ids: list[int] = []
        self.buffer_doc_ids: list[int] = []
        self.next_document_id = 0
        self.global_consumed_tokens = 0
        self._upstream_state_getter = upstream_state_getter
        self._upstream_state_loader = upstream_state_loader
        self._exhausted = False
        self._prefetch_queue: Queue[Record | None] | None = None
        self._prefetch_thread: Thread | None = None
        if self.prefetch_records > 0:
            self._prefetch_queue = Queue(maxsize=self.prefetch_records)
            self._prefetch_thread = Thread(target=self._prefetch_loop, daemon=True)
            self._prefetch_thread.start()

    def __iter__(self) -> Iterator[Batch]:
        while True:
            input_rows: list[list[int]] = []
            doc_rows: list[list[int]] = []
            while len(input_rows) < self.batch_size:
                self._fill_until(self.seq_len)
                if len(self.buffer_ids) < self.seq_len:
                    if not input_rows:
                        return
                    break
                input_rows.append(self.buffer_ids[: self.seq_len])
                doc_rows.append(self.buffer_doc_ids[: self.seq_len])
                del self.buffer_ids[: self.seq_len]
                del self.buffer_doc_ids[: self.seq_len]
            if not input_rows:
                return
            consumed = sum(len(row) for row in input_rows)
            self.global_consumed_tokens += consumed
            yield Batch(
                input_ids=torch.tensor(input_rows, dtype=torch.long),
                document_ids=torch.tensor(doc_rows, dtype=torch.long),
                consumed_tokens=consumed,
            )

    def _fill_until(self, n: int) -> None:
        while len(self.buffer_ids) < n and not self._exhausted:
            record = self._next_record()
            if record is None:
                self._exhausted = True
                return
            token_ids = self.tokenizer.encode(record.text) + [self.tokenizer.eot_id]
            doc_id = self.next_document_id
            self.next_document_id += 1
            self.buffer_ids.extend(token_ids)
            self.buffer_doc_ids.extend([doc_id] * len(token_ids))

    def _next_record(self) -> Record | None:
        if self._prefetch_queue is None:
            try:
                return next(self.records)
            except StopIteration:
                return None
        item = self._prefetch_queue.get()
        return item

    def _prefetch_loop(self) -> None:
        assert self._prefetch_queue is not None
        for record in self.records:
            self._prefetch_queue.put(record)
        self._prefetch_queue.put(None)

    def state_dict(self) -> dict[str, Any]:
        return {
            "packing": {
                "buffer_ids": list(self.buffer_ids),
                "buffer_doc_ids": list(self.buffer_doc_ids),
                "next_document_id": self.next_document_id,
                "exhausted": self._exhausted,
                "prefetch_records": self.prefetch_records,
            },
            "global_consumed_tokens": self.global_consumed_tokens,
            "upstream": self._upstream_state_getter() if self._upstream_state_getter else {},
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        packing = sd["packing"]
        self.buffer_ids = list(packing["buffer_ids"])
        self.buffer_doc_ids = list(packing["buffer_doc_ids"])
        self.next_document_id = int(packing.get("next_document_id", 0))
        self._exhausted = bool(packing.get("exhausted", False))
        self.prefetch_records = int(packing.get("prefetch_records", self.prefetch_records))
        self.global_consumed_tokens = int(sd.get("global_consumed_tokens", 0))
        if self._upstream_state_loader:
            self._upstream_state_loader(sd.get("upstream", {}))


def block_diagonal_attention_mask(document_ids: torch.Tensor) -> torch.Tensor:
    same_doc = document_ids[:, :, None] == document_ids[:, None, :]
    causal = torch.ones(
        (document_ids.shape[1], document_ids.shape[1]),
        dtype=torch.bool,
        device=document_ids.device,
    ).tril()
    return same_doc & causal


def unpack_document_segments(input_ids: list[int], document_ids: list[int], eot_id: int) -> list[list[int]]:
    segments: list[list[int]] = []
    current_doc = None
    current: list[int] = []
    for tok, doc in zip(input_ids, document_ids):
        if current_doc is None:
            current_doc = doc
        if doc != current_doc:
            segments.append(current)
            current = []
            current_doc = doc
        if tok != eot_id:
            current.append(tok)
    if current:
        segments.append(current)
    return segments
