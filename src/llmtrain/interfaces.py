from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

import torch


@dataclass(frozen=True)
class Record:
    id: str
    text: str
    source: str
    domain: str
    language: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    document_ids: torch.Tensor
    consumed_tokens: int


class Stage(Protocol):
    def __call__(self, r: Record) -> Record | None: ...


class DataIterator(Protocol):
    def __iter__(self) -> Iterator[Batch]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, sd: dict) -> None: ...


class Tokenizer(Protocol):
    @property
    def eot_id(self) -> int: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def metadata(self) -> dict: ...
