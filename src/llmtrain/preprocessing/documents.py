from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llmtrain.interfaces import Record


@dataclass
class RawDocument:
    id: str
    text: str
    source: str
    domain: str
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Record:
        return Record(
            id=self.id,
            text=self.text,
            source=self.source,
            domain=self.domain,
            language=self.language,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_record(cls, record: Record) -> "RawDocument":
        return cls(
            id=record.id,
            text=record.text,
            source=record.source,
            domain=record.domain,
            language=record.language,
            metadata=dict(record.metadata),
        )
