from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llmtrain.interfaces import Record


class UnifiedRecord(BaseModel):
    id: str
    text: str
    source: str
    domain: str
    language: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    def to_record(self) -> Record:
        return Record(
            id=self.id,
            text=self.text,
            source=self.source,
            domain=self.domain,
            language=self.language,
            metadata=dict(self.metadata),
        )


def validate_record(data: dict[str, Any]) -> Record:
    return UnifiedRecord.model_validate(data).to_record()


def record_to_dict(record: Record) -> dict[str, Any]:
    return {
        "id": record.id,
        "text": record.text,
        "source": record.source,
        "domain": record.domain,
        "language": record.language,
        "metadata": record.metadata,
    }
