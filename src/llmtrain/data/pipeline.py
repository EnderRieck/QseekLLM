from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from llmtrain.data.config import PipelineConfig
from llmtrain.interfaces import Record, Stage


class NormalizeWhitespace:
    def __call__(self, r: Record) -> Record | None:
        text = re.sub(r"\s+", " ", r.text).strip()
        return Record(r.id, text, r.source, r.domain, r.language, dict(r.metadata))


class LengthFilter:
    def __init__(self, min_chars: int = 1, max_chars: int | None = None) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars

    def __call__(self, r: Record) -> Record | None:
        n = len(r.text)
        if n < self.min_chars:
            return None
        if self.max_chars is not None and n > self.max_chars:
            return None
        return r


class QualityFilter:
    def __init__(self, min_quality_score: float | None) -> None:
        self.min_quality_score = min_quality_score

    def __call__(self, r: Record) -> Record | None:
        if self.min_quality_score is None:
            return r
        score = r.metadata.get("quality_score")
        if score is None:
            return None
        return r if float(score) >= self.min_quality_score else None


class RecordPipeline:
    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "RecordPipeline":
        stages: list[Stage] = []
        if cfg.normalize_whitespace:
            stages.append(NormalizeWhitespace())
        stages.extend([LengthFilter(cfg.min_chars, cfg.max_chars), QualityFilter(cfg.min_quality_score)])
        return cls(stages)

    def process(self, record: Record) -> Record | None:
        cur: Record | None = record
        for stage in self.stages:
            if cur is None:
                return None
            cur = stage(cur)
        return cur

    def apply(self, records: Iterable[Record]) -> Iterator[Record]:
        for record in records:
            out = self.process(record)
            if out is not None:
                yield out
