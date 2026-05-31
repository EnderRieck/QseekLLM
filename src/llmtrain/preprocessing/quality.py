from __future__ import annotations

import math
import re
from dataclasses import dataclass

from llmtrain.preprocessing.config import QualityConfig
from llmtrain.preprocessing.documents import RawDocument


URL_RE = re.compile(r"https?://|www\.", re.I)


@dataclass(frozen=True)
class QualityResult:
    score: float
    signals: dict[str, float]


class HeuristicQualityScorer:
    def __init__(self, cfg: QualityConfig) -> None:
        self.cfg = cfg

    def score(self, doc: RawDocument) -> QualityResult:
        text = doc.text
        chars = max(1, len(text))
        length_score = min(1.0, math.log(chars + 1) / math.log(2000))
        lines = [line for line in text.splitlines() if line.strip()]
        avg_line = sum(len(x) for x in lines) / max(1, len(lines))
        line_score = min(1.0, avg_line / 80.0)
        url_penalty = min(1.0, len(URL_RE.findall(text)) / 10.0)
        digit_punct = sum(1 for ch in text if not (ch.isalpha() or ch.isspace() or "\u4e00" <= ch <= "\u9fff"))
        symbol_ratio = digit_punct / chars
        symbol_score = 1.0 - min(1.0, max(0.0, symbol_ratio - 0.35) / 0.65)
        repeated = float(doc.metadata.get("cleaning", {}).get("repeated_line_ratio", 0.0))
        repetition_score = 1.0 - min(1.0, repeated)
        if doc.domain == "code":
            line_score = 1.0
            symbol_score = max(symbol_score, 0.6)
        score = (
            0.30 * length_score
            + 0.20 * line_score
            + 0.20 * symbol_score
            + 0.20 * repetition_score
            + 0.10 * (1.0 - url_penalty)
        )
        return QualityResult(
            score=max(0.0, min(1.0, score)),
            signals={
                "length": length_score,
                "avg_line": line_score,
                "symbol": symbol_score,
                "repetition": repetition_score,
                "url": 1.0 - url_penalty,
            },
        )
