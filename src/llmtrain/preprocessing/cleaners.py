from __future__ import annotations

import re
import unicodedata
from collections import Counter

from llmtrain.preprocessing.config import CleaningConfig
from llmtrain.preprocessing.documents import RawDocument


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


class TextCleaner:
    def __init__(self, cfg: CleaningConfig) -> None:
        self.cfg = cfg

    def clean(self, doc: RawDocument) -> tuple[RawDocument | None, str | None]:
        text = doc.text
        if self.cfg.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        control_ratio = _ratio(len(CONTROL_RE.findall(text)), len(text))
        if control_ratio > self.cfg.max_control_char_ratio:
            return None, "control_char_ratio"
        text = CONTROL_RE.sub(" ", text)
        if self.cfg.normalize_whitespace:
            lines = [SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
            text = "\n".join(line for line in lines if line)
            text = MULTI_NEWLINE_RE.sub("\n\n", text).strip()
        n = len(text)
        if n < self.cfg.min_chars:
            return None, "too_short"
        if self.cfg.max_chars is not None and n > self.cfg.max_chars:
            text = text[: self.cfg.max_chars].rstrip()
        alpha_ratio = _alpha_or_cjk_ratio(text)
        if alpha_ratio < self.cfg.min_alpha_or_cjk_ratio:
            return None, "low_text_ratio"
        repeated = _repeated_line_ratio(text)
        if repeated > self.cfg.max_repeated_line_ratio:
            return None, "repeated_lines"
        metadata = dict(doc.metadata)
        metadata["cleaning"] = {
            "chars": len(text),
            "alpha_or_cjk_ratio": alpha_ratio,
            "repeated_line_ratio": repeated,
            "control_char_ratio": control_ratio,
        }
        return RawDocument(doc.id, text, doc.source, doc.domain, doc.language, metadata), None


def _ratio(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def _alpha_or_cjk_ratio(text: str) -> float:
    good = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if ch.isalpha() or "\u4e00" <= ch <= "\u9fff":
            good += 1
    return _ratio(good, total)


def _repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(lines)
