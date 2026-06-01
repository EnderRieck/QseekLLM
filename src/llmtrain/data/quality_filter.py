"""Quality filtering for streaming data preprocessing.

Uses CCNet perplexity + heuristic rules to filter low-quality documents.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np


class QualityFilter:
    """Filter low-quality documents using perplexity + heuristics."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        ccnet_model_path: str | None = None,
        min_length: int = 50,
        max_length: int = 100000,
        max_perplexity: float = 1000.0,
        min_quality_score: float = 0.3,
    ) -> None:
        self.enabled = enabled
        self.min_length = min_length
        self.max_length = max_length
        self.max_perplexity = max_perplexity
        self.min_quality_score = min_quality_score

        self.ccnet_lm = None
        if enabled and ccnet_model_path:
            try:
                from cc_net import perplexity

                self.ccnet_lm = perplexity.load_model(ccnet_model_path)
            except ImportError:
                print("Warning: cc_net not installed, skipping perplexity filter")

        self.stats = {"total": 0, "kept": 0, "filtered": 0}

    def should_keep(self, text: str) -> tuple[bool, str]:
        """Return (keep, reason)."""
        if not self.enabled:
            return True, "disabled"

        self.stats["total"] += 1

        # Length filter
        if len(text) < self.min_length:
            self.stats["filtered"] += 1
            return False, "too_short"
        if len(text) > self.max_length:
            self.stats["filtered"] += 1
            return False, "too_long"

        # CCNet perplexity
        if self.ccnet_lm is not None:
            try:
                ppl = self.ccnet_lm.get_perplexity(text)
                if ppl > self.max_perplexity:
                    self.stats["filtered"] += 1
                    return False, f"high_ppl_{ppl:.0f}"
            except Exception:
                pass  # Skip on error

        # Heuristic quality score
        score = self._quality_score(text)
        if score < self.min_quality_score:
            self.stats["filtered"] += 1
            return False, f"low_quality_{score:.2f}"

        self.stats["kept"] += 1
        return True, "ok"

    def _quality_score(self, text: str) -> float:
        """Heuristic quality score 0-1."""
        score = 1.0

        # Character entropy (detect repetition)
        char_freq = Counter(text)
        if len(char_freq) > 1:
            probs = np.array([c / len(text) for c in char_freq.values()])
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            if entropy < 3.0:  # Very repetitive
                score *= 0.2
            elif entropy < 4.0:
                score *= 0.6

        # Punctuation ratio
        punct_count = sum(c in ",.!?;:，。！？；：" for c in text)
        punct_ratio = punct_count / len(text)
        if punct_ratio > 0.2:  # Too much punctuation
            score *= 0.4
        elif punct_ratio < 0.01:  # Almost no punctuation
            score *= 0.7

        # Digit ratio (detect spam/ads)
        digit_ratio = sum(c.isdigit() for c in text) / len(text)
        if digit_ratio > 0.3:
            score *= 0.3

        # Whitespace ratio
        ws_ratio = sum(c.isspace() for c in text) / len(text)
        if ws_ratio > 0.4 or ws_ratio < 0.05:
            score *= 0.5

        # Trigram repetition
        words = text.split()
        if len(words) > 10:
            trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
            if trigrams:
                dup_ratio = 1 - len(set(trigrams)) / len(trigrams)
                if dup_ratio > 0.5:  # >50% duplicate trigrams
                    score *= 0.2
                elif dup_ratio > 0.3:
                    score *= 0.5

        return score

    def get_stats(self) -> dict[str, Any]:
        """Return filtering statistics."""
        total = self.stats["total"]
        kept = self.stats["kept"]
        filtered = self.stats["filtered"]
        return {
            "total": total,
            "kept": kept,
            "filtered": filtered,
            "keep_ratio": kept / max(1, total),
        }
