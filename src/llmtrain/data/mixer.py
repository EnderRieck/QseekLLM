from __future__ import annotations

import pickle
import random
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from typing import Callable

from llmtrain.data.manifest import ShardInfo
from llmtrain.interfaces import Record


def default_token_estimator(record: Record) -> int:
    return max(1, len(record.text.encode("utf-8")) // 4)


class WeightedMixer:
    def __init__(
        self,
        streams: dict[str, Iterable[Record]],
        source_token_weights: dict[str, float],
        *,
        temperature: float = 1.0,
        seed: int = 42,
        token_estimator: Callable[[Record], int] = default_token_estimator,
    ) -> None:
        self.streams = {k: iter(v) for k, v in streams.items()}
        self.source_token_weights = dict(source_token_weights)
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.token_estimator = token_estimator
        self.consumed_tokens_per_source: dict[str, int] = defaultdict(int)
        self._buffers: dict[str, deque[Record]] = defaultdict(deque)
        self._active = set(streams)

    @classmethod
    def from_manifest_streams(
        cls,
        streams: dict[str, Iterable[Record]],
        shards: Iterable[ShardInfo],
        *,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> "WeightedMixer":
        weights: dict[str, float] = defaultdict(float)
        for shard in shards:
            weights[shard.domain] += shard.estimated_tokens * shard.weight
        return cls(streams, weights, temperature=temperature, seed=seed)

    def _probabilities(self) -> tuple[list[str], list[float]]:
        sources = sorted(self._active)
        raw = [max(self.source_token_weights.get(s, 1.0), 1e-12) ** self.temperature for s in sources]
        total = sum(raw)
        return sources, [v / total for v in raw]

    def __iter__(self) -> Iterator[Record]:
        while self._active:
            sources, probs = self._probabilities()
            source = self.rng.choices(sources, weights=probs, k=1)[0]
            try:
                record = self._buffers[source].popleft() if self._buffers[source] else next(self.streams[source])
            except StopIteration:
                self._active.remove(source)
                continue
            self.consumed_tokens_per_source[source] += self.token_estimator(record)
            yield record

    def ratio_stats(self) -> dict[str, float]:
        total = sum(self.consumed_tokens_per_source.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in sorted(self.consumed_tokens_per_source.items())}

    def state_dict(self) -> dict:
        return {
            "rng": pickle.dumps(self.rng.getstate()),
            "consumed_tokens_per_source": dict(self.consumed_tokens_per_source),
            "active": sorted(self._active),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.rng.setstate(pickle.loads(sd["rng"]))
        self.consumed_tokens_per_source = defaultdict(int, sd.get("consumed_tokens_per_source", {}))
        self._active = set(sd.get("active", self.streams.keys()))
