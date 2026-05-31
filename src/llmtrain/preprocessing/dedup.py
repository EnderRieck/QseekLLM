from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from llmtrain.preprocessing.config import DedupConfig
from llmtrain.preprocessing.documents import RawDocument


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class DedupDecision:
    def __init__(self, duplicate: bool, reason: str | None = None, metadata: dict | None = None) -> None:
        self.duplicate = duplicate
        self.reason = reason
        self.metadata = metadata or {}


class StreamingDeduper:
    def __init__(self, cfg: DedupConfig) -> None:
        self.cfg = cfg
        self.exact_hashes: set[str] = set()
        self.simhashes: list[int] = []
        self.state_dir = cfg.state_dir
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            if cfg.load_existing_state:
                self._load_state()

    def check(self, doc: RawDocument) -> DedupDecision:
        normalized = normalize_for_dedup(doc.text)
        metadata: dict = {}
        if self.cfg.exact:
            h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            metadata["dedup_hash"] = h
            if h in self.exact_hashes:
                return DedupDecision(True, "exact_duplicate", metadata)
        fp = None
        if self.cfg.simhash:
            fp = simhash(normalized, bits=self.cfg.simhash_bits)
            metadata["simhash"] = str(fp)
            for old in self.simhashes:
                if hamming_distance(fp, old) <= self.cfg.simhash_threshold:
                    return DedupDecision(True, "near_duplicate", metadata)
        if self.cfg.exact:
            self.exact_hashes.add(metadata["dedup_hash"])
        if self.cfg.simhash and fp is not None:
            self.simhashes.append(fp)
        return DedupDecision(False, metadata=metadata)

    def check_fingerprint(self, exact_hash: str | None, simhash_value: int | None) -> DedupDecision:
        metadata: dict = {}
        if self.cfg.exact and exact_hash:
            metadata["dedup_hash"] = exact_hash
            if exact_hash in self.exact_hashes:
                return DedupDecision(True, "exact_duplicate", metadata)
        if self.cfg.simhash and simhash_value is not None:
            metadata["simhash"] = str(simhash_value)
            for old in self.simhashes:
                if hamming_distance(simhash_value, old) <= self.cfg.simhash_threshold:
                    return DedupDecision(True, "near_duplicate", metadata)
        if self.cfg.exact and exact_hash:
            self.exact_hashes.add(exact_hash)
        if self.cfg.simhash and simhash_value is not None:
            self.simhashes.append(simhash_value)
        return DedupDecision(False, metadata=metadata)

    def save_state(self) -> None:
        if not self.state_dir or not self.cfg.persist_state:
            return
        (self.state_dir / "exact_hashes.txt").write_text("\n".join(sorted(self.exact_hashes)), encoding="utf-8")
        with (self.state_dir / "simhashes.jsonl").open("w", encoding="utf-8") as f:
            for fp in self.simhashes:
                f.write(json.dumps({"simhash": str(fp)}) + "\n")

    def _load_state(self) -> None:
        exact = self.state_dir / "exact_hashes.txt"
        if exact.exists():
            self.exact_hashes.update(line.strip() for line in exact.read_text(encoding="utf-8").splitlines() if line.strip())
        sim = self.state_dir / "simhashes.jsonl"
        if sim.exists():
            with sim.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.simhashes.append(int(json.loads(line)["simhash"]))


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def exact_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()


def simhash(text: str, *, bits: int = 64) -> int:
    vector = [0] * bits
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        tokens = [text.lower()]
    for token in tokens:
        h = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for i in range(bits):
            vector[i] += 1 if (h >> i) & 1 else -1
    fp = 0
    for i, value in enumerate(vector):
        if value >= 0:
            fp |= 1 << i
    return fp


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()
