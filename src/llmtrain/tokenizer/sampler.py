from __future__ import annotations

import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TextIO

from llmtrain.data.manifest import ShardInfo, load_manifest
from llmtrain.data.readers import ShardReader


def sample_tokenizer_corpus(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    byte_budget: int,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    validate_hashes: bool = True,
    output_shard_bytes: int | None = None,
    num_workers: int = 1,
) -> dict:
    ratios = ratios or {}
    out = Path(output_path)
    shards = load_manifest(manifest_path)
    total_ratio = sum(ratios.values())
    domain_budgets = {
        domain: int(byte_budget * ratio / total_ratio) for domain, ratio in ratios.items()
    } if ratios else {}
    workers = max(1, int(num_workers))
    if workers == 1:
        result = _sample_tokenizer_corpus_worker(
            manifest_path=Path(manifest_path),
            output_path=out,
            shards=shards,
            byte_budget=byte_budget,
            domain_budgets=domain_budgets,
            seed=seed,
            validate_hashes=validate_hashes,
            output_shard_bytes=output_shard_bytes,
            worker_id=0,
            worker_count=1,
        )
    else:
        result = _sample_tokenizer_corpus_parallel(
            manifest_path=Path(manifest_path),
            output_path=out,
            shards=shards,
            byte_budget=byte_budget,
            domain_budgets=domain_budgets,
            seed=seed,
            validate_hashes=validate_hashes,
            output_shard_bytes=output_shard_bytes,
            num_workers=workers,
        )
    result.update(
        {
            "byte_budget": byte_budget,
            "seed": seed,
            "sampling_strategy": "seeded_domain_shard_shuffle",
            "num_workers": workers,
            "output_path": str(out),
            "output_shard_bytes": output_shard_bytes,
            "num_manifest_shards": len(shards),
        }
    )
    return result


def _sample_tokenizer_corpus_parallel(
    *,
    manifest_path: Path,
    output_path: Path,
    shards: list[ShardInfo],
    byte_budget: int,
    domain_budgets: dict[str, int],
    seed: int,
    validate_hashes: bool,
    output_shard_bytes: int | None,
    num_workers: int,
) -> dict:
    worker_shards = _assign_worker_shards(shards, num_workers, seed)
    worker_budgets = _split_worker_budgets(byte_budget, domain_budgets, num_workers)
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(
                _sample_tokenizer_corpus_worker,
                manifest_path=manifest_path,
                output_path=output_path,
                shards=worker_shards[worker_id],
                byte_budget=worker_budgets[worker_id]["byte_budget"],
                domain_budgets=worker_budgets[worker_id]["domain_budgets"],
                seed=seed + worker_id,
                validate_hashes=validate_hashes,
                output_shard_bytes=output_shard_bytes,
                worker_id=worker_id,
                worker_count=num_workers,
            )
            for worker_id in range(num_workers)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return _merge_worker_results(results)


def _sample_tokenizer_corpus_worker(
    *,
    manifest_path: Path,
    output_path: Path,
    shards: list[ShardInfo],
    byte_budget: int,
    domain_budgets: dict[str, int],
    seed: int,
    validate_hashes: bool,
    output_shard_bytes: int | None,
    worker_id: int,
    worker_count: int,
) -> dict:
    writer = TokenizerCorpusWriter(
        output_path,
        shard_bytes=output_shard_bytes,
        worker_id=worker_id,
        worker_count=worker_count,
    )
    stats = {
        "bytes_written": 0,
        "records_written": 0,
        "domains": defaultdict(lambda: {"bytes": 0, "records": 0, "estimated_tokens": 0}),
        "sources": defaultdict(lambda: {"bytes": 0, "records": 0, "estimated_tokens": 0}),
    }

    rng = random.Random(seed)
    reader = ShardReader(manifest_path, validate_hashes=validate_hashes)
    with writer:
        if domain_budgets:
            shard_queues = _seeded_domain_shard_queues(shards, domain_budgets, rng)
            while stats["bytes_written"] < byte_budget:
                domains = [
                    domain
                    for domain, budget in domain_budgets.items()
                    if stats["domains"][domain]["bytes"] < budget and shard_queues.get(domain)
                ]
                if not domains:
                    break
                domain = rng.choice(domains)
                shard = shard_queues[domain].pop()
                if _sample_shard(reader, shard, writer, stats, byte_budget, domain_budgets):
                    break
        else:
            shuffled = list(shards)
            rng.shuffle(shuffled)
            for shard in shuffled:
                if _sample_shard(reader, shard, writer, stats, byte_budget, domain_budgets):
                    break
    stats["domains"] = dict(stats["domains"])
    stats["sources"] = dict(stats["sources"])
    stats["output_files"] = [str(path) for path in writer.paths]
    stats["worker_id"] = worker_id
    return stats


class TokenizerCorpusWriter:
    def __init__(
        self,
        output_path: Path,
        *,
        shard_bytes: int | None,
        worker_id: int = 0,
        worker_count: int = 1,
    ) -> None:
        self.output_path = output_path
        self.shard_bytes = shard_bytes
        self.worker_id = worker_id
        self.worker_count = worker_count
        self.paths: list[Path] = []
        self._file: TextIO | None = None
        self._current_bytes = 0
        self._next_index = 0
        self._directory_mode = output_path.suffix == "" or output_path.is_dir()

    def __enter__(self) -> "TokenizerCorpusWriter":
        if self._directory_mode:
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, text: str, encoded_len: int) -> None:
        if self._file is None or self._should_rotate(encoded_len):
            self._open_next()
        assert self._file is not None
        self._file.write(text)
        self._current_bytes += encoded_len

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _should_rotate(self, encoded_len: int) -> bool:
        if not self._directory_mode or not self.shard_bytes:
            return False
        return self._current_bytes > 0 and self._current_bytes + encoded_len > self.shard_bytes

    def _open_next(self) -> None:
        self.close()
        if self._directory_mode:
            prefix = f"part-{self.worker_id:03d}-" if self.worker_count > 1 else "part-"
            path = self.output_path / f"{prefix}{self._next_index:05d}.txt"
            self._next_index += 1
        else:
            path = self.output_path
            if self.paths:
                path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._current_bytes = 0
        self.paths.append(path)


def _assign_worker_shards(shards: list[ShardInfo], num_workers: int, seed: int) -> list[list[ShardInfo]]:
    shuffled = list(shards)
    random.Random(seed).shuffle(shuffled)
    assigned = [[] for _ in range(num_workers)]
    for index, shard in enumerate(shuffled):
        assigned[index % num_workers].append(shard)
    return assigned


def _split_worker_budgets(
    byte_budget: int,
    domain_budgets: dict[str, int],
    num_workers: int,
) -> list[dict[str, Any]]:
    results = [{"byte_budget": 0, "domain_budgets": {}} for _ in range(num_workers)]
    for worker_id in range(num_workers):
        results[worker_id]["byte_budget"] = byte_budget // num_workers + (1 if worker_id < byte_budget % num_workers else 0)
    for domain, budget in domain_budgets.items():
        for worker_id in range(num_workers):
            results[worker_id]["domain_budgets"][domain] = budget // num_workers + (
                1 if worker_id < budget % num_workers else 0
            )
    return results


def _merge_worker_results(results: list[dict]) -> dict:
    merged = {
        "bytes_written": 0,
        "records_written": 0,
        "domains": defaultdict(lambda: {"bytes": 0, "records": 0, "estimated_tokens": 0}),
        "sources": defaultdict(lambda: {"bytes": 0, "records": 0, "estimated_tokens": 0}),
        "output_files": [],
        "workers": [],
    }
    for result in sorted(results, key=lambda item: item["worker_id"]):
        merged["bytes_written"] += result["bytes_written"]
        merged["records_written"] += result["records_written"]
        merged["output_files"].extend(result["output_files"])
        merged["workers"].append(
            {
                "worker_id": result["worker_id"],
                "bytes_written": result["bytes_written"],
                "records_written": result["records_written"],
                "output_files": result["output_files"],
            }
        )
        for bucket in ("domains", "sources"):
            for key, item in result[bucket].items():
                for metric, value in item.items():
                    merged[bucket][key][metric] += value
    merged["domains"] = dict(merged["domains"])
    merged["sources"] = dict(merged["sources"])
    return merged


def _seeded_domain_shard_queues(
    shards: list[ShardInfo],
    domain_budgets: dict[str, int],
    rng: random.Random,
) -> dict[str, list[ShardInfo]]:
    queues = {domain: [shard for shard in shards if shard.domain == domain] for domain in domain_budgets}
    for queue in queues.values():
        rng.shuffle(queue)
    return queues


def _sample_shard(
    reader: ShardReader,
    shard: ShardInfo,
    output: TokenizerCorpusWriter,
    stats: dict,
    byte_budget: int,
    domain_budgets: dict[str, int],
) -> bool:
    reader.shards = [shard]
    reader.shard_index = 0
    reader.shard_byte_offset = 0
    reader.shard_record_offset = 0

    for record in reader:
        text = record.text.strip()
        if not text:
            continue
        encoded_len = len((text + "\n").encode("utf-8"))
        if stats["bytes_written"] + encoded_len > byte_budget:
            return True
        if domain_budgets:
            dstat = stats["domains"][record.domain]
            if dstat["bytes"] + encoded_len > domain_budgets.get(record.domain, 0):
                return False
        output.write(text.replace("\n", " ") + "\n", encoded_len)
        est = max(1, encoded_len // 4)
        stats["bytes_written"] += encoded_len
        stats["records_written"] += 1
        for bucket, key in (("domains", record.domain), ("sources", record.source)):
            stats[bucket][key]["bytes"] += encoded_len
            stats[bucket][key]["records"] += 1
            stats[bucket][key]["estimated_tokens"] += est
    return False
