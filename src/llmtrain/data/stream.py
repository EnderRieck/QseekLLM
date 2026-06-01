from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llmtrain.data.config import DataSourceConfig, PipelineConfig
from llmtrain.data.manifest import ManifestMeta, load_manifest, validate_manifest
from llmtrain.data.mixer import WeightedMixer
from llmtrain.data.pipeline import RecordPipeline
from llmtrain.data.readers import ShardReader
from llmtrain.interfaces import Record


class TrainingRecordStream:
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        sources: list[DataSourceConfig | dict[str, Any]] | None,
        pipeline_cfg: PipelineConfig | dict[str, Any],
        world_size: int = 1,
        rank: int = 0,
        num_workers: int = 1,
        worker_id: int = 0,
        validate_hashes: bool = True,
        shuffle_seed: int | None = None,
        parquet_batch_size: int = 8192,
        mixer_temperature: float = 1.0,
        manifest_meta: ManifestMeta | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.pipeline_cfg = PipelineConfig.model_validate(pipeline_cfg)
        self.world_size = world_size
        self.rank = rank
        self.num_workers = num_workers
        self.worker_id = worker_id
        self.validate_hashes = validate_hashes
        self.shuffle_seed = shuffle_seed
        self.parquet_batch_size = parquet_batch_size
        self.mixer_temperature = mixer_temperature
        self.manifest_meta = manifest_meta or validate_manifest(self.manifest_path, validate_shards=validate_hashes)
        self._manifest_shards = load_manifest(self.manifest_path)
        self._sources = [DataSourceConfig.model_validate(source) for source in (sources or [])]
        self._mode = "mixed" if self._sources else "single"
        self._loaded_state: dict[str, Any] | None = None
        self._reader: ShardReader | None = None
        self._records: Iterator[Record] | None = None
        self._mixers: WeightedMixer | None = None
        self._readers: dict[str, ShardReader] = {}
        self._source_records: dict[str, Iterator[Record]] = {}
        self._build()

    def __iter__(self) -> Iterator[Record]:
        assert self._records is not None
        return self._records

    def state_dict(self) -> dict[str, Any]:
        if self._mode == "single":
            assert self._reader is not None
            return {
                "mode": self._mode,
                "reader": self._reader.state_dict(),
            }
        assert self._mixers is not None
        return {
            "mode": self._mode,
            "mixer": self._mixers.state_dict(),
            "readers": {name: reader.state_dict() for name, reader in self._readers.items()},
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._loaded_state = sd
        if sd.get("mode") != self._mode:
            raise ValueError(f"TrainingRecordStream mode mismatch: expected {self._mode}, got {sd.get('mode')}")
        if self._mode == "single":
            assert self._reader is not None
            self._reader.load_state_dict(sd["reader"])
            return
        assert self._mixers is not None
        for name, reader in self._readers.items():
            reader_state = sd.get("readers", {}).get(name)
            if reader_state is None:
                raise ValueError(f"Missing reader state for source: {name}")
            reader.load_state_dict(reader_state)
        self._mixers.load_state_dict(sd["mixer"])

    def _build(self) -> None:
        if self._mode == "single":
            self._reader = ShardReader(
                self.manifest_path,
                world_size=self.world_size,
                rank=self.rank,
                num_workers=self.num_workers,
                worker_id=self.worker_id,
                validate_hashes=self.validate_hashes,
                shuffle_seed=self.shuffle_seed,
                parquet_batch_size=self.parquet_batch_size,
                manifest_meta=self.manifest_meta,
            )
            if self._loaded_state is not None:
                self._reader.load_state_dict(self._loaded_state["reader"])
            self._records = RecordPipeline.from_config(self.pipeline_cfg).apply(self._reader)
            return

        active_sources = [source for source in self._sources if source.weight > 0]
        source_weights = {source.name: source.weight for source in active_sources}
        self._readers = {}
        self._source_records = {}
        for source in active_sources:
            matching_shards = [
                shard
                for shard in self._manifest_shards
                if shard.domain == source.domain and (source.source_filter is None or shard.source == source.source_filter)
            ]
            if not matching_shards:
                continue
            reader = ShardReader(
                self.manifest_path,
                world_size=self.world_size,
                rank=self.rank,
                num_workers=self.num_workers,
                worker_id=self.worker_id,
                validate_hashes=self.validate_hashes,
                shuffle_seed=self.shuffle_seed,
                parquet_batch_size=self.parquet_batch_size,
                source_filter=source.source_filter,
                domain_filter=source.domain,
                manifest_meta=self.manifest_meta,
            )
            if not reader.shards:
                continue
            self._readers[source.name] = reader
            self._source_records[source.name] = RecordPipeline.from_config(self.pipeline_cfg).apply(reader)
        if not self._source_records:
            raise ValueError("No active sources with assigned shards were found in the manifest")
        self._mixers = WeightedMixer(
            self._source_records,
            source_weights,
            temperature=self.mixer_temperature,
            seed=self.shuffle_seed or 42,
        )
        if self._loaded_state is not None:
            self.load_state_dict(self._loaded_state)
        self._records = iter(self._mixers)
