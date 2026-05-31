from __future__ import annotations

from pathlib import Path
from typing import Any

from llmtrain.observability.sinks import HeartbeatSink, JsonlSink


class Phase1RunLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.metrics = JsonlSink(self.output_dir / "metrics.jsonl")
        self.data_metrics = JsonlSink(self.output_dir / "data_metrics.jsonl")
        self.val_metrics = JsonlSink(self.output_dir / "val_metrics.jsonl")
        self.events = JsonlSink(self.output_dir / "events.jsonl")
        self.heartbeat = HeartbeatSink(self.output_dir / "heartbeat.json")

    def metric(self, **record: Any) -> None:
        self.metrics.write(record)

    def data_metric(self, **record: Any) -> None:
        self.data_metrics.write(record)

    def val_metric(self, **record: Any) -> None:
        self.val_metrics.write(record)

    def event(self, event: str, **record: Any) -> None:
        self.events.write({"event": event, **record})

    def beat(self, **record: Any) -> None:
        self.heartbeat.write(**record)
