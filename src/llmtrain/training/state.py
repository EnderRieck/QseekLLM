from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainerState:
    global_step: int = 0
    consumed_tokens: int = 0
    global_consumed_tokens: int = 0
    consumed_samples: int = 0
    optimizer_steps: int = 0
    best_metric: float | None = None
    last_validation_bucket: int = 0

    def state_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict | None) -> "TrainerState":
        if not state:
            return cls()
        return cls(
            global_step=int(state.get("global_step", 0)),
            consumed_tokens=int(state.get("consumed_tokens", 0)),
            global_consumed_tokens=int(state.get("global_consumed_tokens", state.get("consumed_tokens", 0))),
            consumed_samples=int(state.get("consumed_samples", 0)),
            optimizer_steps=int(state.get("optimizer_steps", 0)),
            best_metric=state.get("best_metric"),
            last_validation_bucket=int(state.get("last_validation_bucket", 0)),
        )
