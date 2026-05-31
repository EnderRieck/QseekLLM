from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PositiveInt

from llmtrain.utils.config import apply_overrides


class RuntimeConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["auto", "fp32", "bf16", "fp16"] = "auto"
    compile_model: bool = False
    allow_tokenizer_fallback: bool = False

    model_config = {"extra": "forbid"}


class GenerationConfig(BaseModel):
    max_new_tokens: PositiveInt = 128
    temperature: float = Field(0.8, ge=0.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    do_sample: bool = True
    repetition_penalty: float = Field(1.0, ge=1.0)
    include_prompt: bool = False
    stop_on_eot: bool = True
    stop_token_ids: list[int] = Field(default_factory=list)
    max_input_tokens: PositiveInt | None = None
    use_kv_cache: bool = True

    model_config = {"extra": "forbid"}


class BatchInferenceConfig(BaseModel):
    prompt_field: str = "prompt"
    output_field: str = "completion"
    include_metadata: bool = True

    model_config = {"extra": "forbid"}


class InferenceConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    batch: BatchInferenceConfig = Field(default_factory=BatchInferenceConfig)

    model_config = {"extra": "forbid"}


def load_inference_config(path: str | Path | None = None, overrides: list[str] | None = None) -> InferenceConfig:
    data = _load_yaml(path) if path else {}
    data = apply_overrides(data, overrides)
    return InferenceConfig.model_validate(data)


def _load_yaml(path: str | Path | None) -> dict:
    if path is None:
        return {}
    p = Path(path).expanduser()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Inference config must be a mapping: {p}")
    return raw
