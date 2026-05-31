from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from llmtrain.checkpointing.manager import CheckpointManager
from llmtrain.inference.config import GenerationConfig, RuntimeConfig
from llmtrain.interfaces import Tokenizer
from llmtrain.models import build_model
from llmtrain.models.decoder import TransformerLM
from llmtrain.tokenizer.adapter import load_tokenizer
from llmtrain.utils.config import Config, load_config


@dataclass(frozen=True)
class GenerationStep:
    token_id: int
    text: str
    index: int
    stopped: bool = False


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    text: str
    token_ids: list[int]
    new_token_ids: list[int]
    stop_reason: str
    input_tokens: int
    generated_tokens: int


class InferenceEngine:
    def __init__(
        self,
        *,
        model: TransformerLM,
        tokenizer: Tokenizer,
        max_context_tokens: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_tokens = int(max_context_tokens)
        self.device = device

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        *,
        checkpoint_path: str | Path | None = None,
        runtime: RuntimeConfig | None = None,
    ) -> "InferenceEngine":
        cfg, _ = load_config(config_path)
        return cls.from_train_config(cfg, checkpoint_path=checkpoint_path, runtime=runtime)

    @classmethod
    def from_train_config(
        cls,
        cfg: Config,
        *,
        checkpoint_path: str | Path | None = None,
        runtime: RuntimeConfig | None = None,
    ) -> "InferenceEngine":
        runtime = runtime or RuntimeConfig()
        _validate_tokenizer_path(cfg, allow_fallback=runtime.allow_tokenizer_fallback)
        device = _select_device(runtime.device)
        dtype = _select_dtype(runtime.dtype, device=device, train_precision=cfg.trainer.precision)
        tokenizer = load_tokenizer(cfg.tokenizer)
        model = build_model(cfg.model)
        ckpt = _resolve_checkpoint(cfg, checkpoint_path)
        CheckpointManager(cfg.run.output_dir).load_model(ckpt, model=model, strict=True)
        model.to(device=device)
        if dtype is not None:
            model.to(dtype=dtype)
        model.eval()
        if runtime.compile_model:
            model = torch.compile(model)  # type: ignore[assignment]
        return cls(model=model, tokenizer=tokenizer, max_context_tokens=cfg.model.max_position_embeddings, device=device)

    @torch.inference_mode()
    def generate(self, prompt: str, cfg: GenerationConfig | None = None) -> GenerationResult:
        cfg = cfg or GenerationConfig()
        prompt_ids = self._encode_prompt(prompt, cfg)
        new_ids: list[int] = []
        stop_ids = self._stop_token_ids(cfg)
        stop_reason = "max_new_tokens"
        sampler = self._prepare_sampler(prompt_ids, cfg)
        for _idx in range(cfg.max_new_tokens):
            next_id = sampler(prompt_ids + new_ids)
            new_ids.append(next_id)
            if next_id in stop_ids:
                stop_reason = "stop_token"
                break
        output_ids = prompt_ids + new_ids if cfg.include_prompt else new_ids
        return GenerationResult(
            prompt=prompt,
            text=self.tokenizer.decode(output_ids),
            token_ids=output_ids,
            new_token_ids=new_ids,
            stop_reason=stop_reason,
            input_tokens=len(prompt_ids),
            generated_tokens=len(new_ids),
        )

    @torch.inference_mode()
    def iter_generate(self, prompt: str, cfg: GenerationConfig | None = None) -> Iterator[GenerationStep]:
        cfg = cfg or GenerationConfig()
        prompt_ids = self._encode_prompt(prompt, cfg)
        new_ids: list[int] = []
        decoded_text = ""
        stop_ids = self._stop_token_ids(cfg)
        sampler = self._prepare_sampler(prompt_ids, cfg)
        for idx in range(cfg.max_new_tokens):
            next_id = sampler(prompt_ids + new_ids)
            new_ids.append(next_id)
            stopped = next_id in stop_ids
            current_text = self.tokenizer.decode(new_ids)
            delta_text = current_text[len(decoded_text) :] if current_text.startswith(decoded_text) else current_text
            decoded_text = current_text
            yield GenerationStep(token_id=next_id, text=delta_text, index=idx, stopped=stopped)
            if stopped:
                break

    def _encode_prompt(self, prompt: str, cfg: GenerationConfig) -> list[int]:
        ids = self.tokenizer.encode(prompt)
        limit = cfg.max_input_tokens or self.max_context_tokens
        if len(ids) > limit:
            ids = ids[-limit:]
        if not ids:
            ids = [self.tokenizer.eot_id]
        return ids

    def _prepare_sampler(self, prompt_ids: list[int], cfg: GenerationConfig):
        if not cfg.use_kv_cache:
            return lambda token_ids: self._next_token(token_ids, cfg)
        context = prompt_ids[-self.max_context_tokens :]
        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
        output = self.model(input_ids, use_cache=True)
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("Model did not return KV cache")
        next_logits = output.logits[:, -1, :].float().squeeze(0)
        pending_logits: torch.Tensor | None = next_logits

        def sample(token_ids: list[int]) -> int:
            nonlocal cache, pending_logits
            if pending_logits is not None:
                logits = pending_logits
                pending_logits = None
            else:
                input_ids = torch.tensor([[token_ids[-1]]], dtype=torch.long, device=self.device)
                output = self.model(input_ids, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                if cache is None:
                    raise RuntimeError("Model did not return KV cache")
                logits = output.logits[:, -1, :].float().squeeze(0)
            return self._sample_from_logits(logits, token_ids, cfg)

        return sample

    def _next_token(self, token_ids: list[int], cfg: GenerationConfig) -> int:
        context = token_ids[-self.max_context_tokens :]
        input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
        document_ids = torch.zeros_like(input_ids)
        logits = self.model(input_ids, document_ids=document_ids).logits[:, -1, :].float().squeeze(0)
        return self._sample_from_logits(logits, token_ids, cfg)

    def _sample_from_logits(self, logits: torch.Tensor, token_ids: list[int], cfg: GenerationConfig) -> int:
        logits = _apply_repetition_penalty(logits, token_ids, cfg.repetition_penalty)
        if not cfg.do_sample or cfg.temperature == 0.0:
            return int(torch.argmax(logits).item())
        logits = logits / cfg.temperature
        logits = _filter_top_k_top_p(logits, top_k=cfg.top_k, top_p=cfg.top_p)
        probs = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
            return int(torch.argmax(logits).item())
        return int(torch.multinomial(probs, num_samples=1).item())

    def _stop_token_ids(self, cfg: GenerationConfig) -> set[int]:
        stop_ids = set(int(x) for x in cfg.stop_token_ids)
        if cfg.stop_on_eot:
            stop_ids.add(int(self.tokenizer.eot_id))
        return stop_ids


def _resolve_checkpoint(cfg: Config, checkpoint_path: str | Path | None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser()
    latest = CheckpointManager(cfg.run.output_dir).latest_checkpoint()
    if latest is None:
        raise FileNotFoundError(f"No latest checkpoint found under {Path(cfg.run.output_dir) / 'checkpoints'}")
    return latest


def _validate_tokenizer_path(cfg: Config, *, allow_fallback: bool) -> None:
    if allow_fallback:
        return
    if cfg.tokenizer.model_path is None:
        raise ValueError("tokenizer.model_path is required for inference unless allow_tokenizer_fallback=true")
    if not Path(cfg.tokenizer.model_path).exists():
        raise FileNotFoundError(f"Tokenizer model does not exist: {cfg.tokenizer.model_path}")


def _select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_dtype(requested: str, *, device: torch.device, train_precision: str) -> torch.dtype | None:
    if requested == "fp32":
        return torch.float32
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if device.type == "cuda":
        if train_precision == "bf16":
            return torch.bfloat16
        if train_precision == "fp16":
            return torch.float16
    return torch.float32


def _apply_repetition_penalty(logits: torch.Tensor, token_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty == 1.0 or not token_ids:
        return logits
    out = logits.clone()
    for token_id in set(token_ids):
        if token_id < 0 or token_id >= out.numel():
            continue
        out[token_id] = out[token_id] / penalty if out[token_id] > 0 else out[token_id] * penalty
    return out


def _filter_top_k_top_p(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0 and top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[sorted_indices[remove]] = float("-inf")
    return filtered
