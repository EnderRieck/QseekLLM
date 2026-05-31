from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist

from llmtrain.checkpointing.manager import CheckpointManager
from llmtrain.inference.config import RuntimeConfig
from llmtrain.inference.engine import _select_dtype, _validate_tokenizer_path
from llmtrain.interfaces import Tokenizer
from llmtrain.models import build_model
from llmtrain.models.decoder import TransformerLM
from llmtrain.tokenizer.adapter import load_tokenizer
from llmtrain.utils.config import Config, load_config

try:  # Optional dependency, installed via llmtrain[eval].
    from lm_eval.api.model import LM as HarnessLM
except Exception:  # pragma: no cover - exercised when optional dependency is absent
    HarnessLM = object  # type: ignore[assignment, misc]


class LogLikelihoodScorer:
    def __init__(
        self,
        *,
        model: TransformerLM,
        tokenizer: Tokenizer,
        max_context_tokens: int,
        device: torch.device,
        batch_size: int = 1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_tokens = int(max_context_tokens)
        self.device = device
        self.batch_size = int(batch_size)
        self.pad_token_id = int(getattr(tokenizer, "pad_id", None) or tokenizer.eot_id)

    def loglikelihood(self, requests: Iterable[tuple[str, str]]) -> list[tuple[float, bool]]:
        prepared = [self._prepare_request(context, continuation) for context, continuation in requests]
        results: list[tuple[float, bool]] = []
        for start in range(0, len(prepared), self.batch_size):
            results.extend(self._score_batch(prepared[start : start + self.batch_size]))
        return results

    def loglikelihood_rolling(self, texts: Iterable[str]) -> list[float]:
        out: list[float] = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if not tokens:
                out.append(0.0)
                continue
            total = 0.0
            pos = 0
            stride = max(1, self.max_context_tokens - 1)
            while pos < len(tokens):
                chunk = tokens[pos : pos + stride]
                request_tokens = [self.tokenizer.eot_id] + chunk
                score_start = 1
                total += self._score_prepared(request_tokens, score_start)[0]
                pos += stride
            out.append(total)
        return out

    def _prepare_request(self, context: str, continuation: str) -> tuple[list[int], int]:
        context_ids = self.tokenizer.encode(context)
        whole_ids = self.tokenizer.encode(context + continuation)
        continuation_ids = whole_ids[len(context_ids) :]
        if not continuation_ids:
            return [self.tokenizer.eot_id], 1
        tokens = context_ids + continuation_ids
        score_start = len(context_ids)
        overflow = len(tokens) - self.max_context_tokens
        if overflow > 0:
            tokens = tokens[overflow:]
            score_start = max(0, score_start - overflow)
        if score_start == 0:
            tokens = [self.tokenizer.eot_id] + tokens
            score_start = 1
        return tokens, score_start

    def _score_prepared(self, tokens: list[int], score_start: int) -> tuple[float, bool]:
        return self._score_batch([(tokens, score_start)])[0]

    @torch.inference_mode()
    def _score_batch(self, batch: list[tuple[list[int], int]]) -> list[tuple[float, bool]]:
        if not batch:
            return []
        max_len = max(len(tokens) for tokens, _ in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long, device=self.device)
        for row, (tokens, _) in enumerate(batch):
            input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=self.device)
        document_ids = torch.zeros_like(input_ids)
        logits = self.model(input_ids, document_ids=document_ids).logits.float()
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        greedy = torch.argmax(logits[:, :-1, :], dim=-1)
        results: list[tuple[float, bool]] = []
        for row, (tokens, score_start) in enumerate(batch):
            logprob = 0.0
            is_greedy = True
            for token_pos in range(score_start, len(tokens)):
                label = int(input_ids[row, token_pos].item())
                pred_pos = token_pos - 1
                logprob += float(log_probs[row, pred_pos, label].item())
                is_greedy = is_greedy and int(greedy[row, pred_pos].item()) == label
            results.append((logprob, is_greedy))
        return results


class LLMTrainHarnessLM(HarnessLM):  # type: ignore[misc, valid-type]
    def __init__(
        self,
        *,
        model: TransformerLM,
        tokenizer: Tokenizer,
        max_context_tokens: int,
        device: torch.device,
        batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        tokenizer_name: str = "llmtrain",
    ) -> None:
        if HarnessLM is object:
            raise ImportError("lm-evaluation-harness is not installed. Install with: pip install -e '.[eval]'")
        super().__init__()
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._device = device
        self._tokenizer_name = tokenizer_name
        self.scorer = LogLikelihoodScorer(
            model=model,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            device=device,
            batch_size=batch_size,
        )
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_train_config_path(
        cls,
        config_path: str | Path,
        *,
        checkpoint_path: str | Path | None = None,
        runtime: RuntimeConfig | None = None,
        batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
    ) -> "LLMTrainHarnessLM":
        cfg, _ = load_config(config_path)
        return cls.from_train_config(
            cfg,
            checkpoint_path=checkpoint_path,
            runtime=runtime,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )

    @classmethod
    def from_train_config(
        cls,
        cfg: Config,
        *,
        checkpoint_path: str | Path | None = None,
        runtime: RuntimeConfig | None = None,
        batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
    ) -> "LLMTrainHarnessLM":
        runtime = runtime or RuntimeConfig()
        _validate_tokenizer_path(cfg, allow_fallback=runtime.allow_tokenizer_fallback)
        device = _select_eval_device(runtime.device, local_rank=local_rank)
        dtype = _select_dtype(runtime.dtype, device=device, train_precision=cfg.trainer.precision)
        tokenizer = load_tokenizer(cfg.tokenizer)
        model = build_model(cfg.model)
        ckpt = resolve_eval_checkpoint(cfg, checkpoint_path)
        CheckpointManager(cfg.run.output_dir).load_model(ckpt, model=model, strict=True)
        model.to(device=device)
        if dtype is not None:
            model.to(dtype=dtype)
        model.eval()
        if runtime.compile_model:
            model = torch.compile(model)  # type: ignore[assignment]
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_context_tokens=cfg.model.max_position_embeddings,
            device=device,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            tokenizer_name=str(cfg.tokenizer.model_path or tokenizer.metadata().get("type", "llmtrain")),
        )

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        return self.scorer.loglikelihood(_request_args(requests))

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        texts = [str(_instance_args(req)[0]) for req in requests]
        return self.scorer.loglikelihood_rolling(texts)

    def generate_until(self, requests: list[Any]) -> list[str]:
        raise NotImplementedError("LLMTrainHarnessLM currently supports loglikelihood benchmarks only")

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size <= 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.stack(gathered) if tensor.dim() == 0 else torch.cat(gathered, dim=0)

    def gather_object(self, obj: Any, dst: int = 0) -> list[Any] | None:
        if self.world_size <= 1:
            return [obj]
        gathered = [None for _ in range(self.world_size)] if self.rank == dst else None
        dist.gather_object(obj, object_gather_list=gathered, dst=dst)
        return gathered

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()


def resolve_eval_checkpoint(cfg: Config, checkpoint_path: str | Path | None) -> Path:
    if checkpoint_path is not None and str(checkpoint_path) != "latest":
        return Path(checkpoint_path).expanduser()
    latest = CheckpointManager(cfg.run.output_dir).latest_checkpoint()
    if latest is None:
        raise FileNotFoundError(f"No latest checkpoint found under {Path(cfg.run.output_dir) / 'checkpoints'}")
    return latest


def _select_eval_device(requested: str, *, local_rank: int) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _instance_args(req: Any) -> tuple[Any, ...]:
    args = getattr(req, "args", req)
    if not isinstance(args, tuple):
        args = tuple(args)
    return args


def _request_args(requests: list[Any]) -> Iterable[tuple[str, str]]:
    for req in requests:
        args = _instance_args(req)
        yield str(args[0]), str(args[1])
