"""In-training validation callback.

Triggered every `interval_tokens` of training tokens, runs a per-source CE/PPL
forward over a held-out val manifest, and emits results to `val_metrics.jsonl`.
The Trainer owns the schedule trigger and `model.train()` mode; this callback
flips to `model.eval()` for the duration of the forward pass and restores
`.train()` on exit.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import torch

from llmtrain.data.manifest import load_manifest
from llmtrain.evaluation.eval_utils import evaluate_source
from llmtrain.training.validation_config import ValidationConfig


_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class ValidationCallback:
    def __init__(
        self,
        *,
        cfg: ValidationConfig,
        tokenizer,
        device: torch.device,
        logger,
        distributed=None,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger
        self.distributed = distributed
        self._dtype = _DTYPE_MAP[cfg.dtype]

        if cfg.val_manifest is None:
            raise ValueError("ValidationCallback requires cfg.val_manifest")
        shards = load_manifest(str(cfg.val_manifest))
        by_source: dict[str, list] = defaultdict(list)
        for s in shards:
            by_source[s.source].append(s)
        self._by_source: dict[str, list] = dict(sorted(by_source.items()))
        self._last_bucket: int = 0
        self._has_run_once: bool = False

    @property
    def _is_main(self) -> bool:
        return self.distributed is None or self.distributed.is_main

    def state_dict(self) -> dict:
        return {"last_bucket": self._last_bucket, "has_run_once": self._has_run_once}

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self._last_bucket = int(state.get("last_bucket", 0))
        self._has_run_once = bool(state.get("has_run_once", False))

    def maybe_run(
        self,
        *,
        model,
        global_consumed_tokens: int,
        global_step: int,
    ) -> None:
        if not self.cfg.enabled:
            return
        bucket = global_consumed_tokens // self.cfg.interval_tokens
        run_due = bucket > self._last_bucket
        run_initial = self.cfg.run_at_start and not self._has_run_once
        if not (run_due or run_initial):
            return
        self._run(
            model=model,
            global_consumed_tokens=global_consumed_tokens,
            global_step=global_step,
            trigger="bucket" if run_due else "start",
        )
        self._last_bucket = bucket
        self._has_run_once = True

    def _run(
        self,
        *,
        model,
        global_consumed_tokens: int,
        global_step: int,
        trigger: str,
    ) -> None:
        was_training = model.training
        model.eval()
        try:
            t_start = time.monotonic()
            per_source: dict[str, dict] = {}
            sum_loss_x_count = 0.0
            sum_count = 0
            if self._is_main and self.logger is not None:
                self.logger.event(
                    "validation_started",
                    global_step=global_step,
                    consumed_tokens=global_consumed_tokens,
                    trigger=trigger,
                )
            for src, shards in self._by_source.items():
                src_t0 = time.monotonic()
                try:
                    res = evaluate_source(
                        model=model,
                        tokenizer=self.tokenizer,
                        shards=shards,
                        seq_len=self.cfg.seq_len,
                        max_tokens=self.cfg.max_tokens_per_source,
                        device=self.device,
                        dtype=self._dtype,
                        eot_id=self.tokenizer.eot_id,
                        batch_size=self.cfg.batch_size,
                    )
                except Exception as e:
                    res = {"error": str(e)}
                res["wall_seconds"] = time.monotonic() - src_t0
                per_source[src] = res
                if "error" not in res:
                    sum_loss_x_count += res.get("sum_loss_x_count", 0.0)
                    sum_count += res.get("n_tokens_eval", 0)
                if self._is_main and self.logger is not None:
                    self.logger.val_metric(
                        global_step=global_step,
                        consumed_tokens=global_consumed_tokens,
                        source=src,
                        **{k: v for k, v in res.items() if k != "sum_loss_x_count"},
                    )

            overall_ce = sum_loss_x_count / max(1, sum_count)
            import math
            overall_ppl = math.exp(overall_ce) if overall_ce < 50 else float("inf")
            wall = time.monotonic() - t_start
            if self._is_main and self.logger is not None:
                self.logger.val_metric(
                    global_step=global_step,
                    consumed_tokens=global_consumed_tokens,
                    source="__overall__",
                    mean_ce=overall_ce,
                    ppl=overall_ppl,
                    n_tokens_eval=sum_count,
                    wall_seconds=wall,
                )
                self.logger.event(
                    "validation_complete",
                    global_step=global_step,
                    consumed_tokens=global_consumed_tokens,
                    overall_ce=overall_ce,
                    overall_ppl=overall_ppl,
                    n_tokens_eval=sum_count,
                    wall_seconds=wall,
                    trigger=trigger,
                )
                err_count = sum(1 for r in per_source.values() if "error" in r)
                summary = (
                    f"validation @ tokens={global_consumed_tokens} step={global_step} "
                    f"overall_ce={overall_ce:.4f} ppl={overall_ppl:.2f} "
                    f"sources={len(per_source)} errors={err_count} "
                    f"({wall:.1f}s)"
                )
                print(summary, flush=True)
        finally:
            if was_training:
                model.train()
