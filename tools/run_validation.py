"""Per-source validation loss / PPL evaluator.

Loads a checkpoint, iterates the val manifest grouped by source, tokenizes,
packs to (1, seq_len), forwards in eval mode, accumulates per-source
sum-of-CE-loss and token counts. Outputs JSON with per-source CE / PPL.

Usage:
  python tools/run_validation.py \
    --config configs/train/stage1_general.yaml \
    --checkpoint runs/stage1_general/checkpoints/milestone_030000000000 \
    --val-manifest /mnt/DataFlow/.../val_60m/manifest.jsonl \
    --output runs/validation_loss/milestone_030.json \
    --max-tokens-per-source 6000000 \
    --seq-len 4096
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmtrain.checkpointing.manager import CheckpointManager  # noqa: E402
from llmtrain.data.manifest import load_manifest  # noqa: E402
from llmtrain.evaluation.eval_utils import evaluate_source  # noqa: E402
from llmtrain.models import build_model  # noqa: E402
from llmtrain.tokenizer.adapter import load_tokenizer  # noqa: E402
from llmtrain.utils.config import load_config  # noqa: E402


def _select_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--max-tokens-per-source", type=int, default=6_000_000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--source", default=None, help="Only evaluate this single source (filename suffix added to output).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    cfg, _ = load_config(args.config)
    device = torch.device(args.device)
    dtype = _select_dtype(args.dtype)

    print(f"Loading tokenizer ...", flush=True)
    tokenizer = load_tokenizer(cfg.tokenizer)
    eot_id = tokenizer.eot_id

    print(f"Building model ...", flush=True)
    model = build_model(cfg.model)

    print(f"Loading checkpoint {args.checkpoint} ...", flush=True)
    CheckpointManager(cfg.run.output_dir).load_model(args.checkpoint, model=model, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()

    val_shards = load_manifest(args.val_manifest)
    by_source = defaultdict(list)
    for s in val_shards:
        by_source[s.source].append(s)

    if args.source is not None:
        if args.source not in by_source:
            print(f"ERROR: source '{args.source}' not in val manifest. Available: {sorted(by_source)}", flush=True)
            sys.exit(1)
        by_source = {args.source: by_source[args.source]}

    results = {}
    t0 = time.time()
    for src in sorted(by_source):
        print(f"\n--- evaluating source: {src} ({len(by_source[src])} shards) ---", flush=True)
        st = time.time()
        try:
            res = evaluate_source(
                model=model,
                tokenizer=tokenizer,
                shards=by_source[src],
                seq_len=args.seq_len,
                max_tokens=args.max_tokens_per_source,
                device=device,
                dtype=dtype,
                eot_id=eot_id,
                batch_size=args.batch_size,
            )
        except Exception as e:
            res = {"error": str(e)}
        dt = time.time() - st
        res["wall_seconds"] = dt
        results[src] = res
        if "error" not in res:
            print(
                f"  {src}: packs={res['n_packs']}  tokens={res['n_tokens_eval']}  CE={res['mean_ce']:.4f}  PPL={res['ppl']:.2f}  ({dt:.1f}s)",
                flush=True,
            )
        else:
            print(f"  {src}: ERROR {res['error']}", flush=True)

    overall_sum_loss = sum(r.get("sum_loss_x_count", 0.0) for r in results.values() if "error" not in r)
    overall_count = sum(r.get("n_tokens_eval", 0) for r in results.values() if "error" not in r)
    overall_ce = overall_sum_loss / max(1, overall_count)
    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "val_manifest": args.val_manifest,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "max_tokens_per_source": args.max_tokens_per_source,
        "dtype": args.dtype,
        "per_source": results,
        "overall": {
            "mean_ce": overall_ce,
            "ppl": math.exp(overall_ce) if overall_ce < 50 else float("inf"),
            "n_tokens_eval": overall_count,
        },
        "wall_seconds": time.time() - t0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"Overall CE={overall_ce:.4f} PPL={summary['overall']['ppl']:.2f}")


if __name__ == "__main__":
    main()
