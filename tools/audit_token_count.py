"""Audit: tokenize a sample of shards from a manifest and compare to estimated_tokens.

For each source, randomly sample N shards, tokenize them in parallel worker
processes (using the project's HF byte-BPE tokenizer), accumulate real-token
counts, and compare to the shard's stated `estimated_tokens` (bytes/4).

Output: per-source mean ratio (real / est) + extrapolated total real tokens
for the entire training manifest.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmtrain.data.manifest import load_manifest  # noqa: E402
from llmtrain.tokenizer.adapter import load_tokenizer  # noqa: E402
from llmtrain.utils.config import load_config  # noqa: E402


_TOKENIZER = None  # per-worker cache


def _worker_init(config_path: str) -> None:
    global _TOKENIZER
    cfg, _ = load_config(config_path)
    _TOKENIZER = load_tokenizer(cfg.tokenizer)


def _tokenize_shard(payload: dict) -> dict:
    """Tokenize a single shard slice [record_start, record_end). Return real-token totals."""
    global _TOKENIZER
    import pyarrow.parquet as pq

    uri = payload["uri"]
    record_start = payload["record_start"]
    record_end = payload["record_end"]  # may be None
    fmt = payload["format"]
    source = payload["source"]
    est_tokens = payload["estimated_tokens"]
    sha256 = payload["sha256"]

    real_tokens = 0
    n_records = 0
    n_chars = 0
    n_bytes = 0

    if fmt == "parquet":
        seen = 0
        pf = pq.ParquetFile(uri)
        for batch in pf.iter_batches(columns=["text"], batch_size=4096):
            for row in batch.to_pylist():
                if seen < record_start:
                    seen += 1
                    continue
                if record_end is not None and seen >= record_end:
                    seen += 1
                    break
                text = row.get("text") or ""
                if text:
                    ids = _TOKENIZER.encode(text)
                    real_tokens += len(ids)
                    n_chars += len(text)
                    n_bytes += len(text.encode("utf-8"))
                    n_records += 1
                seen += 1
            else:
                continue
            break  # break outer when inner breaks via record_end
    else:  # jsonl
        with Path(uri).open("r", encoding="utf-8") as f:
            seen = 0
            for line in f:
                if not line.strip():
                    continue
                if seen < record_start:
                    seen += 1
                    continue
                if record_end is not None and seen >= record_end:
                    break
                d = json.loads(line)
                text = d.get("text") or ""
                if text:
                    ids = _TOKENIZER.encode(text)
                    real_tokens += len(ids)
                    n_chars += len(text)
                    n_bytes += len(text.encode("utf-8"))
                    n_records += 1
                seen += 1

    return {
        "source": source,
        "sha256": sha256,
        "uri": uri,
        "real_tokens": real_tokens,
        "n_records": n_records,
        "n_chars": n_chars,
        "n_bytes": n_bytes,
        "est_tokens": est_tokens,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True, help="Train config (used to load tokenizer).")
    p.add_argument("--output", required=True)
    p.add_argument("--samples-per-source", type=int, default=5)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    all_shards = load_manifest(args.manifest)
    by_source: dict[str, list] = defaultdict(list)
    for s in all_shards:
        by_source[s.source].append(s)

    rng = random.Random(args.seed)
    sample_payload: list[dict] = []
    sample_meta: list[dict] = []
    for src in sorted(by_source):
        shards = list(by_source[src])
        rng.shuffle(shards)
        chosen = shards[: args.samples_per_source]
        for s in chosen:
            sample_payload.append({
                "source": s.source,
                "sha256": s.sha256,
                "uri": s.uri,
                "record_start": s.record_start,
                "record_end": s.record_end,
                "format": s.format,
                "estimated_tokens": s.estimated_tokens,
            })
            sample_meta.append({
                "source": s.source, "bytes": s.bytes,
                "est_tokens": s.estimated_tokens,
            })
        print(f"  {src}: {len(chosen)} shards sampled (of {len(shards)})")

    print(f"\nTotal samples: {len(sample_payload)}")
    print(f"Starting {args.workers} workers...\n")

    per_source = defaultdict(lambda: {"shards": 0, "real": 0, "est": 0, "chars": 0, "bytes": 0, "records": 0})
    t0 = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init, initargs=(args.config,)) as ex:
        futs = [ex.submit(_tokenize_shard, p) for p in sample_payload]
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                print(f"  shard failed: {e}")
                continue
            acc = per_source[r["source"]]
            acc["shards"] += 1
            acc["real"] += r["real_tokens"]
            acc["est"] += r["est_tokens"]
            acc["chars"] += r["n_chars"]
            acc["bytes"] += r["n_bytes"]
            acc["records"] += r["n_records"]
            completed += 1
            if completed % 10 == 0 or completed == len(sample_payload):
                elapsed = time.time() - t0
                rate = completed / elapsed
                print(f"  [{completed}/{len(sample_payload)}] {rate:.1f} shards/s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print()

    # Per-source report and extrapolation to full manifest
    rows = []
    full_real_extrapolated_total = 0
    full_est_total = 0
    print(f"{'source':<28}{'sampled_shards':>15}{'real/est':>10}{'chars/tok':>10}{'real_per_byte':>14}{'src_total_est(B)':>17}{'extrapolated_real(B)':>22}")
    print("-" * 130)
    for src in sorted(per_source):
        acc = per_source[src]
        if acc["est"] == 0:
            continue
        ratio = acc["real"] / acc["est"]
        chars_per_tok = acc["chars"] / max(1, acc["real"])
        real_per_byte = acc["real"] / max(1, acc["bytes"])
        src_total_est = sum(s.estimated_tokens for s in by_source[src])
        extrapolated = int(src_total_est * ratio)
        full_real_extrapolated_total += extrapolated
        full_est_total += src_total_est
        rows.append({
            "source": src,
            "sampled_shards": acc["shards"],
            "real_tokens_sampled": acc["real"],
            "est_tokens_sampled": acc["est"],
            "real_to_est_ratio": ratio,
            "chars_per_token": chars_per_tok,
            "real_per_byte": real_per_byte,
            "src_total_est": src_total_est,
            "extrapolated_total_real": extrapolated,
        })
        print(
            f"{src:<28}{acc['shards']:>15}{ratio:>10.3f}{chars_per_tok:>10.2f}{real_per_byte:>14.3f}"
            f"{src_total_est/1e9:>15.2f}B{extrapolated/1e9:>20.2f}B"
        )
    print("-" * 130)
    overall_ratio = full_real_extrapolated_total / max(1, full_est_total)
    print(f"{'TOTAL':<28}{'':>15}{overall_ratio:>10.3f}{'':>10}{'':>14}{full_est_total/1e9:>15.2f}B{full_real_extrapolated_total/1e9:>20.2f}B")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "manifest": args.manifest,
        "samples_per_source": args.samples_per_source,
        "wall_seconds": elapsed,
        "per_source": rows,
        "totals": {
            "manifest_est_tokens": full_est_total,
            "extrapolated_real_tokens": full_real_extrapolated_total,
            "overall_real_to_est_ratio": overall_ratio,
        },
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
