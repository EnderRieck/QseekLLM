"""One-off: sample shards from a manifest, tokenize each document, and report the
per-document token-length distribution + how much of the corpus (docs and tokens)
sits above the 4096 / 8192 / 16384 context thresholds."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmtrain.data.manifest import load_manifest  # noqa: E402
from llmtrain.tokenizer.adapter import load_tokenizer  # noqa: E402
from llmtrain.utils.config import load_config  # noqa: E402

_TOK = None
THRESH = [2048, 4096, 8192, 16384, 32768]


def _init(config_path: str) -> None:
    global _TOK
    cfg, _ = load_config(config_path)
    _TOK = load_tokenizer(cfg.tokenizer)


def _count_shard(uri: str) -> dict:
    global _TOK
    import gzip
    lengths: list[int] = []
    opener = gzip.open if uri.endswith(".gz") else open
    with opener(uri, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "")
            if not text:
                continue
            lengths.append(len(_TOK.encode(text)))
    return {"uri": uri, "lengths": lengths}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--shards", type=int, default=20)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    shards = load_manifest(args.manifest)
    rng = random.Random(args.seed)
    rng.shuffle(shards)
    chosen = shards[: args.shards]
    print(f"sampling {len(chosen)} / {len(shards)} shards", flush=True)

    all_len: list[int] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.config,)) as ex:
        futs = [ex.submit(_count_shard, s.uri) for s in chosen]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            all_len.extend(r["lengths"])
            if i % 5 == 0 or i == len(chosen):
                print(f"  [{i}/{len(chosen)}] docs so far: {len(all_len):,}", flush=True)

    all_len.sort()
    n = len(all_len)
    total_tok = sum(all_len)

    def pct(q: float) -> int:
        return all_len[min(n - 1, int(q * n))]

    print(f"\ndocs={n:,}  total_tokens={total_tok:,}  mean={total_tok/n:.0f}")
    print("percentiles (tokens): " + "  ".join(f"p{int(q*100)}={pct(q):,}" for q in (0.5, 0.75, 0.9, 0.95, 0.99)) + f"  max={all_len[-1]:,}")
    print(f"\n{'thresh':>8}{'docs>=':>12}{'%docs':>9}{'tokens>=':>16}{'%tokens':>10}{'%tok_capped':>13}")
    print("-" * 70)
    for t in THRESH:
        d = sum(1 for x in all_len if x >= t)
        tok_in = sum(x for x in all_len if x >= t)
        # tokens that would actually be trained at >= t position if each doc packed alone:
        capped = sum(max(0, x - t) for x in all_len)  # tokens beyond position t across corpus
        print(f"{t:>8}{d:>12,}{100*d/n:>8.2f}%{tok_in:>16,}{100*tok_in/total_tok:>9.2f}%{100*capped/total_tok:>12.2f}%")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "docs": n, "total_tokens": total_tok, "mean": total_tok / n,
            "percentiles": {str(q): pct(q) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
            "max": all_len[-1],
            "by_threshold": {str(t): {"docs_ge": sum(1 for x in all_len if x >= t),
                                       "tokens_ge": sum(x for x in all_len if x >= t)} for t in THRESH},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
