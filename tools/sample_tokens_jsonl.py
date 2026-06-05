#!/usr/bin/env python
"""Quick per-source token estimate over raw JSONL (pre-preprocess).
Samples up to N docs per file, tokenizes with the project byte-BPE tokenizer,
reports mean tokens/doc, p50/p90 length, CJK ratio, and total-token extrapolation
(mean_tokens * n_docs). Handles .jsonl and .jsonl.zst.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llmtrain.tokenizer.adapter import load_tokenizer  # noqa
from llmtrain.utils.config import load_config  # noqa

random.seed(0)

def _cjk_ratio(s: str) -> float:
    if not s: return 0.0
    c = sum(1 for ch in s if "一" <= ch <= "鿿")
    return c / len(s)

def _iter_text(path: Path, field_order=("text",)):
    if path.name.endswith(".zst"):
        import zstandard as zstd, io
        with path.open("rb") as fh:
            r = zstd.ZstdDecompressor().stream_reader(fh)
            for line in io.TextIOWrapper(r, encoding="utf-8", errors="ignore"):
                line = line.strip()
                if line:
                    yield line
    else:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-config", default="configs/tokenizer/hf_byte_bpe_150k.yaml")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--text-field", default="text")
    args = ap.parse_args()

    cfg, _ = load_config(args.tokenizer_config)
    tok = load_tokenizer(cfg.tokenizer)

    grand_total = 0
    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            print(f"MISSING {inp}"); continue
        # reservoir sample lines + count total
        sample = []
        n = 0
        for line in _iter_text(path):
            n += 1
            if len(sample) < args.sample:
                sample.append(line)
            else:
                j = random.randint(0, n - 1)
                if j < args.sample:
                    sample[j] = line
        toks, chars, cjk = [], 0, 0.0
        for line in sample:
            try:
                t = json.loads(line, strict=False).get(args.text_field, "")
            except Exception:
                continue
            if not t: continue
            ids = tok.encode(t)
            toks.append(len(ids))
            chars += len(t)
            cjk += _cjk_ratio(t) * len(t)
        toks.sort()
        if not toks:
            print(f"{path.name}: no tokens"); continue
        mean = sum(toks) / len(toks)
        p50 = toks[len(toks)//2]
        p90 = toks[int(len(toks)*0.9)]
        ge8 = sum(1 for x in toks if x >= 8192) / len(toks)
        ge16 = sum(1 for x in toks if x >= 16384) / len(toks)
        cjkr = cjk / max(chars, 1)
        total = mean * n
        grand_total += total
        print(f"{path.name:30s} n={n:7d} mean={mean:8.0f} p50={p50:7d} p90={p90:7d} "
              f">=8k:{ge8:4.0%} >=16k:{ge16:4.0%} cjk={cjkr:.2f}  ~total={total/1e9:.2f}B")
    print(f"=== grand total (sampled extrapolation): {grand_total/1e9:.2f}B tokens ===")

if __name__ == "__main__":
    main()
