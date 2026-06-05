#!/usr/bin/env python
"""Normalize yuyijiong/LongData-Corpus LongData_en sources (books + long web)
into clean plain-text JSONL {text, source, doc_id}, mirroring extract_longdata_zh.

  RedPajamaBook-100k/*.jsonl.zst       -> source=redpajama_book
  RedPajamaCommonCrawl-32k-*.json      -> source=redpajama_cc  (line-delimited JSONL)
Both carry the document in a `text` field.
"""
from __future__ import annotations
import argparse, glob, json, io
from pathlib import Path


def _iter_rows(path: Path):
    name = path.name
    if name.endswith(".zst"):
        import zstandard as zstd
        with path.open("rb") as fh:
            r = zstd.ZstdDecompressor().stream_reader(fh)
            for line in io.TextIOWrapper(r, encoding="utf-8", errors="ignore"):
                line = line.strip()
                if line:
                    yield line
    else:  # .json / .jsonl (line-delimited)
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-dir", required=True, help="LongData_en directory.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-chars", type=int, default=4000)
    args = ap.parse_args()

    en = Path(args.en_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = {
        "redpajama_book": sorted(glob.glob(str(en / "RedPajamaBook-100k" / "*.jsonl.zst"))),
        "redpajama_cc":   sorted(glob.glob(str(en / "RedPajamaCommonCrawl-32k*.json"))),
    }
    for source, files in groups.items():
        if not files:
            print(json.dumps({"source": source, "status": "NO_FILES"})); continue
        out_path = out_dir / f"{source}.jsonl"
        n_in = n_out = 0
        with out_path.open("w", encoding="utf-8") as out:
            for fp in files:
                for i, line in enumerate(_iter_rows(Path(fp))):
                    n_in += 1
                    try:
                        row = json.loads(line, strict=False)
                    except json.JSONDecodeError:
                        continue
                    text = row.get("text") or row.get("content") or ""
                    if not isinstance(text, str) or len(text) < args.min_chars:
                        continue
                    out.write(json.dumps({"text": text, "source": source,
                                          "doc_id": f"{source}_{n_out}"}, ensure_ascii=False) + "\n")
                    n_out += 1
        print(json.dumps({"source": source, "input_rows": n_in, "output_docs": n_out,
                          "output": str(out_path)}, ensure_ascii=False), flush=True)
    print("ALL_LONGDATA_EN_DONE", flush=True)


if __name__ == "__main__":
    main()
