#!/usr/bin/env python
"""Normalize yuyijiong/LongData-Corpus `LongData_zh/*` files into clean plain-text
JSONL for the stream_preprocess pipeline.

The corpus mixes formats and all carry the long document in a `text` field:
  - *.csv               (维基 single `text` col; 学习强国/悟道 MNBVC-style with a `text` col)
  - *.json              actually line-delimited JSONL of {title?, text}
  - *.jsonl.zst         zstd-compressed JSONL of {text, ...}

Each input file -> one output JSONL datasets/LongData-Corpus/extracted/<source>.jsonl
with rows {text, source, doc_id}. doc_id falls back to title / 文件名 / running index.
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import sys
from pathlib import Path

csvmod.field_size_limit(sys.maxsize)

# filename (under LongData_zh/) -> source label
SOURCES = {
    "悟道200G数据-32000字以上-16000条.csv": "wudao",
    "万卷-专利-16k-16715条.json": "wanjuan_patent",
    "万卷-新闻-16k-2490条.json": "wanjuan_news",
    "CCI中文互联网语料-大于16k字-30000条.jsonl.zst": "cci",
    "SkyPile_大于16k字_9720条.json": "skypile",
    "学习强国1.6w字以上459条.csv": "xuexi",
    "中文维基百科-16000字以上-708条.csv": "wiki_zh",
    "中外名著71本.json": "mingzhu",
    "金庸小说15本.json": "jinyong",
    "政府工作报告1.6w字以上170条.csv": "govreport",
}


def _text_of(row: dict) -> str:
    t = row.get("text") or row.get("content")
    if isinstance(t, str) and t.strip():
        return t
    # MNBVC-style fallback: join 段落 内容
    paras = row.get("段落")
    if isinstance(paras, str):
        try:
            paras = json.loads(paras)
        except json.JSONDecodeError:
            return paras
    if isinstance(paras, list):
        return "\n".join(p.get("内容", "") for p in paras if isinstance(p, dict))
    return str(t or "")


def _iter_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csvmod.DictReader(f):
            yield row


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line, strict=False)
                except json.JSONDecodeError:
                    continue


def _iter_zst(path: Path):
    import zstandard as zstd
    import io
    with path.open("rb") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line, strict=False)
                except json.JSONDecodeError:
                    continue


def _iter_rows(path: Path):
    n = path.name
    if n.endswith(".csv"):
        yield from _iter_csv(path)
    elif n.endswith(".jsonl.zst") or n.endswith(".zst"):
        yield from _iter_zst(path)
    else:  # .json / .jsonl (line-delimited)
        yield from _iter_jsonl(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh-dir", required=True, help="LongData_zh directory.")
    ap.add_argument("--out-dir", required=True, help="output dir for per-source jsonl.")
    ap.add_argument("--min-chars", type=int, default=4000, help="drop docs shorter than this (these are long-doc sources).")
    args = ap.parse_args()

    zh = Path(args.zh_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for fname, source in SOURCES.items():
        path = zh / fname
        if not path.exists():
            summary.append({"source": source, "status": "MISSING", "file": fname})
            continue
        out_path = out_dir / f"{source}.jsonl"
        n_in = n_out = 0
        with out_path.open("w", encoding="utf-8") as out:
            for i, row in enumerate(_iter_rows(path)):
                n_in += 1
                if not isinstance(row, dict):
                    continue
                text = _text_of(row)
                if len(text) < args.min_chars:
                    continue
                doc_id = row.get("title") or row.get("文件名") or f"{source}_{i}"
                out.write(json.dumps({"text": text, "source": source, "doc_id": doc_id}, ensure_ascii=False) + "\n")
                n_out += 1
        summary.append({"source": source, "input_rows": n_in, "output_docs": n_out, "output": str(out_path)})
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    print("ALL_LONGDATA_ZH_DONE", flush=True)


if __name__ == "__main__":
    main()
