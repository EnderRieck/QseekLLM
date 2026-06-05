#!/usr/bin/env python
"""Concatenate wdndev/webnovel-chinese per-chapter rows into long, coherent
per-novel documents for long-context training.

Input rows are {"title", "chapter", "text"} and chapters of one novel appear as
a CONTIGUOUS run (in reading order). We stream the file, accumulate consecutive
same-title chapters, and flush one document per novel. Very long novels are split
at chapter boundaries once they exceed --max-chars so no single doc is unwieldy.

Output: JSONL with {text, title, source, n_chapters, part} ready for the standard
stream_preprocess pipeline (type: jsonl).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _flush(out, title: str, chapters: list[str], part: int) -> int:
    if not chapters:
        return part
    text = "\n\n".join(chapters)
    out.write(json.dumps({
        "text": text,
        "title": title,
        "source": "webnovel_cn",
        "n_chapters": len(chapters),
        "part": part,
    }, ensure_ascii=False) + "\n")
    return part + 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="webnovel jsonl shard(s).")
    p.add_argument("--output", required=True, help="output jsonl of concatenated novels.")
    p.add_argument("--max-chars", type=int, default=600_000,
                   help="split a novel into multiple docs once it exceeds this many chars (~200k tokens). 0 = never split.")
    p.add_argument("--min-chars", type=int, default=2000, help="drop novels shorter than this.")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_docs = n_novels = 0
    cur_title = None
    buf: list[str] = []
    buf_chars = 0
    part = 0

    with out_path.open("w", encoding="utf-8") as out:
        for inp in args.inputs:
            with open(inp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line, strict=False)
                    except json.JSONDecodeError:
                        continue
                    n_rows += 1
                    title = (r.get("title") or "").strip() or "untitled"
                    text = (r.get("text") or "").strip()
                    if not text:
                        continue
                    if title != cur_title:
                        # flush previous novel
                        if buf and buf_chars >= args.min_chars:
                            n_before = part
                            part = _flush(out, cur_title, buf, part)
                            n_docs += part - n_before
                            n_novels += 1
                        elif buf:
                            pass  # too short, drop
                        cur_title, buf, buf_chars, part = title, [], 0, 0
                    buf.append(text)
                    buf_chars += len(text) + 2
                    if args.max_chars and buf_chars >= args.max_chars:
                        part = _flush(out, cur_title, buf, part)
                        n_docs += 1
                        buf, buf_chars = [], 0
        # final flush
        if buf and buf_chars >= args.min_chars:
            n_before = part
            part = _flush(out, cur_title, buf, part)
            n_docs += part - n_before
            n_novels += 1

    print(json.dumps({
        "input_rows": n_rows,
        "novels": n_novels,
        "output_docs": n_docs,
        "output": str(out_path),
        "max_chars": args.max_chars,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
