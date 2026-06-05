#!/usr/bin/env python
"""Flatten liwu/MNBVC rows into clean plain-text JSONL for the preprocess pipeline.

MNBVC stores each document's `text` field as a JSON STRING encoding a list of
paragraph objects: [{"行号", "是否重复", "是否跨文件重复", "md5", "内容"}, ...].
We parse it and join the `内容` fields into one document. `meta` (also a JSON
string) carries 文件名 etc. Output rows: {text, source, doc_id, n_paragraphs}.

Handles .jsonl.gz inputs. One MNBVC row = one (often very long) document.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def _iter_rows(path: Path):
    op = gzip.open if path.name.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def _join_paragraphs(paras, *, drop_dup_paragraphs: bool) -> str:
    if isinstance(paras, str):
        try:
            paras = json.loads(paras)
        except json.JSONDecodeError:
            return paras  # already plain text
    if not isinstance(paras, list):
        return str(paras or "")
    out = []
    for p in paras:
        if not isinstance(p, dict):
            out.append(str(p))
            continue
        if drop_dup_paragraphs and (p.get("是否重复") or p.get("是否跨文件重复")):
            continue
        c = p.get("内容")
        if c:
            out.append(c)
    return "\n".join(out)


def _extract_text(row: dict, *, drop_dup_paragraphs: bool) -> str:
    # MNBVC schemas vary by subset:
    #  - standard (book/patent/...): paragraphs in `段落` (list of {内容,...})
    #  - co_ann_report: `text` is a JSON-string of the same paragraph list
    #  - law/judgement: full plain text in `详情`
    if row.get("段落") is not None:
        return _join_paragraphs(row["段落"], drop_dup_paragraphs=drop_dup_paragraphs)
    if row.get("详情"):
        return str(row["详情"])
    if row.get("text") is not None:
        return _join_paragraphs(row["text"], drop_dup_paragraphs=drop_dup_paragraphs)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="MNBVC .jsonl.gz file(s).")
    ap.add_argument("--output", required=True, help="Output JSONL of flattened docs.")
    ap.add_argument("--source", required=True, help="source label, e.g. mnbvc_law / mnbvc_patent / mnbvc_book.")
    ap.add_argument("--min-chars", type=int, default=200, help="drop docs shorter than this.")
    ap.add_argument("--drop-dup-paragraphs", action="store_true",
                    help="skip paragraphs flagged 是否重复/是否跨文件重复 (MNBVC's own dedup flags).")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with out_path.open("w", encoding="utf-8") as out:
        for inp in args.inputs:
            for line in _iter_rows(Path(inp)):
                n_in += 1
                row = json.loads(line)
                text = _extract_text(row, drop_dup_paragraphs=args.drop_dup_paragraphs)
                if len(text) < args.min_chars:
                    continue
                doc_id = row.get("文件名") or row.get("案件id")
                if doc_id is None:
                    meta = row.get("meta")
                    if isinstance(meta, str):
                        try:
                            doc_id = json.loads(meta).get("文件名")
                        except json.JSONDecodeError:
                            doc_id = None
                    elif isinstance(meta, dict):
                        doc_id = meta.get("文件名")
                out.write(json.dumps({
                    "text": text,
                    "source": args.source,
                    "doc_id": doc_id,
                }, ensure_ascii=False) + "\n")
                n_out += 1
    print(json.dumps({"input_rows": n_in, "output_docs": n_out, "output": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
