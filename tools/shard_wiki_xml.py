#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401
from llmtrain.preprocessing.parsers import _clean_wiki_markup, _find_text, _strip_ns


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a MediaWiki XML dump into local JSONL shards.")
    parser.add_argument("--input", required=True, help="Path to a decompressed MediaWiki XML dump.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSONL shards.")
    parser.add_argument("--source-name", required=True, help="Stable source name written into ids and metadata.")
    parser.add_argument("--language", required=True, help="Language code written into metadata.")
    parser.add_argument("--license", default="CC-BY-SA")
    parser.add_argument("--shard-docs", type=int, default=50_000, help="Maximum pages per shard.")
    parser.add_argument("--limit", type=int, default=None, help="Debug only: stop after this many article pages.")
    parser.add_argument("--force", action="store_true", help="Remove existing output shards before writing.")
    args = parser.parse_args()

    summary = shard_wiki_xml(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        source_name=args.source_name,
        language=args.language,
        license_name=args.license,
        shard_docs=args.shard_docs,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def shard_wiki_xml(
    *,
    input_path: Path,
    output_dir: Path,
    source_name: str,
    language: str,
    license_name: str,
    shard_docs: int,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if shard_docs <= 0:
        raise ValueError("--shard-docs must be positive")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in output_dir.glob("*.jsonl"):
            path.unlink()
        for path in output_dir.glob("*.tmp"):
            path.unlink()

    completed_docs = _completed_docs(output_dir)
    written_docs = 0
    seen_articles = 0
    shard_index = completed_docs // shard_docs
    current_count = 0
    current_file = None
    current_tmp: Path | None = None

    progress = tqdm(
        desc=source_name,
        total=limit,
        initial=min(completed_docs, limit) if limit is not None else completed_docs,
        unit="page",
        dynamic_ncols=True,
    )
    try:
        for row in _iter_wiki_rows(input_path, source_name, language, license_name):
            seen_articles += 1
            if limit is not None and seen_articles > limit:
                break
            if seen_articles <= completed_docs:
                continue
            if current_file is None:
                current_tmp = output_dir / f"{source_name}_{shard_index:06d}.jsonl.tmp"
                current_file = current_tmp.open("w", encoding="utf-8")
                current_count = 0
            current_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            current_count += 1
            written_docs += 1
            progress.update(1)
            if current_count >= shard_docs:
                current_file.close()
                current_file = None
                assert current_tmp is not None
                current_tmp.replace(output_dir / f"{source_name}_{shard_index:06d}.jsonl")
                shard_index += 1
                current_tmp = None
    finally:
        progress.close()
        if current_file is not None:
            current_file.close()
            assert current_tmp is not None
            if current_count:
                current_tmp.replace(output_dir / f"{source_name}_{shard_index:06d}.jsonl")
            else:
                current_tmp.unlink(missing_ok=True)

    shards = sorted(output_dir.glob("*.jsonl"))
    total_docs = completed_docs + written_docs
    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "source_name": source_name,
        "language": language,
        "shard_docs": shard_docs,
        "num_shards": len(shards),
        "completed_before": completed_docs,
        "written_docs": written_docs,
        "total_docs": total_docs,
    }
    (output_dir / "shard_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _completed_docs(output_dir: Path) -> int:
    total = 0
    for path in sorted(output_dir.glob("*.jsonl")):
        with path.open("rb") as f:
            total += sum(1 for _ in f)
    return total


def _iter_wiki_rows(input_path: Path, source_name: str, language: str, license_name: str):
    count = 0
    context = ET.iterparse(input_path, events=("end",))
    for _event, elem in context:
        if _strip_ns(elem.tag) != "page":
            continue
        title = _find_text(elem, "title") or ""
        ns = _find_text(elem, "ns") or "0"
        page_id = _find_text(elem, "id") or str(count)
        text = _find_text(elem, "text") or ""
        elem.clear()
        if ns != "0" or not text.strip():
            continue
        yield {
            "id": f"{source_name}/{page_id}",
            "text": _clean_wiki_markup(text),
            "title": title,
            "page_id": page_id,
            "source": source_name,
            "language": language,
            "license": license_name,
        }
        count += 1


if __name__ == "__main__":
    main()
