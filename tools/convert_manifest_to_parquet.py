#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import _bootstrap  # noqa: F401
from llmtrain.data.manifest import ShardInfo, load_manifest, write_manifest
from llmtrain.utils.config import sha256_file


PARQUET_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("source", pa.string()),
        ("domain", pa.string()),
        ("language", pa.string()),
        ("metadata", pa.string()),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert manifest jsonl shards to compressed parquet shards.")
    parser.add_argument("--manifest", required=True, help="Input manifest.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Output run directory for the converted manifest.")
    parser.add_argument("--source", action="append", default=[], help="Only convert/include this source. Repeatable.")
    parser.add_argument("--domain", action="append", default=[], help="Only convert/include this domain. Repeatable.")
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "none"])
    parser.add_argument("--compression-level", type=int, default=3, help="Compression level for codecs that support it.")
    parser.add_argument("--row-group-size", type=int, default=8192)
    parser.add_argument("--batch-records", type=int, default=8192)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit-shards", type=int, default=None, help="Debug only: convert at most N matched shards.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--copy-existing-parquet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Copy parquet input shards into the output. By default existing parquet shards are referenced as-is.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    shards_dir = output_dir / "shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    selected_sources = set(args.source)
    selected_domains = set(args.domain)
    input_shards = [
        shard
        for shard in load_manifest(manifest_path)
        if (not selected_sources or shard.source in selected_sources)
        and (not selected_domains or shard.domain in selected_domains)
    ]
    if args.limit_shards is not None:
        input_shards = input_shards[: args.limit_shards]
    if not input_shards:
        raise SystemExit("No shards matched the requested filters.")

    jobs = [
        {
            "input_index": index,
            "shard": shard.model_dump(mode="json"),
            "output_path": str(_output_path(shards_dir, shard)),
            "compression": None if args.compression == "none" else args.compression,
            "compression_level": args.compression_level,
            "row_group_size": args.row_group_size,
            "batch_records": args.batch_records,
            "skip_existing": args.skip_existing,
            "copy_existing_parquet": args.copy_existing_parquet,
        }
        for index, shard in enumerate(input_shards)
    ]

    workers = max(1, min(args.max_workers, len(jobs)))
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_convert_one, job) for job in jobs}
        with tqdm(total=len(futures), desc="convert parquet", unit="shard") as progress:
            while futures:
                done, futures = wait(futures, timeout=2.0, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    results.append(result)
                    progress.update(1)
                    progress.set_postfix_str(
                        f"saved={_format_bytes(sum(max(0, r['input_bytes'] - r['output_bytes']) for r in results))}",
                        refresh=False,
                    )

    results.sort(key=lambda item: item["input_index"])
    converted = [ShardInfo.model_validate(item["shard"]) for item in results]
    paths = write_manifest(converted, output_dir)
    summary = {
        "input_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "manifest": str(paths.manifest),
        "manifest_meta": str(paths.meta),
        "num_shards": len(converted),
        "input_bytes": sum(item["input_bytes"] for item in results),
        "output_bytes": sum(item["output_bytes"] for item in results),
        "saved_bytes": sum(item["input_bytes"] for item in results) - sum(item["output_bytes"] for item in results),
        "compression": args.compression,
        "compression_level": args.compression_level,
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _output_path(shards_dir: Path, shard: ShardInfo) -> Path:
    source = _safe_path_part(shard.source)
    uri = Path(shard.uri)
    parts = uri.parts
    if "shards" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("shards")
        rel = Path(*parts[index + 1 :])
        return shards_dir / source / rel.with_suffix(".parquet")
    return shards_dir / source / f"{uri.stem}_{shard.sha256[:12]}.parquet"


def _convert_one(job: dict[str, Any]) -> dict[str, Any]:
    shard = ShardInfo.model_validate(job["shard"])
    output_path = Path(job["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shard.format == "parquet" and not job["copy_existing_parquet"]:
        return {
            "input_index": job["input_index"],
            "input_bytes": shard.bytes,
            "output_bytes": shard.bytes,
            "shard": shard.model_dump(mode="json"),
        }

    if shard.format == "parquet":
        if output_path.exists() and job["skip_existing"]:
            out = _inspect_converted_parquet(output_path, original=shard)
        else:
            tmp = output_path.with_suffix(output_path.suffix + ".tmp")
            shutil.copy2(shard.uri, tmp)
            tmp.replace(output_path)
            out = _inspect_converted_parquet(output_path, original=shard)
        return {
            "input_index": job["input_index"],
            "input_bytes": shard.bytes,
            "output_bytes": out.bytes,
            "shard": out.model_dump(mode="json"),
        }

    if output_path.exists() and job["skip_existing"]:
        out = _inspect_converted_parquet(output_path, original=shard)
        return {
            "input_index": job["input_index"],
            "input_bytes": shard.bytes,
            "output_bytes": out.bytes,
            "shard": out.model_dump(mode="json"),
        }

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    records = 0
    estimated_tokens = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            tmp,
            PARQUET_SCHEMA,
            compression=job["compression"],
            compression_level=job["compression_level"] if job["compression"] == "zstd" else None,
            use_dictionary=True,
        )
        batch: list[dict[str, Any]] = []
        with Path(shard.uri).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = str(item.get("text") or "")
                metadata = item.get("metadata") or {}
                batch.append(
                    {
                        "id": str(item.get("id") or ""),
                        "text": text,
                        "source": str(item.get("source") or shard.source),
                        "domain": str(item.get("domain") or shard.domain),
                        "language": str(item.get("language") or shard.language),
                        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    }
                )
                records += 1
                estimated_tokens += max(1, len(text.encode("utf-8")) // 4)
                if len(batch) >= job["batch_records"]:
                    _write_batch(writer, batch, row_group_size=job["row_group_size"])
                    batch.clear()
        if batch:
            _write_batch(writer, batch, row_group_size=job["row_group_size"])
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(output_path)
    out = ShardInfo(
        id=output_path.stem,
        uri=str(output_path.resolve()),
        source=shard.source,
        domain=shard.domain,
        language=shard.language,
        format="parquet",
        num_records=records,
        bytes=output_path.stat().st_size,
        estimated_tokens=estimated_tokens,
        sha256=sha256_file(output_path),
        weight=shard.weight,
        license=shard.license,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return {
        "input_index": job["input_index"],
        "input_bytes": shard.bytes,
        "output_bytes": out.bytes,
        "shard": out.model_dump(mode="json"),
    }


def _write_batch(writer: pq.ParquetWriter, rows: list[dict[str, Any]], *, row_group_size: int) -> None:
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    writer.write_table(table, row_group_size=row_group_size)


def _inspect_converted_parquet(path: Path, *, original: ShardInfo) -> ShardInfo:
    pf = pq.ParquetFile(path)
    records = pf.metadata.num_rows
    estimated_tokens = 0
    for batch in pf.iter_batches(columns=["text"], batch_size=8192):
        for text in batch.column(0).to_pylist():
            estimated_tokens += max(1, len(str(text or "").encode("utf-8")) // 4)
    return ShardInfo(
        id=path.stem,
        uri=str(path.resolve()),
        source=original.source,
        domain=original.domain,
        language=original.language,
        format="parquet",
        num_records=records,
        bytes=path.stat().st_size,
        estimated_tokens=estimated_tokens,
        sha256=sha256_file(path),
        weight=original.weight,
        license=original.license,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "unknown"


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    v = float(value)
    for unit in units:
        if abs(v) < 1024 or unit == units[-1]:
            return f"{v:.1f}{unit}" if unit != "B" else f"{v:.0f}{unit}"
        v /= 1024
    return f"{v:.1f}PB"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("conversion interrupted", file=sys.stderr)
        raise SystemExit(130)
