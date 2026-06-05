#!/usr/bin/env python
"""Filter chinese_cosmopedia shards to a high-quality subset by metadata.score.

Reads the preprocessed cosmopedia shards, keeps only docs with score >= THRESH,
writes new shards (1:1 with input, empty ones skipped) under OUT/shards/ and a
manifest.jsonl in the same schema the reader expects (source stays
'chinese_cosmopedia' so the existing source_filter still matches).
"""
import json, glob, os, hashlib, datetime, sys
from concurrent.futures import ProcessPoolExecutor

SRC = "/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/stream_preprocess_chinese_cosmopedia"
OUT = "/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/cosmopedia_hq_0885"
THRESH = 0.885

os.makedirs(f"{OUT}/shards", exist_ok=True)


def process(args):
    idx, path = args
    out_path = f"{OUT}/shards/clean_chinese_cosmopedia_hq_{idx:06d}.jsonl"
    n = 0
    nbytes = 0
    h = hashlib.sha256()
    with open(path) as fin, open(out_path, "wb") as fout:
        for line in fin:
            try:
                d = json.loads(line)
            except Exception:
                continue
            sc = d.get("metadata", {}).get("score")
            if sc is None or sc < THRESH:
                continue
            raw = (json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8")
            fout.write(raw)
            h.update(raw)
            nbytes += len(raw)
            n += 1
    if n == 0:
        os.remove(out_path)
        return None
    return {
        "bytes": nbytes,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "domain": "zh",
        "estimated_tokens": nbytes // 4,
        "format": "jsonl",
        "id": f"clean_chinese_cosmopedia_hq_{idx:06d}",
        "language": "zh",
        "license": "apache-2.0",
        "num_records": n,
        "record_end": None,
        "record_start": 0,
        "sha256": h.hexdigest(),
        "source": "chinese_cosmopedia",
        "uri": out_path,
        "weight": 1.0,
    }


def main():
    shards = sorted(glob.glob(f"{SRC}/shards/part_*/clean_*.jsonl"))
    print(f"filtering {len(shards)} shards at score >= {THRESH}", file=sys.stderr)
    records = []
    with ProcessPoolExecutor(max_workers=32) as ex:
        for i, rec in enumerate(ex.map(process, list(enumerate(shards)))):
            if rec is not None:
                records.append(rec)
            if (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{len(shards)}", file=sys.stderr)
    with open(f"{OUT}/manifest.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    tot_docs = sum(r["num_records"] for r in records)
    tot_est = sum(r["estimated_tokens"] for r in records)
    print(f"\nwrote {len(records)} shards, {tot_docs:,} docs, "
          f"est={tot_est/1e9:.2f}B real~={tot_est*0.772/1e9:.2f}B")
    print(f"manifest: {OUT}/manifest.jsonl")


if __name__ == "__main__":
    main()
