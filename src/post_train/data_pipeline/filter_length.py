"""按 tokenizer 真实 token 数过滤 SFT parquet（≤ max_len），输出 *_8k.parquet。

2026-06-10 新增：F1 时这步是临时手工做的没留脚本,补成正式管线件保证可复现。
用数字切分 tokenizer(与训练一致)对 messages 整体计 token。

用法:
  python -m data_pipeline.filter_length \
    --in /data/zilu/data_unified_v2/parquet/train_sft_foundation.parquet \
    --max-len 8192
输出: 同目录 <名字>_8k.parquet + 按 source 的丢弃统计。
"""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

TOKENIZER = "/data/zilu/QseekLLM/src/llmtrain/qseek_digitsplit_base"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    args = ap.parse_args()
    out = args.out or args.inp.replace(".parquet", f"_{args.max_len//1024}k.parquet")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    pf = pq.ParquetFile(args.inp)
    writer = None
    kept = 0
    drop = Counter()
    for batch in pf.iter_batches(batch_size=2048):
        rows = batch.to_pylist()
        texts = ["\n".join(m["content"] for m in r["messages"]) for r in rows]
        lens = [len(ids) for ids in tok(texts, add_special_tokens=True).input_ids]
        keep_rows = []
        for r, L in zip(rows, lens):
            if L <= args.max_len:
                keep_rows.append(r)
            else:
                drop[r["data_source"].split(":")[0]] += 1
        if keep_rows:
            t = pa.Table.from_pylist(keep_rows)
            writer = writer or pq.ParquetWriter(out, t.schema)
            writer.write_table(t)
            kept += len(keep_rows)
    if writer:
        writer.close()
    print(f"≤{args.max_len} token: 保留 {kept:,}  丢弃 {sum(drop.values()):,} -> {out}")
    for k, v in drop.most_common():
        print(f"  丢 {k}: {v:,}")
    json.dump({"kept": kept, "dropped": dict(drop), "max_len": args.max_len},
              open(out.replace(".parquet", ".filter_manifest.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
