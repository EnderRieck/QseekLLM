"""把统一 jsonl 池转成 Verl 友好的 parquet。

- SFT parquet：列 `messages` = prompt + {role:assistant, content:gold_response}（Verl multiturn_sft_dataset 读 messages，
  apply_chat_template 后只对 assistant 段算 loss）。合并数学 SFT + 通用 SFT，带 ability/data_source 便于配比。
- RL parquet：列 `prompt`(chat) + `reward_model`{ground_truth,style} + data_source + extra_info（Verl RLHFDataset 直读）。

用法:
  python -m data_pipeline.to_verl_parquet --in /data/zilu/data_unified --out /data/zilu/data_unified/parquet
"""
from __future__ import annotations
import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 50000


def _write_parquet(rows_iter, path, schema_probe_n=2000):
    """流式写 parquet，pyarrow 从 Python 对象推断 list<struct> 等嵌套类型。"""
    writer = None
    buf = []
    n = 0
    for row in rows_iter:
        buf.append(row)
        if len(buf) >= CHUNK:
            tbl = pa.Table.from_pylist(buf)
            if writer is None:
                writer = pq.ParquetWriter(path, tbl.schema)
            writer.write_table(tbl)
            n += len(buf)
            buf = []
    if buf:
        tbl = pa.Table.from_pylist(buf)
        if writer is None:
            writer = pq.ParquetWriter(path, tbl.schema)
        writer.write_table(tbl)
        n += len(buf)
    if writer:
        writer.close()
    return n


def sft_rows(jsonl_paths):
    for p in jsonl_paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                if not o.get("gold_response"):
                    continue
                messages = list(o["prompt"]) + [{"role": "assistant", "content": o["gold_response"]}]
                yield {
                    "messages": messages,
                    "data_source": o["data_source"],
                    "ability": o["ability"],
                    "extra_info": json.dumps(o.get("extra_info", {}), ensure_ascii=False),
                }


def rl_rows(jsonl_path):
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if not o.get("reward_model", {}).get("ground_truth"):
                continue
            yield {
                "prompt": o["prompt"],
                "reward_model": o["reward_model"],
                "data_source": o["data_source"],
                "ability": o["ability"],
                "extra_info": json.dumps(o.get("extra_info", {}), ensure_ascii=False),
            }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", default="/data/zilu/data_unified")
    ap.add_argument("--out", default="/data/zilu/data_unified/parquet")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = args.indir

    print("写 SFT parquet（数学 + 通用合并）...", flush=True)
    n_sft = _write_parquet(sft_rows([f"{d}/train_sft.jsonl", f"{d}/train_general_sft.jsonl"]),
                           f"{args.out}/train_sft.parquet")
    print(f"  SFT parquet: {n_sft:,} 行", flush=True)

    print("写 RL parquet...", flush=True)
    n_rl = _write_parquet(rl_rows(f"{d}/train_rl.jsonl"), f"{args.out}/train_rl.parquet")
    print(f"  RL parquet: {n_rl:,} 行", flush=True)

    print(f"输出 -> {args.out}/{{train_sft,train_rl}}.parquet", flush=True)


if __name__ == "__main__":
    main()
