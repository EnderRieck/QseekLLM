#!/bin/bash
# 弹药库 v2 全链重建(2026-06-10 审计修复版)。日志: logs/rebuild_v2.log
set -e
PY=.venv/bin/python
OUT=/data/zilu/data_unified_v2
echo "=== [1/4] build 数学弹药库 $(date) ==="
$PY -m data_pipeline.build --out $OUT
echo "=== [2/4] build_general 通用池(含隔离+跨池去重) $(date) ==="
$PY -m data_pipeline.build_general --out $OUT
echo "=== [3/4] reweight F2 配方切片 $(date) ==="
$PY -m data_pipeline.reweight_sft --out $OUT/parquet/train_sft_foundation.parquet
echo "=== [4/4] 8k 长度过滤 + RL parquet $(date) ==="
$PY -m data_pipeline.filter_length --in $OUT/parquet/train_sft_foundation.parquet --max-len 8192
$PY -m data_pipeline.to_verl_parquet --in $OUT --out $OUT/parquet
echo "=== 完成 $(date) ==="
