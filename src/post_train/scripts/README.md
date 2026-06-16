# scripts —— 工具脚本索引

post_train 的一次性/可复跑工具脚本统一放这里，按用途分子目录。所有脚本默认**只读**，
不修改源数据。运行无需 uv 环境（纯标准库），直接 `python scripts/<...>.py`。

## 目录

| 路径 | 用途 | 备注 |
|------|------|------|
| `qc/quality_report.py` | Compute_Cot 数据集**质量体检**：格式合规、答案一致性、verified 占比、source/难度分布、split 内去重率、train↔test 泄漏率 | 流式扫全部 split，几十秒 |
| `qc/inspect_samples.py` | 按 source / difficulty **抽取完整样本**，人工核查 `<think>` 推演质量 | 合并自早期 `_qc_sample*.py` |

## 常用命令

```bash
# 全量质量体检 (默认读 /data/zilu/fastrl/Compute_Cot/data)
python scripts/qc/quality_report.py
python scripts/qc/quality_report.py --data /path/to/other/data   # 指定数据目录

# 抽样核查某个 source 的推演
python scripts/qc/inspect_samples.py --source arithmetic.fraction_division -n 5
python scripts/qc/inspect_samples.py --match long_division --difficulty hard -n 2
```

## 已知发现 (2026-06-09 首轮体检)

> 详细结论见 `docs/` 下的数据质量报告。关键点：
> - **格式/答案校验满分**（550k 全过 verified、格式、boxed==answer）。
> - **重复严重**：train 去重后仅 ~26.5 万唯一题（标称 55 万），s3 重复率 73.5%。
> - **泄漏超标**：id_test 34.4% / val 29.4% 题面与 train 完全重合（违反 CLAUDE.md 约束）。
> - **推演系统 bug**（验证器盲区）：
>   - `arithmetic.decimal_division_by_decimal`：100% 样本含错误整数除法等式（如 `168÷140=12`）+ 伪"补小数点"步骤。
>   - `arithmetic.fraction_division`：负号样本符号推演自相矛盾。
