# 数据卡片 · brendel-group/MATH-Beyond

> 定位：**hard eval（不训练）** —— 专测"超出常见 MATH/AIME 饱和点"的难题。
> 本地：`/data/zilu/math_sft_raw/math-beyond`（全量，2.1M）

## 总体说明

- **来源**：从 DAPO-Math-17K 与 DeepScaleR 子集抽取，**刻意挑选当前小模型普遍解不出**的难题，避免 MATH/AIME 饱和。
- **规模**：仅 **181** 题（精选难题评测）。
- **语言**：英文。
- **格式**：`problem` / `answer` / `data_source` / `topic` / `difficulty` + **大量 per-model 列**：各模型的 `*_unsolved`(是否解不出) 与 `*_pass@{64..1024}`（不同采样预算下的通过情况），还含 gpt5-mini/o4-mini 的参考 response。

## 抽样

- problem: `Let a_1,...,a_100 be non-negative integers such that ... (复杂数论/组合约束)`
- answer: `40940`，topic=Geometry，difficulty=2.0
- 附带 ~20 个模型的 unsolved 标记与 pass@k 曲线（如 qwen2.5-7b-instruct 全 pass@1024=0，qwen3-8b pass@64=1）

## 对本项目的评估

- 🎯 **纯 hard eval，绝不进训练**：用来测我们模型在"难题前沿"的能力，且自带众多基线模型的 pass@k 对照，**可直接横向比较**我们的模型在哪些题上还解不出。
- ⚠️ 难度极高（很多 7B 模型 pass@1024 仍为 0），仅作天花板诊断，不期待短期攻克。
- 🔎 量极小(181)，定位是"困难 OOD 评测点"，与 gsm8k/cmath（基础评测）形成难度两端。**注意它的题来自 DAPO/DeepScaleR——若那两个进了训练，必须从训练集剔除这 181 题防泄漏。**

## 审计补充（2026-06-10 复核）

- 实为 181 行 × 360+ 列(收集"开源 RL 模型 pass@1024 仍不解"的极难题,带 21 模型 unsolved 标记与 pass@k 网格)。
- 确认:对 1.7B 模型完全超纲,**维持纯隔离源/极限探针定位,不进任何训练或常规评测**。
