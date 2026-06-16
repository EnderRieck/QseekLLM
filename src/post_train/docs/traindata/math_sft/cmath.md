# 数据卡片 · weitianwen/cmath

> 定位：**中文小学数学评测**（仅 eval，**不进训练**）。
> 本地：`/data/zilu/math_sft_raw/cmath`（全量，dev + test 两 jsonl）

## 总体说明

- **来源**：CMATH（Chinese Elementary School Math Test），中文小学 1–6 年级数学题评测集。
- **规模**：dev **600** + test **~1.7k**（按年级/位数分层）。
- **语言**：中文。
- **License**：见 repo（评测用途）。
- **格式**：`grade`(年级 1–6) / `question`(中文题) / `golden`(标准答案) / `reasoning_step`(所需步数) / `num_digits`(涉及位数)。

## 抽样

- grade=1, num_digits=2, reasoning_step=1
- Q: `芳芳买了一本书有99页，看了90页，她还剩多少页没有看？`
- golden: `9`

## 对本项目的评估

- 🎯 **纯评测用**：是我们"中文小学数学"能力的标准标尺，应纳入过程评测/最终汇报。带 `grade`/`reasoning_step`/`num_digits` 三个分层维度，**非常适合按难度细分看模型在哪个年级/位数/步数上掉链子**——正好喂我们 eval 驱动课程的诊断。
- ⚠️ **绝不进训练**（评测集）。
- ✅ 答案是纯数值 `golden`，自动判分简单。
- 🔎 中文评测侧与 gsm8k（英文）配对，构成中英小学数学双标尺。

## 审计补充（2026-06-10 全量复核）

- ✅ 纯评测定位正确：**没有 train split**(dev 600 + test 1,098,官方训练段未发布),不能也不应进训练。
- 🔎 `grade`(1-6)/`reasoning_step`/`num_digits` 三个分层标注目前 heldout 只用了 grade —— 按"步数/位数"切曲线对诊断算术短板(如多位数乘法)更对症,建议 eval 侧补维度。
- 实测训练池对 cmath 题面泄漏:仅 1 条(经 chinese-r1 进通用池),已在审计报告 §一记录,需随 build_general 隔离修复一并清除。
