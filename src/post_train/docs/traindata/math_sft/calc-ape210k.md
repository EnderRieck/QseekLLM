# 数据卡片 · MU-NLPC/Calc-ape210k

> 定位：**中文小学应用题 + 计算器 trace**（结构化分步计算，理念最接近我们 Compute_Cot）。
> 本地：`/data/zilu/fastrl/data/train/calc-ape210k`（已有，含 `default` 与 `original-splits` 两 config，**可直接复用**）

## 总体说明

- **来源**：在中文应用题数据集 **Ape210K** 之上，由 MU-NLPC 标注**计算器调用链**（gadget/output/result 格式），用于训练模型"调用计算器逐步算"的能力。
- **规模**：train **195,179**；两个 config 仅 test/val 切分不同——`original-splits`(test/val 各 4867) vs `default`(test 1785 / val 1783)，train 相同。
- **语言**：**中英双语**（`question` 英文译版 + `question_chinese` 中文原题）。
- **License**：MIT。
- **格式**：`id` / `question`(英) / `question_chinese`(中) / `chain`(计算器 trace) / `result` / `result_float` / `equation`(列式)。**无题型子类**（全为小学应用题）。

## `chain` 计算器格式抽样

题：`王艳家买了一台洗衣机和一台电冰箱，一共花了6000元，电冰箱的价钱是洗衣机的(3/5)，求洗衣机的价钱．`
```
<gadget id="calculator">3 / 5</gadget> <output>3/5 = around 0.6</output>
<gadget id="calculator">1 + (3/5)</gadget> <output>8/5 = around 1.6</output>
<gadget id="calculator">6_000 / (8/5)</gadget> <output>3_750</output>
<result>3_750</result>
```
- `equation`: `x=6000/(1+(3/5))` ，`result_float`: 3750.0。

## 对本项目的评估

- ✅ **中文小学应用题主力**，恰好补 orca-math（英文）的中文侧；中英对照还可做翻译/对齐。
- ✅ 分步计算理念与我们 Compute_Cot 高度一致（强调"每步算对"），`equation` 字段给了可验证的列式，便于**重新校验 + 转成我们 `<think>` 格式**。
- ⚠️ **trace 是 `<gadget>` 计算器标记格式，不是自然语言逐步推演**；若要并入我们的数据，需把 gadget 链**改写成自然语言分步**（或决定是否引入"调用计算器"范式——但我们的目标是教模型**自己心算**，gadget 范式可能与之冲突，需谨慎）。
- ⚠️ 答案用 `3_750` 这种下划线千分位，需归一。
- 🔎 **建议**：作为**中文基础应用题**的来源很好；但 chain 格式要么弃用（只取 question+equation+result 重新生成 CoT），要么明确决定是否教"计算器调用"。本阶段倾向**只复用题面+答案，自己生成符号推演**。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：仅当 RL 题面+答案用(193,419 条 RL 池)，SFT 0 —— **被严重低估的最大中文数学资产**。

- ✅ train 195,179 全中文小学应用题(中位 43 字)；另有 val/test 各 4,867(原始 split，未用)。
- 🔎 **`equation` 字段 100% 可 eval**(抽 3,000：`%`/`**` 归一后全部可算)，eval 结果与 `result_float` 匹配 98.6%；**`chain` 字段 95% 自带 `<gadget>49+1</gadget><output>50</output>` 形式的分步计算链**——模板化改写成中文 `<think>` worked CoT 的可行性高，不必从头规划步骤。
- 多步算式占主体(≥2 步 79.7%)；~5% equation 是退化 `x=常数`(丢弃或靠 chain)；1.4% eval 与 result 不匹配(取整/单位题，过滤)。
- **建议(高优)**：阶段2 用 chain→中文 worked CoT 转换器生成 15-18万条中文数学 SFT——中文数学缺口(architecture 文档 P0)目前唯一成规模的解。英文侧 calc-gsm8k 的 chain 同格式可复用转换器。
