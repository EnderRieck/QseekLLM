# 数据卡片 · MU-NLPC/Calc-gsm8k

> 定位：**GSM8K 的计算器-trace 版**（英文小学应用题 + 结构化分步计算）。
> 本地：`/data/zilu/math_sft_raw/calc-gsm8k`（全量，`data` + `original-splits`）

## 总体说明

- **来源**：MU-NLPC 在 GSM8K 之上标注**计算器调用链**（与 Calc-ape210k 同一范式，区别是英文 GSM8K 题源）。
- **规模**：train ~7k（GSM8K train）+ val/test，两 config 仅切分不同。
- **语言**：英文。
- **License**：MIT。
- **格式**：`id` / `question` / `chain`(gadget 计算链) / `result` / `result_float`。

## 抽样

题同 GSM8K（Janet's ducks）：
```
Janet sells 16 - 3 - 4 = <gadget id="calculator">16-3-4</gadget> <output>9</output> 9 duck eggs a day.
She makes 9 * 2 = $<gadget id="calculator">9*2</gadget> <output>18</output> 18 every day.
<result>18</result>
```

## 对本项目的评估

- ✅ 与 Calc-ape210k（中文）构成"中英计算器-trace"对，理念贴近我们 Compute_Cot 的逐步算。
- ⚠️ 同样问题：**`<gadget>` 计算器范式 ≠ 我们要教的"自己心算逐步推演"**。若引入会让模型学会"调用外部计算器"，与本项目目标可能冲突。
- 🔎 **建议**：和 calc-ape210k 一样，**只复用 question+result**、丢弃 gadget 链自己重新生成 CoT；或明确决定项目是否走"工具调用"路线（当前倾向不走）。量也小（~7k），可有可无。

## 审计补充（2026-06-10 复核）

- 确认冗余:与 gsm8k 完全同源(Calc-X 重格式化),管线未接,**维持弃用**。
- 唯一价值:`chain` 字段与 calc-ape210k 同格式——若实现 ape210k 的 chain→CoT 转换器,可零成本复用到英文侧生成 7.3k 带步骤样本(优先级低,gsm8k 自带 <<>> 已够)。
