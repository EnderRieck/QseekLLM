# 数据卡片 · bespokelabs/Bespoke-Stratos-17k

> 定位：**小而精的长思维链蒸馏**（DeepSeek-R1 风格 reflection 推理）；难度高。
> 本地：`/data/zilu/math_sft_raw/bespoke-stratos-17k`（全量）

## 总体说明

- **来源**：Bespoke Labs 用 **DeepSeek-R1** 蒸馏（Berkeley Sky-T1 配方），覆盖数学/代码/谜题的长推理。
- **规模**：**16,710** 条。
- **语言**：英文。
- **License**：见 repo（Apache-2.0 系）。
- **格式**：`system`(长思考型系统提示) / `conversations`（[{from, value}] 多轮，user 问 + assistant 长推理含 `\boxed{}`）。**无显式 category 列**（混合数学/代码/逻辑）。

## 抽样

- system: `Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions... analysis, summarizing, exploration, reassessment, reflection, backtracing...`
- conversations[0].user: `Return your final response within \boxed{}. The operation ⊗ is defined by a⊗b=a²/b. Determine [(1⊗2)⊗3]-[1⊗(2⊗3)]. (A) -2/3 ...`
- assistant: 长篇 reflection 式推理（反复检查、回溯）+ 最终 `\boxed{}`。

## 对本项目的评估

- ⚠️ **难度与风格都偏后期**：长 reflection 推理（"wait, let me reconsider..."），**不适合阶段1/2 教基础**——会让弱模型学到冗长跑题。
- ✅ 体量小但质量高，适合**阶段3/4 的长 CoT 风格引入**或做 reflection 风格的少量配比实验；`system` 提示明确定义了"先长思考再给精确解"的范式。
- ✅ `\boxed{}` 答案格式与我们一致，便于判分。
- 🔎 本阶段**不混入**；留作后续"长推理/自检风格"的种子。与 openr1（题量大）相比，Bespoke 更小更聚焦，可做风格对照。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：入池 3,494(SFT-only)，8k 过滤后在训 3,057。入池量远小于 16,710 的原因：评测隔离剔除 7,644(MATH 题面重合)+ 无 boxed 的丢弃。

- 🟠 **构成混杂**：boxed 数学题仅 62.8%；**32.3% 是 Python 代码生成题、4.9% 是物理/化学/常识 QA**。适配器靠"answer 段含 boxed"间接过滤掉了大部分代码题，但未显式按前缀分流——建议显式剔除 codegen 前缀(`Generate an executable Python function…`)。
- markup 是 `<|begin_of_thought|>/<|begin_of_solution|>`(非 `<think>`)，wrap_think_boxed 的 `<|…|>` 清洗能剥掉标记但会把 thought/solution 两段揉成一段。
- 长度：中位 10,141 字符、22.5% 超 24,000；markup 100% 闭合(无截断，比 OT3 干净)。
- 定位维持：量小、R1 风格种子，阶段2-3 少量混入即可。
