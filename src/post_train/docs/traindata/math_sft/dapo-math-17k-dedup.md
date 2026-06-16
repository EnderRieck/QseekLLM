# 数据卡片 · YouJiacheng/DAPO-Math-17k-dedup

> 定位：**RL 起步 prompt 集**（题 + 可验证答案，DAPO/GRPO 直接可用）。
> 本地：`/data/zilu/math_sft_raw/dapo-math-17k-dedup`（全量去重版，3M）

## 总体说明

- **来源**：DAPO 系统使用的数学 RL 数据，原 `BytedTsinghua-SIA/DAPO-Math-17k` 的**去重版**。
- **规模**：**17,398** 条（去重后）。
- **语言**：英文。
- **License**：见 repo。
- **格式（RL 专用）**：`prompt`(chat，含指令"Solve step by step... 末行 `Answer: $Answer`") / `reward_model`({`ground_truth`, `style`})。**无 solution——只给可验证答案，专为 RL 设计。**

## 抽样

- prompt: `[{role:user, content:"Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer ..."}]`
- reward_model: `{"ground_truth": "162", "style": "rule-lighteval/MATH_v2"}`

## 对本项目的评估

- 🎯 **阶段4 GRPO/DAPO 的标准起步集**：prompt 已带"逐步 + 末行 Answer:"的指令模板，`ground_truth` + lighteval 规则 style 直接对接奖励函数，**开箱即用**。
- ⚠️ **只适合 RL，不适合 SFT**（无解答过程，教不了"怎么解"）。
- 🔎 量小(17k)、干净去重，适合 GRPO smoke / 第一轮 RL；难度更高/量更大时换 Big-Math-RL-Verified 主池。答案验证风格是 MATH_v2 规则，与我们的判分器要对齐。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：RL-only 16,919 入池。适配器已正确剥掉 lighteval 模板前缀(实测验证)。

- ✅ ground_truth 100% 纯整数,规则判分零成本,GRPO 起步池定位没问题。
- 🔎 **18.9%(3,282 条)题面是中文竞赛风格题**——目前管线当英文池用,这是少数现成中文数学题源之一,可单独捞出参与中文 RL/评测。
- ⚠️ 与 big-math/deepscaler 源头(AMC/AIME 系)重叠,全局去重已覆盖(dapo 在 SOURCE_ORDER 最后,dup 479)。
