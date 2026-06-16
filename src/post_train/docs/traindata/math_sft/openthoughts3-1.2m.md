# 数据卡片 · open-thoughts/OpenThoughts3-1.2M

> 定位：**数学/代码/科学的长推理蒸馏大池子**；难度高、风格长。
> 本地：`/data/zilu/math_sft_raw/openthoughts3-1.2m`（**采样 9/120 分片**，跨域取样；全量 1.2M）

## 总体说明

- **来源**：OpenThoughts 团队用强推理模型（QwQ/R1 系）蒸馏的 reasoning 数据，全量 **1.2M** = 850k math + 250k code + 100k science。
- **本地采样**：取了 0,1,40,50,60,70,80,90,119 共 9 个分片（90k 条），覆盖三域（**分片按域分组**：前段 code、中段 math、尾段 science）。要全量改 allow_patterns 取全部 120 片。
- **语言**：英文。
- **License**：Apache-2.0。
- **格式**：`difficulty` / `source` / `domain`(code/math/science) / `conversations`([{from,value}] 多轮，含长推理)。

## 子域（domain，3 类）抽样

| domain | 采样片内 | 典型 source |
|--------|---------:|------|
| math | 60,000 | ai2-adapt-dev/openmath-2-math |
| code | 20,000 | nvidia/OpenCodeReasoning, stackexchange_codegolf |
| science | 10,000 | stackexchange-physics |

- **math**：`In a row of twenty consecutive positive integers, the gcd of the first and last terms is 13. Find the sum...`
- **code**：`I am in desperate need of some ear defenders... I have headphones, a microphone...`（codegolf 题，difficulty=7）
- **science**：`The conditions for steady current ∂ρ/∂t=0 and ∂J/∂t=0... Combining...`（物理推导）

## 对本项目的评估

- ⚠️ **难度/风格偏后期**：长 CoT 推理，含 code 与 science，**与本阶段"基本算术+数学方法"目标偏离较大**。
- ⚠️ **混入了大量 code(250k) 与 science(100k)**——如果只想要数学，需用 `domain=="math"` 严格过滤（math 占 ~71%）。
- ✅ math 子域（85w）是**高质量长推理题源**，留作阶段3/4 难题/RL 的备选；可按 `difficulty` 调难度。
- 🔎 本阶段**不混入**。若后续要用，只取 `domain=math`，并注意它与 numinamath/openr1 的题源可能重叠（都含 openmath/olympiad 系），需跨数据集去重。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：入池 38,436(按"末轮含 boxed"过滤后)，cap 5万，但 **8k 长度过滤后实际在训仅 1,071 条**——该源在 foundation 阶段事实上缺席。

- 🔴 **math 子集 ~70% 回答截断**：`</think>` 闭合仅 31.9%、含 boxed 仅 39.6%、"闭合且有 boxed" 仅 29.1%(抽 1.5万)。适配器靠 boxed 过滤恰好挡住了大部分截断样本(这是运气不是设计——"think 未闭合但有 boxed"的残样本仍可能进池，建议显式加 `</think>` 闭合检查)。
- 🔴 **超长**：math 对话总长中位 48,282 字符(约 16k token)、98.8% 超 24,000 字符 → 8k/16k 上下文都难用，**与"教基础算术"目标错配**。
- math 子集 difficulty 字段全 None(无难度信号)；source 全部 `openmath-2-math`；全英文。
- **建议**：阶段3 前不再投入；若用，过滤条件改为"think 闭合 ∧ boxed ∧ 长度分桶"，且只小比例做长推理风格种子。
