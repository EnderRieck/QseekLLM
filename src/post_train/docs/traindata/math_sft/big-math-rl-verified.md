# 数据卡片 · SynthLabsAI/Big-Math-RL-Verified

> 定位：**阶段4 RL 主数据池**（25万 可验证答案数学题，明确为 RL 设计，自带难度信号）。
> 本地：`/data/zilu/math_sft_raw/big-math-rl-verified`（全量，32M；gated，已授权下载）

## 总体说明

- **来源**：SynthLabs，聚合多源数学题并清洗为**可验证答案、closed-form、开放式问题**，**明确为 RL（PPO/GRPO）设计**。
- **规模**：**251,122** 条。
- **语言**：英文。
- **License**：见 repo（gated:auto，需网页接受条款）。
- **格式**：`problem` / `answer`(可验证,closed-form) / `source` / `domain` / **`llama8b_solve_rate`(Llama-8B 通过率，0~1，天然难度信号)**。

## 子类分布（看全类别）

**source（题源，跨难度）**：orca_math 83,215 / cn_k12 63,609 / big_math 47,010 / olympiads 33,485 / math 8,963 / aops_forum 5,740 / harp 2,996 / omnimath 2,478 —— **从小学(orca)到奥数(olympiads)全谱**。
**domain**：Math Word Problems 84,963 / Algebra-Inequalities 14,364 / Plane Geometry 13,921 / Algebra-Other 12,732 / Sequences 7,421 / Number Theory 5,577 / Trigonometry 5,246 / Probability 5,059 …

## 抽样

- problem: `Given p: |4x-3|≤1 and q: x²-(2a+1)x+a²+a≤0, find the range of a if p is a necessary [condition]...`
- answer: `[0, \frac{1}{2}]` · source: cn_k12 · domain: Algebra-Inequalities · **llama8b_solve_rate: 0.125**

## 对本项目的评估

- 🎯 **阶段4 GRPO 的主力 RL 池**：量大(25万)、可验证答案、closed-form，正合 RL；比 DAPO(17k 起步)规模大得多，适合 RL 主训练。
- ✅ **`llama8b_solve_rate` 是宝**：直接当难度过滤/课程信号——RL 时挑 solve_rate 适中的题（太易无梯度、太难无信号），或按 solve_rate 做难度课程。
- ✅ `source` 跨小学→奥数全谱，可按难度/题源调配比。
- ⚠️ 仅 RL（答案集，无 solution，不教"怎么解"）。
- 🔎 题源含 orca_math/olympiads/math/omnimath —— **与 orca-math-200k / numinamath / openr1 / deepscaler 高度重叠**，混用及评测前务必跨集去重 + 与 gsm8k/cmath/math-beyond 隔离防泄漏。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：RL-only 入池 71,746(去重/泄漏剔除后；原始 251,122)。

- 🔎 **`llama8b_solve_rate` 是现成难度标签但未用于课程**：双峰分布(0-5% 占 20%、90-100% 占 20.8%)，正好支撑阶段3"易多难少→难升"的课程调度与阶段4 RL 难度采样,无需自评 solve_rate。
- ⚠️ 去重损耗大的原因：source=orca_math 占 33.1%(83,215)与 orca 直接重叠、cn_k12 25.3% 与 numina 重叠 → first-wins 被先入源吃掉 169,994 条。
- cn_k12 已全部英译(CJK 仅 0.03%)，**不能补中文**。answer 28% 是 LaTeX,验证器需 math_verify(已是)。
