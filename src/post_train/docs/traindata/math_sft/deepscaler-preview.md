# 数据卡片 · agentica-org/DeepScaleR-Preview-Dataset

> 定位：**hard math，题+答案+参考解** —— 阶段3 难题 SFT 与阶段4 RL 皆可用。
> 本地：`/data/zilu/math_sft_raw/deepscaler-preview`（全量，21M）

## 总体说明

- **来源**：DeepScaleR 项目所用数据，约 **40,315** 道高难数学题，源自 **AIME / AMC / Omni-MATH / Still** 等竞赛集。
- **语言**：英文。
- **License**：见 repo（MIT 系）。
- **格式**：`problem` / `answer`(最终答案，常为 LaTeX) / `solution`(参考解，逐步)。
  —— **同时具备 solution（可做 SFT）与可验证 answer（可做 RL）**。

## 抽样

- problem: `The operation ⊗ is defined by a⊗b=a²/b. Determine [(1⊗2)⊗3] - [1⊗(2⊗3)].`
- answer: `-\frac{2}{3}`
- solution: `1. Apply ⊗ to innermost parentheses: (1⊗2)⊗3 = (1²/2)⊗3 = (1/2)... ` （逐步参考解）

## 对本项目的评估

- ✅ **阶段3 难题 SFT 候选**：有逐步 solution，可教"解竞赛题"；难度高（AIME/AMC），适合课程后期加难。
- ✅ **阶段4 RL 候选**：answer 可验证，适合 GRPO。
- ⚠️ 难度高，弱 base 早期不可用；需排在算术地基 + 中等题之后。
- ⚠️ LaTeX 答案需归一/验证器适配（`-\frac{2}{3}` 等）。
- 🔎 与 numinamath/openr1 题源（AIME/AMC/Omni）**高度重叠**，混用前跨集去重；且与 MATH-Beyond（同系 hard eval）注意训练/评测隔离。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：SFT 5,750 全留在训 + RL 18,669。

- 🔎 **18.3%(7,391 条)有非空 solution**(中位 1,079 字符,编号分步风格)——管线已将其入 SFT(5,750 为去重泄漏后),其余 82% 仅 RL,使用基本正确。
- ⚠️ 原始 928 条(2.3%)题面重复;answer 形态 numeric 61.5%/fraction 18.8%/latex 15.8%。
- 泄漏剔除 7,715 条(MATH 全量隔离误杀的一部分,基准侧改 test-only 后可部分释放)。
