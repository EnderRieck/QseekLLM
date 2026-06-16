# 阶段3 数据审计与下一轮训练建议报告

> 2026-06-10，基础 SFT（计划阶段1-2，foundation_8k）epoch 2 进行中。
> 本报告回答三个问题：①第一轮配比差了什么 ②本地+网络还有什么弹药 ③下一轮怎么训。
> 关联文档：`data_audit_and_architecture.md`（弹药库架构）、`eval_tracking.md`（评测集 v2）、`traindata/`（数据卡片）。

---

## 1. 当前训练效果快照（step 4500 / 9288）

| 指标 | step500 | step4500 | 趋势 |
|---|---|---|---|
| overall acc | 14.3% | 34.4% | 线性爬升中，未平台 |
| compute_cot | 26.8% | 62.6% | 主引擎，斜率未减 |
| svamp | 8.3% | 23.7% | 跟涨，开始震荡 |
| cmath | 1.0% | 15.5% | grade1-2 撑的，grade3+ 全 0 |
| gsm8k | 2.8% | 7.8% | 慢（老师只占 0.6%） |
| gsm-plus | 1.5% | 3.5% | 与 gsm8k 比值 ~0.6，无记模板信号 |
| competition-math | 2.5% | ~2% | 噪声地板，15万竞赛数据在陪跑 |

**epoch 2 结束预估**：overall 40~44%；compute_cot 72-78%；svamp 30-36%；gsm8k 10-14%；cmath 顶在 15-20%。
**风险信号**：gsm-plus/gsm8k 比值 <0.5 或 svamp 持续回落 = 第二遍开始记模板，提前取优 ckpt。

**已确诊的失败模式**（step4000 错例分析）：
1. **组装能力缺失**：单技能（log/数列/导数各 60%+）但串不起来——18 道高考风探针几乎全 0。
2. **答案收口脱节**：gsm8k 错题中 17% "think 里出现过 gold、boxed 却答错"，不随训练下降。
3. **风格税**：见应用题就"设 x 列方程"（orca/metamath 28.7万 vs 直接算 2.6万 = 11:1），探针"反模板·直接算"全 0。

---

## 2. 第一阶段配比审计：哑铃结构

foundation_8k（118.9万）按"题的性质"归堆：

| 堆 | 量 | 占比 | eval 回报 |
|---|---|---|---|
| 单技能计算（compute_cot，覆盖到导数/矩阵/三角） | ~39万 | 33% | ✅ 在吸收 |
| 小学/GSM 应用题（orca 17.7万+metamath GSM* 10.5万+gsm8k 0.7万） | ~29万 | 24% | 🟡 svamp 涨、gsm8k 慢 |
| **高考风标准多步（中间带）** | **≈0** | **0%** | ❌ 探针全 0 |
| 竞赛长链（numina 9万+openr1 5万+其他 1.5万） | ~15.4万 | 13% | ❌ 全程 ≈0，纯陪跑 |
| 通用对话（tulutalk/infinity/coig/chinese-r1 等） | ~34.7万 | 29% | 防退化用 |

**结论：两头重、中间空。** 缺口按优先级：
1. **中间带**（2-3 技能组合的标准题）——组装能力的教材，完全缺席
2. **中文数学**（cmath grade3+ / 中文应用题）——训练集里中文数学 ≈0（chinese-r1 的数学被埋在 general 标签里）
3. **MCQ+boxed 档**（审计项⑤）——缺
4. **直接算多步**（GSM 风）——被"设方程"风压 11:1
5. 竞赛 13% 对 1.7B 是无效预算——应释放给中间带

**根因都在标签/配方层，不在弹药库本身**（与 6/9 审计结论一致）：numina adapter 丢了原始 `source` 字段并一律贴 hard，导致 cn_k12 标准题和 olympiads 奥数在库里不可区分；chinese-r1 整体贴 general，数学子集捞不出来。

---

## 3. 本地数据资产盘点（含 3 个新发现）

### 🆕 新发现一：numina-1.5 原始 cn_k12 —— 英文中间带主矿
`/data/zilu/math_sft_raw/numinamath-1.5`，按原始 `source` 字段拆：

| source | 量 | 定位 |
|---|---|---|
| **cn_k12** | **268,819（双valid 256,526）** | **中国 K12 课标题英译 = 中间带本体** |
| synthetic_math | 148,712 | MATH 风合成，中间带上沿 |
| olympiads/aops/amc/cn_contest | ~30万 | 竞赛（本轮压到 0） |
| orca_math/metamath/gsm8k | ~17万 | 已有，去重 |

cn_k12 质量：**100% 带 `\boxed{}`、99% 可验证 answer、solution 中位数 839 字符**（教科书式短解答，无 R1 绕弯，1.7B 理想教材）。题型 = word-problem 14.9万 + **MCQ 11.6万**（顺手填 MCQ 档缺口）。答案可验证 → **SFT/RL 双轨资产**。

### 🆕 新发现二：chinese-r1 的数学子集 —— 中文中间带（本以为缺货）
`/data/zilu/fastrl/data/train/chinese-deepseek-r1-distill`（110k 全量），按 `repo_name`：

| repo | 量 | 内容 |
|---|---|---|
| **EduChat-Math** | **19,729** | 中文 K12 数学（解方程/函数值域/高考风 MCQ）|
| meta-math/GSM8K_zh | 8,776 | GSM8K 中文版（中文多步直接算！）|
| gavinluo/applied_math | 7,493 | 中文应用题 |
| Haijian/Advanced-Math | 570 | 中文高数 |

**合计 ~36.6k 中文数学，R1 风格 CoT，自带 0-10 质量分**（score≥8 占 96%）。注意 R1 长 CoT 需做长度过滤/截断检查（8192 cap）。

### 🆕 新发现三：numinamath-cot（86 万，一直没动过）
`/data/zilu/fastrl/data/train/numinamath-cot`：859,494 条 problem+solution+messages，其中 **cn_k12 276,554 / synthetic_math 167,874**。与 numina-1.5 同题源但解答是 GPT-4 风格 CoT（更详细）。**与 numina-1.5 二选一或对照实验用，入库必须互相去重。**

### 其他在库/在盘资产（已知，状态确认）
- **big-math-rl-verified** 251k：RL 主池，`llama8b_solve_rate` 难度信号，cn_k12 6.4万/orca 8.3万——GRPO 课程直接用
- **calc-ape210k** 195k：中文小学应用题+可验证答案，RL 用（gadget 链弃用，只取题面+答案）
- **dapo-math-17k / deepscaler 5.7k**：GRPO 起步池
- **openthoughts3-1.2m(90k 采样)/bespoke 17k/openr1 22万**：竞赛长链，**本轮全部冷藏**
- **mgsm/zh**：250 条，可入 eval 不入训练
- 通用池（tulutalk/infinity-instruct/coig/dolly/no_robots/dynamics）：继续按 15-20% 配

---

## 4. 网络补源调研（hf-mirror 可直连，均不 gated）

| 数据集 | 规模 | 定位 | 建议 |
|---|---|---|---|
| math-eval/TAL-SCQ5K | 5K zh + 5K en | 好未来 K12 **MCQ 带分步解**，质量高 | ⭐ 下载，中文 MCQ 中间带 |
| BelleGroup/school_math_0.25M | 25万 zh | GPT 生成中小学题解，**质量混杂** | 候补：抽验后再定，只取能通过数值校验的子集 |
| Azure99/blossom-math-v4 | 1万 zh/en | 多源清洗+验证答案 | 小而干净，顺手带上 |
| meta-math/GSM8K_zh | 8.8k | 已通过 chinese-r1 间接持有 | 下原版做对照/去重基准 |
| hails/agieval-gaokao-mathqa + mathcloze | ~600 | **高考真题评测集** | ⭐ 下载入 **eval**：填"中文中间带评测"空白（现在 cmath 只测小学）|

> 高考类资源基本都是评测集（AGIEval 系），没有大规模训练集——**中文中间带训练数据的主力仍是本地 EduChat-Math + cn_k12 英译**，这是现实约束。

---

## 5. 下一轮训练方案（计划阶段3：解题课程 SFT）

### 5.1 目标
教**组装**（单技能→多步组合），让中间带立起来；竞赛缓议。

**验收线**（阶段3 SFT 结束时，v2 评测集 + 新增 gaokao 评测）：
- svamp ≥ 45% / gsm8k ≥ 20% / gsm-plus:gsm8k 比值 ≥ 0.6
- 高考风探针（18题）从全 0 → 至少 30% 有对
- agieval-gaokao-mathqa ≥ 25%（四选一基线 25%，超过即真信号）
- cmath ≥ 25%（grade3+ 开始非 0）
- compute_cot 不回落超过 5pt（地基防遗忘）

### 5.2 数据配方 v2 草案（~45万，1-2 epoch）

| 桶 | 内容 | 量 | 占比 |
|---|---|---|---|
| **中间带 EN** | cn_k12 word-problem 12万 + cn_k12 MCQ 5万 + synthetic_math 4万 | 21万 | 47% |
| **中文数学 ZH** | EduChat-Math 2万 + GSM8K_zh 0.9万 + applied_math 0.7万 + TAL-SCQ5K 0.5万 + blossom 1万 | ~5万 | 11% |
| **直接算多步** | gsm8k 0.75万(升权×2) + metamath GSM_Rephrased 3万（SV/FOBAR 降权） | ~4万 | 9% |
| **地基防遗忘** | compute_cot 重采 8万（侧重 eval 弱子源） | 8万 | 18% |
| **通用防退化** | tulutalk/infinity/coig/no_robots | 7万 | 15% |
| 竞赛 | — | 0 | 0% |
| orca 设方程风 | 降到 2万（保多样性即可） | 2万 | 4% |

设方程:直接算 从 11:1 → 约 1:2~1:3（中间带大多是直接演算风格）。

### 5.3 管线改动清单（弹药库不动，全在 adapter/配方层）
1. **numina adapter**：保留 `source`/`question_type`；难度重标 `cn_k12/synthetic_math→medium、olympiads 等→hard`（修审计项③的 numina 部分）
2. **chinese-r1 adapter**：按 `repo_name` 拆出 math 子集，单独打 `zh-math` 标签 + `score` 透传
3. **格式统一**：所有新源包成 `<think>…</think>\n#### \boxed{}`；**强制末尾复述答案**（修 17% 收口脱节——cn_k12 solution 本身以 boxed 结尾，改写成本低）
4. **去重链**：numina-1.5 ↔ numinamath-cot ↔ big-math ↔ metamath（同题源重叠严重）；训练集 ↔ {cmath, gaokao-mathqa/cloze, gsm8k/gsm-plus/svamp, math-beyond} 题面隔离
5. **eval 升级**：下载 agieval-gaokao-mathqa/mathcloze + TAL-SCQ5K test 切片，入 heldout v3（中文中间带标尺）；v2→v3 跨断点不比绝对值
6. **下载**：TAL-SCQ5K / blossom-math-v4 / GSM8K_zh / agieval-gaokao×2（hf-mirror，共 <100MB）

### 5.4 阶段4 GRPO 衔接（既定计划微调）
- 起点：阶段3 最优 ckpt；LoRA + Verl 异步（按 CLAUDE.md 既定路线）
- 课程数据：**solve_rate 筛选 pass@8∈(0,1)**——big-math 自带 llama8b_solve_rate 先粗筛，再用己方模型 pass@8 精筛（脚本可在 epoch 2 间隙用闲置 A4000 开发）
- RL 池：big-math(去重后) + cn_k12 answer + calc-ape210k(中文) + dapo/deepscaler + gsm8k-train
- 收口脱节问题在 RL 里会被 boxed-only reward 直接矫正（双保险）

### 5.5 不做的事
- ❌ epoch 3 重复 foundation_8k（边际收益耗尽，短板不在重复）
- ❌ 竞赛长链（openr1/openthoughts3/bespoke）入阶段3——pass@8≈0 无梯度、SFT 学成瞎编，留阶段4 后期
- ❌ Belle school_math 直接全量入（质量混杂，须先数值校验抽检）
- ❌ calc gadget 链直接训（与"心算"目标冲突，只取题面+答案）

---

## 6. 执行清单（按优先级）

| # | 事项 | 依赖 | 预估 |
|---|---|---|---|
| 1 | 下载 4 个小数据集（TAL/blossom/GSM8K_zh/agieval-gaokao×2） | 网络 | 0.5h |
| 2 | numina + chinese-r1 adapter 改造 & 重跑入库 | — | 2-3h |
| 3 | heldout v3（+gaokao zh 标尺）+ 隔离去重 | 1,2 | 1-2h |
| 4 | 配方 v2 cap 表 + 重切 train_sft_stage3.parquet | 2 | 1h |
| 5 | solve_rate 打分脚本（pass@8，vLLM，闲置 A4000） | — | 2-3h（可并行）|
| 6 | epoch 2 完成 → 选 ckpt（看比值+曲线）→ 启动阶段3 | 4 | 等训练 |

1-5 全部可以在 epoch 2 剩余 ~14h 内完成，训练一结束即可无缝接阶段3。
