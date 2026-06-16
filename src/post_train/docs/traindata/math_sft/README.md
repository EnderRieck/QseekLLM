# 数学数据集总览（阶段2/3 数学课程 + 评测）

> 10 个数据集的信息卡片索引。下载脚本 `scripts/download_math_sft.py`（镜像直连）；
> 原始数据在 `/data/zilu/math_sft_raw/`，另有 3 个**已在 `/data/zilu/fastrl/data/train/`**（可直接复用）。
> 本阶段目标 = **基本算术 + 数学方法**（不做多步组合/竞赛），故难题集仅留作后续阶段。

## 一览（按"适用阶段"分组）

| 数据集 | 语言 | 规模(本地) | 难度 | 格式特点 | 适用阶段 | 卡片 |
|--------|------|-----------|------|---------|---------|------|
| [orca-math-200k](orca-math-200k.md) | EN | 200k(fastrl) | 小学 | 自然语言逐步解 | **阶段2/3 主力** | ✓ |
| [metamathqa](metamathqa.md) | EN | 395k 全量 | 小学→高中 | 自然语言解+type增广 | **阶段2/3 主力** | ✓ |
| [calc-ape210k](calc-ape210k.md) | ZH/EN | 195k(fastrl) | 小学 | 计算器gadget链 | 中文应用题(取题面) | ✓ |
| [gsm8k](gsm8k.md) | EN | 7.5k+1.3k 全量 | 小学 | `#### 答案` | **评测主benchmark** | ✓ |
| [cmath](cmath.md) | ZH | 600+1.7k 全量 | 小学(1-6年级) | 分层(grade/步数/位数) | **评测(仅eval)** | ✓ |
| [calc-gsm8k](calc-gsm8k.md) | EN | ~8k 全量 | 小学 | 计算器gadget链 | 可选(取题面) | ✓ |
| [numinamath-1.5](numinamath-1.5.md) | EN | 298k 采样(全量896k) | **竞赛/奥数** | 题+参考解 | 阶段3/4 难题 | ✓ |
| [openr1-math-220k](openr1-math-220k.md) | EN | 225k(fastrl) | **竞赛** | R1长CoT+正确性标注 | 阶段3/4 长推理 | ✓ |
| [openthoughts3-1.2m](openthoughts3-1.2m.md) | EN | 90k 采样(全量1.2M) | **难** | 长CoT(math/code/sci) | 阶段4(仅math子域) | ✓ |
| [bespoke-stratos-17k](bespoke-stratos-17k.md) | EN | 17k 全量 | **难** | R1长reflection | 阶段4 风格种子 | ✓ |

## 关键判断（为配比铺垫）

**本阶段（基本算术+方法）真正能用的：**
- **英文应用题主力**：`orca-math-200k`(20w, 易) + `metamathqa`(39.5w, 可按 type 调难度, GSM* 段适合早期)。两者**去重后混用**。
- **中文应用题**：`calc-ape210k`(19.5w) + `cmath`(评测) + dynamics 的 math 子集（见通用卡片）。注意 calc-* 的 gadget 链**不直接用**，只取题面+答案自己生成符号 CoT。
- **评测标尺**：英文 `gsm8k` + 中文 `cmath`（**test 严禁进训练**）；cmath 的 grade/步数/位数分层非常适合诊断弱项喂动态课程。

**阶段3 解题 SFT 主力（带解答，教"怎么解"——解题能力在这里建立，不是只留给 RL）：**
- `numinamath-1.5`(题+参考解) / `openr1-math-220k`(题+R1长CoT, 可筛 correctness) / `deepscaler-preview`(题+答案+解, hard) / `openthoughts3-1.2m`(math子域, 长CoT)。
- 课程顺序：阶段2 算术地基 → 阶段3 易(orca)→中(metamath)→难(numina/openr1/deepscaler)。**弱 base 不可早喂竞赛题**（会学成长篇瞎编，见 v3 诊断），故排在算术之后，但必须 SFT。

**阶段4 RL 数据池（题+可验证答案，模型自生成、奖励判对，无需 solution）：**
- `big-math-rl-verified`(25万+, RL 主池) / `dapo-math-17k-dedup`(17k, GRPO 起步, 开箱即用) / `deepscaler-preview`(hard) / gsm8k-train。
- `math-beyond`(181, **hard eval 不训练**)。

> 关键区分：**SFT 解题**需"题+解答/CoT"；**RL**需"题+可验证答案"。numina/openr1/deepscaler 两者皆可（有解答又有答案）；big-math/dapo 只给答案、仅 RL。

## 两条贯穿提醒

1. **格式归一**：这些集大多是自然语言解 / R1 think / gadget 链，与我们 Compute_Cot 的 `<think>…</think>\n#### \boxed{}` 不同。混入前需统一包裹/抽取答案。**计算器 gadget 范式与"教模型自己心算"冲突，倾向弃用 chain 只留题面+答案。**
2. **跨集去重 + 防泄漏**：numina/openr1/openthoughts/metamath 题源高度重叠（openmath/olympiad/GSM/MATH 系），且都可能含 GSM8K/MATH 题——**用 gsm8k/cmath 评测前，务必把训练集与评测题面去重**，否则评测虚高（与我们 Compute_Cot 审计里同样的泄漏教训）。

## 分阶段优先级小结
- **阶段2（算术地基 SFT）**：Compute_Cot(自产 39万,worked CoT) 为主 + orca/metamath(GSM段) 自然语言应用题。
- **阶段3（解题 SFT，易→难）**：orca(易) → metamathqa(中) → openr1/numinamath-1.5/deepscaler(难,带解答)。**解题能力在此建立。**
- **阶段4（RL）**：big-math-rl-verified(主池,待授权) / dapo-math-17k-dedup(起步) / deepscaler / gsm8k-train。
- **评测**：gsm8k(英) + cmath(中) 基础；math-beyond(181) hard 天花板。**test 全部隔离防泄漏。**
- **弃用**：calc-* 的 gadget 链（只取题面）；bespoke-stratos 量小作风格备选。

---

## ⚠️ 2026-06-10 全面审计补充（各卡片已附"审计补充"节,总报告见 `docs/data_audit_report_20260610.md`）

**逐源审计要点速查**：orca 答案抽取 bug(56% 错,P0) / numina 未用 `*_is_valid` 过滤+boxed{proof} 1.8万在训(P1) / openr1 错用 solution 而非 R1 generations+correctness(P1) / ot3 70% 截断+超长,在训仅 1,071 条 / bespoke 1/3 是代码题 / metamathqa·gsm8k 干净 / **ape210k 的 chain 字段=最大中文数学资产(19.5万,95% 带分步链)** / big-math solve_rate=现成课程难度标签 / dapo 含 3,282 条中文题。

**未采纳资产盘点结论**(详见总报告 §四)：
- **接**:tulu-3 `open_math_2_gsm8k`(增量 ~2.9万,格式最干净) / chinese-r1 math repos 34,327(从通用池移入,带 R1 think) / dynamics math_full ~9,000 中文(需跨语言去污染)。
- **选择性**:numinamath-CoT(回填 1.5 缺失解答 + synthetic_amc 62k) / flan cot_gsm8k+aqua 13k / tulutalk Math 29万(英文,带 5 档难度,阶段3 课程用)。
- **不接**:openthoughts-114k(被 ot3 覆盖+超长) / opus-100 / calc-gsm8k(冗余)。
- **mgsm-zh(250)只进评测侧**(它是 gsm8k test 的中文翻译,进训练=污染)。
- **中文数学缺口真实库存**:ape210k 19.5万(chain→CoT) ≫ chinese-r1 3.4万(现成 think) ≫ dynamics 0.9万 ≫ dapo 0.3万;numina 两版 cn_k12 都是英文翻译,指望不上。
