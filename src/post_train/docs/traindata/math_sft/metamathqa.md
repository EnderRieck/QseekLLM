# 数据卡片 · meta-math/MetaMathQA

> 定位：**英文数学 SFT 主力之一**（GSM8K+MATH 的自举增广，量大、难度跨小学到高中竞赛）。
> 本地：`/data/zilu/math_sft_raw/metamathqa`（全量，单文件 `MetaMathQA-395K.json`）

## 总体说明

- **来源**：MetaMath 论文，对 GSM8K 与 MATH 的**训练集**做多种自举增广（改写、自我验证、正反向构造），**数据卡声明不含测试集泄漏**。
- **规模**：**395,000** 条。
- **语言**：英文。
- **License**：MIT。
- **格式**：`query`(题) / `response`(逐步解，含最终答案) / `type`(增广方式) / `original_question`(原题)。

## 子类（type，8 种增广 = 看全所有类别）

| type | 数量 | 含义 |
|------|-----:|------|
| GSM_Rephrased | 80,000 | GSM8K 题面改写 |
| GSM_AnsAug | 80,000 | GSM8K 答案增广（多解法） |
| MATH_AnsAug | 75,000 | MATH 答案增广 |
| MATH_Rephrased | 50,000 | MATH 题面改写 |
| GSM_FOBAR | 40,000 | 反向构造（已知答案求未知量） |
| GSM_SV | 40,000 | Self-Verification 自验证式 |
| MATH_FOBAR | 15,000 | MATH 反向构造 |
| MATH_SV | 15,000 | MATH 自验证 |

> 即：GSM*（小学，~24w）+ MATH*（高中竞赛，~15.5w）。

## 抽样

- type=MATH_AnsAug
- query: `Joe chooses 1+2i, Gracie chooses -1+i. How far apart are the points?`
- response: `The distance formula is √((x2-x1)²+(y2-y1)²). Joe (1,2), Gracie (-1,1). So √((-2)²+(-1)²)=√5.`（逐步、自然语言）

## 对本项目的评估

- ✅ **量大（39.5w）、自然语言逐步解、含最终答案**，是阶段2/3 英文数学 SFT 的主力候选；GSM* 部分难度适中（适合早期课程），MATH* 部分偏难（适合后期加难）。
- ✅ 可按 `type` 精细配比：早期多喂 GSM_*，后期掺 MATH_*；FOBAR/SV 增广提供"反向/自检"多样性。
- ⚠️ 解答是自然语言（非 `<think>` 格式），混入需**格式归一**；答案散在文末需抽取。
- ⚠️ MATH 部分用 LaTeX，渲染/清洗要注意。
- 🔎 与 orca-math（更易、更口语）互补：MetaMathQA 覆盖更广、更"标准解法"。建议两者去重后混用。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：入池 132,770(泄漏剔除 75,081 后)，cap 11万(easy 优先)在训。

- ✅ **答案抽取最可靠**："The answer is:" 收尾覆盖 100%(抽 5,000 全中)，实测在训切片 boxed 与解答结论不一致仅 0.7% —— 是 orca bug 的干净对照组。
- 🟠 **泄漏剔除 75,081 条过狠的根因**：competition-math 基准目录是全量 MATH 12.5k(含 train 7.5k)，MATH_* 增强段(39.2%，15.5万)凡保留原题题面的全被误杀。基准侧换成 MATH test-only 后可释放大批 MATH-train 增强样本(阶段3 难度阶梯急需)。
- 🔎 **`original_question` 字段 100% 存在但未用**：可用于①回填 MATH level 难度(与 MATH 原集串匹配)；②增强题与原题的防泄漏审计。
- type 分布：GSM 系 60.8%(24万)/MATH 系 39.2%；MATH 段约 18% 答案是 LaTeX 表达式(math_verify 可处理)。
- ⚠️ 45% 题是 FOBAR/SV 反推式("求未知变量 X")——与"风格冲突"诊断相关，阶段2 建议对此类降权。
