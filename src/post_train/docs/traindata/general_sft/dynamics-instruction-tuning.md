# 数据卡片 · ChiyuSONG/dynamics-of-instruction-tuning

> 定位：**中文、按"能力维度"分类的指令数据**（含数学/推理子集，天然适配按能力切桶的课程）。
> 本地：`/data/zilu/general_sft_raw/dynamics-instruction-tuning`（全量，89M）

## 总体说明

- **来源**：论文《Dynamics of Instruction Tuning》配套数据，把中文指令按**10 大能力**人工整理（curated）+ 一份 GPT-4 合成通用集（synthetic）。
- **规模**：curated full 合计 ~41k + synthetic 41k ≈ **82k**。
- **语言**：中文（understanding/创作等），少量英文 code。
- **格式**：`messages`(chat) / `idx` / `type`(=能力类别) / `question_format`。每个能力有 `1000`(小样)、`full`(全量)、`test`、`valid` 四档切分。

## 能力类别（10 类 curated，看全所有类别 + 规模）

| 能力(type) | full 行数 | 说明 |
|-----------|---------:|------|
| reasoning | 12,751 | 逻辑/常识推理 |
| math | 11,501 | **中文数学题**（含小学应用题） |
| code | 4,968 | 代码 |
| understanding | 4,885 | 阅读理解/语言理解 |
| history | 1,856 | 历史知识 |
| chinese | 1,434 | 中文语言（成语/古文等） |
| biology | 1,041 | 生物知识 |
| creative_writing | 1,000 | 创意写作 |
| ethics | 1,000 | 价值观/伦理 |
| role_play | 1,000 | 角色扮演 |

- 另有 `curated/test/` 下的 **C-Eval** 子集（physician / teacher_qualification / urban_and_rural_planner）作专业评测。
- `synthetic/`：`general_instruction_gpt4_41k`（GPT-4 生成的通用指令 41k）。

## 抽样（math，中文逐步解）

- U: `小红夏天喜欢吃西瓜，爸爸一共买回来3个西瓜，小红和小弟弟吃了2个，还剩几个西瓜？`
- A: `首先理解题意…用数学符号表示：3 - 2 = ? 计算得出：3 - 2 = 1。答案：还剩下1个西瓜。`（**中文、逐步、口语化**）

## 对本项目的评估

- ✅ **中文 + 按能力分类**，与我们"按 source/能力切桶的动态课程"理念天然契合：可直接按 type 做均衡或课程。
- ✅ 自带 `math`/`reasoning` 中文子集（数学逐步解风格朴素），对中文数学+逻辑有正面帮助；自带 test/valid 切分省事。
- ✅ 与 COIG-CQIA 互补，两者构成**中文通用 SFT 主力**（CQIA 偏真实网络场景，dynamics 偏能力维度齐整）。
- ⚠️ 量不大（~82k）；synthetic 部分是 GPT-4 合成，质量一般。
- 🔎 math 子集可与我们 Compute_Cot 中文题协同，但其解答偏口语，若并入需轻度格式归一。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：38,885 全量入池在训(curated/full + synthetic 41k)。

- 🔎 **math_full 11,501 条(79% 中文、96% 带多步推理、LaTeX 规范)是现成优质中文数学 CoT**——目前混在通用池里当纯对话训,应捞进数学池包 think 格式。
- 🔴 **但это MATH/GSM8K 的中文翻译**:若捞进数学池必须与 gsm8k/MATH 评测做跨语言去污染(qhash 截不住翻译,需按数值/结构匹配或回链原题)。
- 🔴 **reasoning_full 12,751 条全是"选择题→裸字母答案"**(无推理过程)——正是"MCQ 空区/裸字母"病征的训练侧来源之一,原样混训在教模型不思考作答;建议从通用池剔除或重造带 CoT 的版本(自带 answer 字段可验证)。
- curated/{test,valid} 切分现成,可作过程评测题源。
