# 评测 Benchmark 卡片集（fastrl 已下载，可直接复用）

> 本地：`/data/zilu/fastrl/data/benchmark/`（save_to_disk 格式，`load_from_disk` 可读）。
> 全部仅作**评测，绝不进训练**；与训练集**题面去重**（competition-math/MATH 系尤其要防，MetaMath/numina/big-math 都含 MATH 题）。

## 可用性总览

| benchmark | 语言 | 规模(test) | 测什么 | 答案/判分 | 对本项目 |
|---|---|---|---|---|---|
| **gsm8k** | EN | 1,319 | 小学应用题多步 | `#### 数字` | 🎯 核心数学评测 |
| **cmath** | ZH | 1,098(+600val) | 小学数学(1-6年级) | 数值`golden`，带 grade/步数/位数分层 | 🎯 核心中文数学，**分层诊断神器** |
| **competition-math** | EN | 12,500 | MATH 竞赛(代数~微积分) | `\boxed{}`，带 level1-5/type | 🎯 难数学评测；**注意防泄漏** |
| **agieval-gaokao-mathcloze** | ZH | 118 | 高考数学填空 | 数值`answer` | 🎯 中文高考数学 |
| **agieval-gaokao-mathqa** | ZH | 351 | 高考数学选择 | `gold`(选项idx) | 🎯 中文高考数学 MCQ |
| **cmmlu**(math 两科) | ZH | ~210 | 大学/高中数学知识 MCQ | `Answer`(A-D) | ✅ 中文数学知识，量小 |
| **bbh** | EN | 27×250=6,750 | Big-Bench Hard 推理(逻辑/算术/…) | `target` | ✅ 通用推理评测 |
| **zebralogic** | EN | 1,000 | 逻辑网格谜题 | 结构化`solution` | ✅ 逻辑推理(难) |
| **ifeval** | EN | 541 | 指令跟随(格式约束) | 规则校验(可验证) | ✅ **阶段1 指令跟随评测** |
| mmlu-stem | EN | 3,153 | STEM 知识 MCQ | `answer`(idx) | ⚪ 弱相关(知识非推理) |
| gpqa-diamond | EN | 198 | 研究生级理科 QA | `Correct Answer` | ⚪ PhD 级超纲，非数学，暂不用 |

> 🎯=核心评测 ✅=有用 ⚪=边缘/暂不用

---

## 逐个详情

### gsm8k（EN 小学数学）
- test 1,319 / train 7,473。`question` / `answer`(多步+`<<计算>>`标注+`#### 数字`)。
- 例：`Natalia sold clips to 48 friends... half as many in May. How many altogether?` → `…#### 72`。
- 判分：解析 `####` 后数字精确匹配。Verl `reward_score/gsm8k` 现成。**英文小学数学主标尺。**

### cmath（ZH 小学数学，分层）
- val 600 / test 1,098。`grade`(1-6) / `question` / `golden`(数值) / `reasoning_step` / `num_digits`。
- 例：grade1 `芳芳买了一本书有99页，看了90页，还剩多少页？` → `9`。
- 判分：数值匹配。**最大价值=三维分层**（年级/步数/位数）→ 直接诊断模型在哪个难度档掉链子，喂动态课程。**中文小学数学主标尺。**

### competition-math（EN MATH 竞赛）
- 12,500 题。`problem` / `level`(Level 1-5) / `type`(Algebra/Geometry/…) / `solution`(含 `\boxed{}`)。
- 例：分段函数连续性求 `a+b`（Level 5, Algebra）。
- 判分：抽 `\boxed{}` 用 `math_verify`(sympy 等价)。可按 level/type 分层。
- ⚠️ **泄漏高危**：MetaMathQA / numinamath / big-math / openthoughts 都含/源于 MATH → 用作评测前**必须与训练集题面去重**。标的是 `train` split，实际就是 MATH 题库，按需取子集做 eval。

### agieval-gaokao-mathcloze（ZH 高考填空）
- test 118。`query`(中文题,LaTeX) / `answer`(数值/表达式)。
- 例：分段函数 `f[f(√6)]=...` 求值 → `2`。判分：math_verify/数值匹配。**中文高考数学(填空)。**

### agieval-gaokao-mathqa（ZH 高考选择）
- test 351。`query` / `choices`(4选项) / `gold`(正确项 idx 列表)。
- 例：集合 `A∩B` → 选 (D)。判分：选项匹配。**中文高考数学(选择)。**

### cmmlu（ZH 数学知识 MCQ）
- 两科：`college_mathematics`(105) + `high_school_mathematics`。`Question`/`A`/`B`/`C`/`D`/`Answer`。
- 例：n阶矩阵行列式 `|-2(...)|` → B。判分：选项匹配。量小，作中文数学**知识**补充评测。

### bbh（EN 通用推理，27 任务）
- 27 个 config 各 250 题（boolean_expressions / logical_deduction / multistep_arithmetic_two / object_counting / …）。`input` / `target`。
- 例：`not ( True ) and ( True ) is` → `False`。判分：target 匹配（部分需抽取）。**通用推理评测**；其中 multistep_arithmetic/object_counting 与数学相关。

### zebralogic（EN 逻辑谜题）
- test 1,000。`id`/`size`(如5×6)/`puzzle`(爱因斯坦式网格约束) / `solution`(结构化表)。
- 判分：结构化解匹配（较复杂）。**逻辑推理(难)评测**，呼应"逻辑思维"目标。

### ifeval（EN 指令跟随）
- 541。`prompt` / `instruction_id_list` / `kwargs`（如 `no_comma`、`number_words≥300`、`highlighted_sections=3`）。
- 例：`Write a 300+ word summary... Do not use any commas...`。判分：**规则可验证**（是否满足各格式约束）。**🎯 阶段1 指令跟随的标准评测**——比主观问答客观。

### mmlu-stem（EN STEM 知识）⚪
- test 3,153。`question`/`choices`/`subject`/`answer`(idx)。知识型 MCQ，与"数学推理"弱相关，优先级低。

### gpqa-diamond（EN 研究生理科）⚪
- 198 题，PhD 级物理/化学/生物 MCQ（专家都需~15-30min），**远超纲且非数学**，暂不用；字段极多（含各 validator 标注），真用只取 `Question`/`Correct Answer`/`Incorrect Answer 1-3`。

---

## 评测集组合建议

- **数学主评测**：gsm8k(英小学) + cmath(中小学,分层) + competition-math(英难,去重后) + agieval-gaokao-math{cloze,qa}(中高考)。
- **过程诊断**：cmath 的 grade/步数/位数、competition-math 的 level/type → 按难度分层看弱项，喂动态课程。
- **能力旁路**：ifeval(指令跟随,阶段1) + bbh/zebralogic(推理) 作辅助，看是否"只会数学不会通用/逻辑"。
- **难度天花板**：math-beyond(见 math_sft) + competition-math Level5；gpqa/mmlu-stem 暂搁置。
- **铁律**：所有 benchmark **test 隔离 + 与训练集题面去重**（尤其 MATH 系）。
