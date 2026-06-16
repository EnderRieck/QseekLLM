# 数据卡片 · microsoft/orca-math-word-problems-200k

> 定位：**英文小学应用题主力**（自然语言解题，非长思维链）。
> 本地：`/data/zilu/fastrl/data/train/orca-math-200k`（已有，save_to_disk 格式，**可直接复用**）

## 总体说明

- **来源**：微软用 GPT-4 增广生成的小学数学应用题（基于 Orca-Math 论文的 agent-generated 题库）。
- **规模**：**200,035** 条（单 train split）。
- **语言**：英文。
- **License**：MIT。
- **格式**：极简两字段 `question` / `answer`。`answer` 是**自然语言的逐步解题**（非 `<think>` 包裹，非计算器 trace），结尾给出数值答案。**无子类标签**。

## 抽样

**例1（简单推理）**
- Q: `Jungkook is the 5th place. Find the number of people who crossed the finish line faster than Jungkook.`
- A: `If Jungkook is in 5th place, then 4 people crossed the finish line faster than him.`

**例2（组合枚举，过程清晰）**
- Q: `...make a three-digit number from digits 1,6,8 without repetition... find the sum of the second smallest and third smallest...`
- A: 列出全部 6 个排列 `168/186/618/681/816/861` → 升序 → 指出第二小 186、第三小 618 → 求和。**逐步、干净**。

## 对本项目的评估

- ✅ **量大（20w）、纯英文小学应用题**，覆盖加减乘除/分数/比例/组合等基础运算的"应用"场景，正好补我们 Compute_Cot 合成题"偏纯符号、缺自然语言情境"的不足。
- ✅ MIT、字段极简，易转成我们的 chat 格式。
- ⚠️ 解答是**自然语言风格**，不是我们的 `<think>…</think>\n#### \boxed{}` 格式——若要混入需做**格式归一 / 重新包裹**，且答案散落在文末需抽取。
- ⚠️ 质量有噪声（GPT-4 生成，少数题解略啰嗦或含小瑕疵），不像 no_robots 那样人工精校。
- ⚠️ 纯英文；中文应用题侧由 calc-ape210k 覆盖。
- 🔎 无难度/子类标签，难度整体偏"小学"，适合**阶段2/3 早期课程**（简单多），不适合喂难题阶段。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：foundation 全留 176,899 条(SFT+RL 双池)，占当前训练集 14.9%。

- 🔴 **答案抽取 bug 实锤**(`data_pipeline/adapters.py:_num_from_text`)：原始只有 question/answer 两字段、无独立 gold，答案靠正则从解答抽；回退正则在末 200 字符**取第一个 `=`/"answer is" 后的数** → 解答末尾常是连串中间等式，抓到中间结果。实测 **56.4%(99,268 条) boxed 数值不出现在解答结论句**；`ground_truth` 同源同错 → **RL 池同样被毒**。
- 解答风格统计(抽 5,000)："answer is" 收尾仅 0.3%(管线主正则基本不命中)；68.7% 末句以 So/Therefore/Thus 开头；34.7% 末句含多个数字(末数策略高风险)。
- **修复策略已验证**：「末个含数句中取最后一个 is/are/=/: 之后的数 + 分数(`a/b`)整体捕获 + 百分数归一」人工抽读 20 条 90% 正确(预计 92-95%)；约 5% 低置信(末句多数字无引导词)建议直接丢弃。
- ⚠️ **SVAMP 评测泄漏源**：svamp 基准 300 题有 63 题(21%)逐字在 orca 里(orca 题库吸收了 SVAMP/ASDiv 系题源) → heldout svamp 读数虚高，修复见审计报告 §二。
- ⚠️ 与 big-math 重叠：big-math 的 source=orca_math 占其 33%(83,215 条)，跨集去重靠 qhash first-wins 已处理(orca 先入)。
