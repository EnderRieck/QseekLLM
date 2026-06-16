# 数据卡片 · open-r1/OpenR1-Math-220k

> 定位：**长思维链（R1 蒸馏）高难度推理**，偏竞赛/奥数，远超"基本算术"。
> 本地：`/data/zilu/fastrl/data/train/openr1-math-220k/all`（已有，**可直接复用**）

## 总体说明

- **来源**：open-r1 项目，用 **DeepSeek-R1** 对 22 万道数学题各生成 2–4 条带 `<think>` 的长推理 trace，并用 math-verify + Llama 双重判对。
- **规模**：`all` config **225,129** 行（每行=一道题 + 多条 R1 生成）。
- **语言**：英文（题面）。
- **License**：Apache-2.0。
- **格式（字段丰富）**：`problem` / `solution`(参考解) / `answer` / `problem_type` / `question_type` / `source` / `generations`(R1 的 `<think>...` 长链列表) / `correctness_math_verify` / `correctness_llama` / `correctness_count` / `messages`(chat 含 think) / `uuid`。

## 子类分布（看全所有类别）

**problem_type（题型）**
| 类型 | 数量 |
|------|-----:|
| Algebra | 103,241 |
| Geometry | 46,386 |
| Number Theory | 25,249 |
| Combinatorics | 20,733 |
| Calculus | 10,674 |
| Inequalities | 9,114 |
| Logic and Puzzles | 7,928 |
| Other | 1,804 |

**source（来源难度）**：olympiads 97k / cn_k12 91k / cn_contest 18k / aops_forum 12k / amc_aime 3.9k / inequalities 1.3k / olympiads_ref 665 / number_theory 523 —— **以奥数/竞赛 + 中国 K12 竞赛为主**。

**question_type**：math-word-problem 164k / MCQ 56k / proof 3.7k / other 221。

平均每题 ~2 条 R1 生成（`correctness_count` 记录其中判对的条数，可用于筛"全对"的高质量 trace）。

## 抽样

- problem: `9.043. $\sqrt{x^{2}-x-12}<x$.` （Inequalities / olympiads）
- solution: 转成不等式组 `x²-x-12≥0, x>0, ...` → 解得 `x∈[4,∞)`
- generations[0]: `<think>\nOkay, let's see. I need to solve √(x²-x-12)<x. First, the expression inside the root must be non-negative...` —— **典型 R1 长链：反复试探、自我检查、回溯**，篇幅很长。
- messages: 标准 chat，assistant 内容含完整 `<think>` 推理 + 最终答案。

## 对本项目的评估

- ⚠️ **难度远超本阶段目标**（本阶段=基本算术+方法）。奥数/竞赛题 + R1 超长回溯式推理，**不适合阶段1/2 教弱 base 模型基本运算**——会引入过难分布和冗长跑题风格。
- ✅ **但对后续阶段（阶段3 难题课程、阶段4 GRPO）极有价值**：现成的高质量长 CoT、带正确性标注、可按 `correctness_count` 筛全对 trace、按 `problem_type`/`source` 调难度配比。
- ✅ 字段齐全，`messages` 已是 think-包裹的 chat，便于直接用。
- 🔎 **建议**：本阶段**不混入或仅极少量**；留作难题/RL 阶段的主力长推理来源。筛选时优先 `correctness_count≥2` 且 `question_type=math-word-problem`。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：入池 188,685(SFT)/188,690(RL)，cap 5万在训。

- 🔴 **核心错用：适配器拿原始 `solution`(人写参考解)当 CoT，完全没用 R1 字段**。原始 14 字段里有 `generations`(R1 完整 CoT，含 think)/`messages`(现成 user/assistant 对话)/`correctness_math_verify`/`correctness_llama`/`correctness_count`/`is_reasoning_complete`——这正是该集的价值所在；而 `solution` **52.6% 不含 boxed**、6% 无干净收尾，实测在训切片 14.4% 解答结论与 boxed 脱节 + 763 条空 think。**计划里"筛 correctness_count≥2"从未实现**。
- **修法**：改用 `correctness_math_verify=True` 的 generation(87.3% 题至少一条全对)或直接用 `messages`；`correctness_count`(=1 占 16.3%，=2 占 82.4%)可当难度代理。
- ⚠️ **长度**：best-generation 中位 9,886 字符、**13.8% 超 24,000 字符(约 8k token)**——改用 R1 CoT 后需 16k 上下文或按长度分桶。
- 构成：source 以 olympiads 43.2%/cn_k12 40.6% 为主(均英文)；MCQ 25.1%；中文 0.76%。题面引用图片 2.9%。
