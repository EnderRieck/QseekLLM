# 数据卡片 · AI-MO/NuminaMath-1.5

> 定位：**高中/竞赛符号数学主力**（标准参考解，非长 CoT）；难度偏高。
> 本地：`/data/zilu/math_sft_raw/numinamath-1.5`（**已全量 3 分片 = 896,215 条**，旧注"1/3 采样"过时）

## 总体说明

- **来源**：Numina 项目第二版，聚合奥数/竞赛/教材题 + 清洗后的**参考解答**与最终答案。
- **规模**：全量 **896k**；本地采样了 `train-00000-of-00003.parquet` ≈ **298,739** 条（够看全类别+抽样；如需全量改 allow_patterns 取另两片）。
- **语言**：英文。
- **License**：Apache-2.0。
- **格式**：`problem` / `solution`(参考解) / `answer` / `problem_type` / `question_type` / `source` / `problem_is_valid` / `solution_is_valid` / `synthetic`。

## 子类分布（采样片内，看全类别）

**problem_type（题型）**：Geometry 87.9k / Algebra 75k / Number Theory 48k / Combinatorics 37k / Logic&Puzzles 18.9k / Inequalities 16.8k / Calculus 10.3k / Other 4.6k。
**source（来源）**：olympiads 194k / aops_forum 65.8k / cn_k12 18.3k / metamath 11k / amc_aime 5.5k / olympiads_ref 3.6k —— **以奥数/AoPS 论坛为主，难度高**。

## 抽样

- source=olympiads, problem_type=Number Theory
- problem: `Find all primes p for which there exist positive integers x,y,z such that x^p+y^p+z^p-x-y-z is a product of exactly three distinct primes.`
- solution: `Let A=... For p=2 take x=y=4,z=3 → A=30=2·3·5. For p=3... For p=5... Assume p≥7...`（**简洁的标准证明**，非长回溯）
- answer: `proof`（注意：很多题答案是 `proof`/区间/集合，非单一数值）

## 对本项目的评估

- ⚠️ **难度远超本阶段**（基本算术+方法）；竞赛/证明题为主，**不适合阶段1/2 教弱 base 模型**。
- ✅ 适合**阶段3 难题课程**与**阶段4 GRPO**的题源：题量大、`problem_type`/`source` 可调难度、有 `*_is_valid` 质量旗标可筛。
- ⚠️ 大量 `answer=proof`/非数值答案，**不利于自动判分**（GRPO 需可验证答案——应优先筛 `question_type=math-word-problem` 且 answer 为数值/表达式的子集）。
- 🔎 与 openr1-math-220k 关系：NuminaMath 是**题+参考解**，openr1 是在类似题上加 **R1 长 CoT**。本阶段两者都先不混；难题阶段优先 openr1（带长推理）或在 numina 上自行蒸馏。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：foundation 池入 504,136 条(SFT)/401,570(RL)，cap 90k 在训。**适配器问题多**：
- 🔴 **`problem_is_valid`/`solution_is_valid` 完全没用**：全量 4.1%(~36,859 行)题面非 Yes(Incomplete 3.39% 含波兰语残题 69 行——训练集已实际见到)；solution 非 Yes 另有 ~46k(Incomplete+Problem not solved)。**必须按双 Yes 过滤**。
- 🔴 **`\boxed{proof}` 占位**：answer∈{proof/null/notfound/空} 合计 16.4%；当前在训切片实测 18,207 条 boxed{proof}(占其 20%)+1,129 条空 think。
- 🟠 **字段误用**：适配器把 `problem_type`(学科)当 source 细分，真正的 `source`(cn_k12 30%/olympiads 22%/orca_math 17%/synthetic_math 16.6%/aops_forum 7.6%…)被忽略 → 无法按来源做难度分层(orca/synthetic/cn_k12 偏易，olympiads/cn_contest/amc_aime 偏难)，难度一刀切 hard 是错的。
- 🟠 MCQ 16.3%(boxed 常是选项字母)、`synthetic` 34.8% 未单独配比；aops_forum 32,429 行空 solution(只能 RL)。
- 🟡 RL ground_truth 抽检见脏值(`=b==\frac{1}{3}`、`3,1,=2`)，入池建议过 math_verify.parse 可解析性检查。
- 长度无忧：solution max ~1万字符，全部进 8k 上下文。
- **与 numinamath-CoT(本地已下载未用)的关系**：CoT 版 859k，题面 70% 重合，但其 solution 97.3% 含 boxed 且全短(p99 3.3k 字符)——**可按题面 hash join 回填 1.5 缺/坏解答**；CoT 版独有 synthetic_amc 62k。cn_k12 两版都是英文翻译，**填不了中文缺口**。
