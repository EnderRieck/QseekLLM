# 数据卡片 · Congliu/Chinese-DeepSeek-R1-Distill-data-110k

> 定位：**中文 R1 蒸馏**（通用+数学，防中文退化 + 推理曝光）。
> 本地：`/data/zilu/fastrl/data/train/chinese-deepseek-r1-distill`（HF arrow，110,000 行）

## 总体说明

- **字段**：`input`(中文题面/指令)、`content`(最终回答)、**`reasoning_content`(R1 完整 think 轨迹,100% 非空,median 515 token/p90 1,493)**、`repo_name`(领域来源)、`score`、各段 token 长度。
- **repo_name 分布**：coig/neo 52,893(48.1%) / EduChat-Math 19,729(17.9%) / GSM8K_zh 8,776(8.0%) / applied_math 7,493(6.8%) / stem_zh 四科 12,648(11.5%) / zhihu 2,534 / coig_exam 1,954 / 其余小类。
- **score 语义分 repo**：数学类是二值 10/0(可验证对错)；coig/neo 等是 0-10 主观分(峰值 8-9)。score<7 仅 2.9%。

## 审计发现（2026-06-10）

**实际使用现状**：通用池 cap 8万(score≥7 过滤,基本不筛掉什么),只用 `content` —— **`reasoning_content` 全量闲置**。

- 🔴 **全场最优中文数学资产被埋没**：math repos(EduChat-Math+GSM8K_zh+applied_math+Advanced-Math+kaoyan)= 36,945 条,其中 **score≥8(验证通过)= 34,327 条**:中文题面 + R1 think 轨迹 + 规范解答 + \boxed 答案,可直接映射到我们 `<think>` 格式。中文数学缺口(P0)的最快解。
- 🔎 repo_name 是现成领域标签,可做配比;stem_zh/phy 3,181 条理科计算可一并捞。
- ⚠️ 去污染:GSM8K_zh 是 gsm8k 的中文翻译——与 gsm8k 评测跨语言去污染(qhash 截不住翻译;建议按数值答案+结构匹配,或整体回避 test 段)。已实测 1 条 cmath 评测题经此源泄入通用池。

## 对本项目的评估

- ✅ 中文 R1 蒸馏里规模/质量/标注三全的唯一选项;通用段(coig/neo)继续当防退化中文池。
- 行动:阶段2 数据构建时,把 math repos 子集从通用池移到数学池,改用 reasoning_content 包 think。
