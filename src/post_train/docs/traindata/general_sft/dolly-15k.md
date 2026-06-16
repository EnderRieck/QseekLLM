# 数据卡片 · databricks/databricks-dolly-15k

> 定位：**英文通用 SFT（商用友好）**，人工手写、8 类指令。
> 本地：`/data/zilu/general_sft_raw/dolly-15k`（全量）

## 总体说明

- **来源**：Databricks **5000+ 名员工手写**（2023），仿 InstructGPT 8 类能力。
- **规模**：**15,011** 条（单 split）。
- **语言**：英文。
- **License**：**CC-BY-SA-3.0（可商用）**——相对 no_robots(NC) 的优势。
- **格式**：`instruction` / `context`(可空，仅部分类有) / `response` / `category`。

## 子类（category，8 类）

| category | 数量 | 是否带 context |
|----------|-----:|------|
| open_qa | 3,742 | 否（靠知识答） |
| general_qa | 2,191 | 否 |
| classification | 2,136 | 否 |
| closed_qa | 1,773 | **是**（给 passage） |
| brainstorming | 1,766 | 否 |
| information_extraction | 1,506 | **是** |
| summarization | 1,188 | **是** |
| creative_writing | 709 | 否 |

## 各类抽样

- **closed_qa**：`When did Virgin Australia start operating?` + [维基段落] → `commenced services on 31 August 2000 as Virgin Blue...`
- **classification**：`Which is a species of fish? Tope or Rope` → `Tope`
- **open_qa**：`Why can camels survive for long without water?` → `Camels use the fat in their humps...`
- **information_extraction**：`If I have more pieces at stalemate, have I won?` + [规则段] → `No. Stalemate is a drawn position...`
- **creative_writing**：`write a scene between two actors discussing Inception` → 对白脚本
- **brainstorming**：`Why mobile is bad for human` → `We are always engaged one phone which is not good.`（**注意：部分回答很短、质量参差**）

## 对本项目的评估

- ✅ **可商用** + 人工手写，类目与 no_robots 几乎一一对应（两者可互补/合并）；阶段1 指令跟随的好料。
- ✅ closed_qa/info_extraction/summarization 带 context，能教"基于给定材料作答"（抗幻觉）。
- ⚠️ **质量不如 no_robots 均匀**：部分回答过短、口语化甚至带语病（brainstorming 尤甚），混入前建议**按长度/质量过滤**。
- ⚠️ 纯英文；三段式需转 chat（context 拼进 user）。
- 🔎 与 no_robots 配合：no_robots 当"高质量精料高权重"，dolly 当"商用安全的量补充"，注意去重（两者题材有重叠风格）。

## 审计补充（2026-06-10 复核）

- 14,785 全量在训。复核:质量中等,open_qa 常一句话短答,有过时/简陋样本——"过滤后用"的计划未实现,实际全量进了。
- 建议:要么降权,要么只留 closed_qa/information_extraction/summarization 等带 context 任务(~4.5k)。优先级低。
