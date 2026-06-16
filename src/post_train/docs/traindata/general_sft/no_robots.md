# 数据卡片 · HuggingFaceH4/no_robots

> 用途定位：**阶段1 初步 SFT 的高质量种子**（指令跟随 + 对话格式对齐 + 语言表达）。
> 本地路径：`/data/zilu/general_sft_raw/no_robots`（全量）

## 总体说明

- **来源**：HuggingFace H4 团队，仿 OpenAI InstructGPT 论文的指令类型，由**专业标注员纯手写**的 10k 条高质量指令-回答（"No Robots" = 没有模型生成、全人工）。
- **规模**：9,500 train + 500 test = **10k**。
- **语言**：**纯英文**。
- **License**：CC-BY-NC-4.0（**非商用**，研究/内部训练可用，需留意）。
- **格式**：chat 多消息。字段 `prompt`(首问) / `prompt_id` / `messages`([{role, content}]) / `category`。
  绝大多数是**单轮**（user→assistant）；只有 `Chat` 类是**多轮**且可能带 `system`。

## 子类（category 列，10 类）

| category | 数量 | 占比 | 说明 |
|----------|-----:|-----:|------|
| Generation | 4,346 | 45.7% | 开放式生成（写信/文案/故事…），**绝对主力** |
| Open QA | 1,182 | 12.4% | 开放问答（靠模型知识答） |
| Brainstorm | 1,060 | 11.2% | 头脑风暴/列点建议 |
| Chat | 796 | 8.4% | **多轮对话**（含角色设定 system） |
| Rewrite | 625 | 6.6% | 改写/风格转换 |
| Summarize | 395 | 4.2% | 给定文本做摘要 |
| Coding | 334 | 3.5% | 写代码/解释代码 |
| Classify | 334 | 3.5% | 分类并给理由 |
| Closed QA | 245 | 2.6% | 给定 passage 抽取式问答 |
| Extract | 183 | 1.9% | 结构化抽取（列表/字段） |

## 各子类抽样

**Generation**
- U: `Help write a letter of 100-200 words to my future self for Kyra, reflecting on her goals...`
- A: `Dear Future Self, I hope you're happy and proud of what you've achieved...`（流畅、有结构的长文）

**Open QA**
- U: `Why is Aberdeen called the Granite City?`
- A: `Aberdeen's nickname comes from the city's history and contribution of granite...`（无给定材料，靠知识）

**Closed QA**（给材料，抽取式）
- U: `Answer using the passage below: Who was Harry Potter written by? ... [Harry Potter is a series of seven fantasy novels written by J.K. Rowling...]`
- A: `Harry Potter was written by British author J. K. Rowling. Published in the UK by Bloomsbury and US by Scholastic Press.`

**Chat**（多轮 + system 设定）
- S: `Frank is a French translational chatbot, it translates all words into French.`
- U: `What a wonderful morning!` → A: `Quelle merveilleuse matinée !` →（继续多轮翻译）

**Coding**
- U: `I have 2 lists... show me how to make a bar chart using Python` → A: 安装说明 + 完整 ```python``` 代码块（matplotlib）

**Classify**
- U: `Classify this review as Positive or Negative. Provide the reason. Review: I recently flew with Delta...`
- A: `Positive\nFood, service, and entertainment were praised...`

**Extract**
- U: `Please extract a bulleted list of the percentages... If the data is not given, please state so.`
- A: `- Ages 25 to 34: 32 percent\n- Ages 35 to 44: (data not provided)...`（忠实标注缺失项）

**Summarize / Rewrite / Brainstorm**：分别为"给文本→摘要目标"、"按角色/风格改写全文"、"列点式建议（如柏林选区推荐）"。

## 对本项目的评估

- ✅ **质量极高、纯人工**，指令类型覆盖广（生成/问答/抽取/改写/分类/代码/多轮）；正是阶段1"指令跟随 + 对话格式对齐 + 语言表达"想要的。
- ✅ 格式已是标准 chat `messages`，可直接对齐我们的对话模板。
- ⚠️ **纯英文**——中文能力得靠 COIG-CQIA / dynamics 等补；做中英配比时这是英文侧的"质量锚"。
- ⚠️ **仅 10k**，是"精"不是"量"：适合做高质量种子/小比例高权重混入，不能当通用语料的主体。
- ⚠️ **NC 非商用** license，内部研究训练 OK，若日后商用需替换。
- 🔎 类目极不均衡（Generation 占 46%），若按 category 均衡采样需上采样 Extract/Closed QA 等小类。

## 审计补充（2026-06-10 复核）

- 9,948 全量在训,人写质量高,定位正确(英文精料锚)。
- 🔎 test_sft 500 条未用——可作通用对话过程评测题源(probe-chat 扩充)。category 含 Extract/Classify 等小类,上采样计划未实现(影响小)。
