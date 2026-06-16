# 数据卡片 · m-a-p/COIG-CQIA

> 定位：**中文通用 SFT 主力**（指令跟随 + 多领域中文表达 + 防中文退化）。
> 本地：`/data/zilu/fastrl/data/train/coig-cqia`（fastrl 已下 13 个子来源，**可直接复用**；download_general_sft 也会下一份）

## 总体说明

- **来源**：m-a-p 的 **COIG-CQIA**（Chinese Open Instruction Generalist - Quality Is All You Need），从中文互联网真实场景采集 + **人工校验**的高质量中文指令数据。
- **规模**：13 个子来源合计 **44,694** 条。
- **语言**：中文（含少量繁体）。
- **License**：见 repo（研究用途）。
- **格式**：`instruction` / `input`(可空) / `output` + 丰富元信息 `task_type{major,minor}` / `domain` / `human_verified` / `answer_from` / `copyright`。**子类 = 13 个来源目录**。

## 子来源（13 类，全部覆盖）

| 子来源 | 条数 | 内容 / 典型 task_type |
|--------|-----:|------|
| finance | 11,288 | 金融文本的摘要/标题生成/结构化抽取（最大） |
| wiki | 10,603 | 百科问答（健康/常识…） |
| zhihu | 5,631 | 知乎问答（含技术，如"求整数素因子思路"） |
| exam | 4,856 | 中学考试题（语文阅读理解等） |
| douban | 3,086 | 图书介绍 |
| coig_pc | 3,000 | 各类 NLP 任务（NER/信息抽取，带 {{类型:文本}} 标记） |
| xhs | 1,508 | 小红书风格文案生成 |
| wikihow | 1,485 | how-to 步骤说明 |
| chinese_traditional | 1,111 | 成语释义 / 传统文化 |
| human_value | 1,007 | 价值观类开放问答 |
| segmentfault | 458 | 编程/代码问答 |
| logi_qa | 421 | **逻辑推理**（如"只说真话/假话"经典谜题） |
| ruozhiba | 240 | 弱智吧——**隐喻理解/脑筋急转弯**，以"刁钻但答得好"著称 |

## 抽样

**ruozhiba（高难隐喻）**
- I: `天下没有不散的筵席，那么我们相聚的意义又是什么`
- O: `尽管相聚时光有限，但意义在于创造美好回忆和珍贵关系…让我们感受到生活的丰富和美好。`

**logi_qa（逻辑谜题）**
- I: `甲只说假话、乙只说真话，只点头摇头…面对 A/B 两条路通向京城…如何一问辨路？`
- O: `问其中一人"如果我问另一个人这条路通不通京城，他会怎么回答"…都摇头则走这条，都点头走另一条。`

**zhihu（技术问答）**
- I: `求一个整数的所有素数因子的思路是什么？`
- O: `1. 简单筛法… 2. 更大的数用 Pollard ρ 算法：x_k=(x_{k-1}²+1) mod n …`

**coig_pc（结构化抽取）**：把新闻文本标成 `{{location:中关村医院}}…{{person_name:方非}}…` 格式。

## 对本项目的评估

- ✅ **中文通用 SFT 的最佳来源之一**：人工校验、领域广（问答/抽取/生成/逻辑/文案/考试）、风格自然，正是阶段1"中文指令跟随 + 语言表达"和阶段2"防中文退化"想要的。
- ✅ 含 `logi_qa`/`ruozhiba` 的**逻辑与隐喻**，对"逻辑思维"目标有正面帮助；`zhihu`/`segmentfault` 含技术与轻量代码。
- ✅ 已有 fastrl 本地副本，省下载。
- ⚠️ `instruction`+`input`+`output` 三段式，需转成我们的 chat `messages`（input 拼进 user）。
- ⚠️ 领域极不均衡（finance+wiki 占近半），且 finance/xhs 偏"特定风格文案"，做通用配比时建议**对 logi_qa/ruozhiba/human_value 等小而精的类上采样**，对 finance/xhs 下采样。
- 🔎 与 dynamics-of-instruction-tuning（也是中文、按能力分类）互补，两者一起构成中文侧主力。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：43,356 全量入池在训(43,343 经 8k 过滤)。

- 各子集质量复核：logi_qa(421,带推理过程,好)/zhihu/wiki/wikihow/segmentfault(人写,好)/ruozhiba(澄清式回答,好)/finance(课文式,11,288 占 1/4,口语自然度差)/coig_pc(NLU 老式任务,平)/chinese_traditional 抽检见 1 条答非所问脏数据。
- ⚠️ **exam 子集数学仅 ~57 条**(法律 2,515/历史 608 为主)——别指望它补中文数学。
- 🔎 `human_verified`/`task_type.major` 字段可做优先级与类型配比,目前未用。
- 定位维持:中文通用增味料全量混入;chinese_traditional 建议抽检剔错。
