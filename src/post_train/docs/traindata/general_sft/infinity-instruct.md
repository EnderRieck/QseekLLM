# 数据卡片 · BAAI/Infinity-Instruct

> 定位：**大规模双语基础指令 SFT 主力**（千万级、对话格式、带细粒度能力标签）。
> 本地：`/data/zilu/general_sft_raw/infinity-instruct`（**6 个 config 各采样 1 分片 = 9×100k = 90 万条**；全量千万级）
> 注：该 repo 在 hf-mirror 镜像报 403，本地是用**真 hub+代理**下的。

## 总体说明

- **来源**：BAAI（智源）整理的大规模高质量指令数据，聚合+清洗多来源，**中英双语**，配套细粒度能力标注。
- **规模**：全量 **7M+**（多个 config 累加）；本地每 config 取首个分片(各 10 万)做抽样。
- **语言**：**英文为主（~97%）+ 中文（zh-cn）** + 零星 fr/es/ko/pt。
- **License**：见 repo。
- **格式**：`id` / `conversations`([{from,value}] 多轮 chat) / `label`(能力标签) / `langdetect`(语言) / `source`。

## config / 子域（看全类别）

| config | 全量规模(约) | 说明 |
|--------|----|------|
| 0625 | ~659k | 2024-06-25 发布快照 |
| 3M | ~3M | 300 万条混合 |
| 7M | ~7M | 700 万条（最大主集） |
| 7M_core | ~1.5M | 7M 的核心精选 |
| **7M_domains** | 分域 | **按域分子目录：`code` / `math` / `commonsense` / `subjective`** |
| Gen | ~1.5M | 合成生成 |

**label 能力标签**（细粒度，可用于按能力采样）：每条带 `ability_en/zh`（如 `['logical reasoning','programming ability','mathematical reasoning']`）+ `cate_ability_en/zh`（如 `数学能力 / 编程与软件开发 / 逻辑与推理`）。

## 抽样

- source=Subjective, langdetect=en, label.ability=['逻辑推理','编程能力','数学推理']
- human: `In a certain country populated predominantly by wizards, there is an impending demonstration that needs careful planning...`（带编程/逻辑/数学的复合推理题）
- gpt: 多轮解答（conversations 格式）

## 对本项目的评估

- ✅ **双语 + 千万级 + 已带能力标签**，是阶段1/2 通用 SFT 的**量级主力**——既能补中文（zh-cn 子集）又有海量英文；`7M_domains/math` 与 `label` 里的"数学推理"标签可定向抽数学相关样本。
- ✅ `conversations` 已是多轮 chat，能力标签让我们能**按 ability 精筛/均衡**（呼应按能力切桶的课程）。
- ⚠️ **体量太大**，必须采样使用；本地只抽了样，正式用需规划取多少 config/分片 + 跨数据集去重（它聚合了很多公开集，可能与 tulutalk/dolly 等重叠）。
- ⚠️ 质量虽经 BAAI 清洗，但来源庞杂、风格不一；中文占比小（~1%），要中文主力仍靠 COIG-CQIA / dynamics。
- 🔎 用法：作通用英文 SFT 的**量补充**（高质量子集 7M_core 优先），按 label 抽"逻辑/数学/编程"相关增强推理；中文侧只取其 zh-cn 子集做补充。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：通用池 cap 15万(随机) + "末轮含 \boxed" 捞 infinity-math 3,918 条。

- 🔴 **本地下载不完整**：9 个子集各只有 shard 0(共 90 万行)，全集 7M=75 shards、7M_domains/math=15 shards 等——本地只是 1/8~1/35 抽样。扩量前先补下载。
- 🔴 **跨池格式冲突主源**：7,247 条同题既在数学池(think+boxed)又在通用池(纯对话)——通用 adapter 只挡"末轮含 boxed"的行,同题不带 boxed 的副本照进。
- 🔎 **boxed 启发式既漏又混**：label.cate_ability_en(7M/7M_core/Gen/0625 完整)比 boxed 准确得多——7M_core 数学占 46%;但 3M 和 7M_domains/math 的 label 全空,别依赖。英文数学大头直接用 7M_domains/math(本地 10 万,纯 MetaMath/Orca/MathInstruct——注意与 metamathqa 去重,源头相同)。
- ⚠️ **中文数学指望不上**：全部本地分片 zh boxed 合计仅 9 条;中文集中在 Gen(22.6% zh)/3M(21.4%),但都不是数学。
