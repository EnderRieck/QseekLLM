# 数据卡片 · aladinDJ/tulutalk-annotated

> 定位：**重标注的英文多轮指令对话**（自带 difficulty/quality/task_category/reward/safety 标注，便于精筛配比）。
> 本地：`/data/zilu/general_sft_raw/tulutalk-annotated`（全量 12 分片 ≈ **80 万条**）

## 总体说明

- **来源**：在多个开源 SFT 集（smol-magpie-ultra / apigen / openhermes / systemchats…）之上，用 **Llama-3.3-70B** 批量打标注（难度/质量/类别/奖励/安全），整理成统一多轮对话。
- **规模**：每片 67,361 × 12 ≈ **808k**。
- **语言**：**英文为主**（EN 占 ~99%，余 LA/DE/FR 等零星）。
- **格式**：`messages`(多轮 chat) + `instruction`/`response` + 海量标注列：`task_category` / `source` / `difficulty` / `input_quality` / `intent` / `knowledge` / `mt_instruct_reward`(多轮奖励分) / `st_instruct_reward` / `llama_guard_2`(安全) / `Turn` / `language`。

## 关键标注分布（看全类别）

**task_category（10+ 类）**：Information seeking 11.6k / Math 10.8k / Coding & Debugging 8.3k / Editing 7k / Brainstorming 5.8k / Advice seeking 4.8k / Data analysis 4.5k / Creative writing 4.4k / Role playing 3.5k / Reasoning 3.3k（每片计）。
**source**：smol-magpie-ultra 43.8k / apigen-80k 7.4k / smol-summarize 6.5k / smol-rewrite 4.3k / systemchats-30k 2.5k / openhermes-100k 2.2k …
**difficulty**：hard 28.4k / medium 22.4k / easy 15k / very easy 1.3k / very hard 391。
**Turn**：**全部 Multi-Turn**（多轮）。 **input_quality**：全部 `excellent`（已预筛）。

## 抽样

- source=smol-summarize, task_category=Planning, difficulty=easy
- system: `Extract and present the main key point of the input text in one very short sentence...`
- user: `Hi Michael, I wanted to follow up on our conversation about collaborating on decimal operation worksheets...`
- intent(标注): `The user wants to schedule a meeting with Michael to discuss collaborating...`
- 奖励/安全标注: mt_instruct_reward=5, llama_guard_2=safe

## 对本项目的评估

- ✅ **多轮 + 已打满标注**，是阶段1"指令跟随 + 多轮对话格式"的好料；最大价值是**标注让我们能精准配比/过滤**（按 task_category 均衡、按 difficulty 做课程、按 reward/safety 筛优）。
- ✅ 含 Math/Reasoning/Coding 子类，与数学训练有协同；apigen 源含函数调用（工具使用）。
- ⚠️ **纯英文**；底层 source 多为 LLM 合成（magpie/hermes），非人工，质量虽过滤但不及 no_robots 手写。
- ⚠️ 量大但同源重复风险（smol-magpie 占一半），配比时按 source 去重/降权。
- 🔎 用法：当英文多轮主力之一，**用 reward≥阈值 + 按 task_category 均衡**抽样；与 no_robots(精)、dolly(商用) 形成英文三件套。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：入池 cap 15万(纯随机)，foundation 再随机 cap 8万在训 —— **全部标注被浪费**(计划"按 reward 筛、类目均衡"未实现)。

- 全量 808,322 行 = Tulu-3-mixture 子样本(~304k,带 tulu_id 回链)+ SmolTalk 子样本(~504k) 的 Llama-3.3-70B 重标注版。语言 98.4% 英文(ZH 仅 0.2%)。
- **可用质量信号**：`st_instruct_reward`(单轮连续分,median 1.99,<0.75 约 10% 可滤) > `difficulty`(5 档) > `llama_guard_2`(unsafe 1.9% 应剔) ；`mt_instruct_reward` 几乎全 5 分无区分度,`input_quality` 98.3% excellent 无用。
- **task_category 失衡**：Math 36.6% + Coding 19.4% 占半壁——纯随机抽 15万,通用池其实混进了大量英文数学/代码(这解释了通用池里 3 条 MATH 泄漏题的来路)。
- 🔎 **数学子集可单独捞**：task_category=Math 共 296,223 条、自带 difficulty 5 档(hard 132k/easy 85k/medium 51k/very hard 21k)——阶段3 英文课程素材;捞取需按 source 白名单(剔 apigen tool-call 约 5.5k 假 Math)。
- **建议过滤器**：safe ∧ (单轮 st_reward≥0.75 ∨ 多轮 mt_reward=5),再按 task_category 配比(压 Math+Code,保 Creative/Chat/Brainstorm 防退化本职)。
