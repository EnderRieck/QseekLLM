# 数据卡片 · Open-Orca/FLAN

> 定位：**超大规模英文学术 NLP 任务合集**（FLAN Collection）；偏"任务泛化"，非对话。
> 本地：`/data/zilu/general_sft_raw/flan`（**采样 14 个 submix 各 1 分片**，1.8G；全量 100M–1B 级、TB 体量）

## 总体说明

- **来源**：Google FLAN Collection 的社区重打包（Open-Orca），把 5 大任务混合 × 提示风格组合成 14 个 submix。
- **规模**：**全量 1 亿~10 亿级**（2157 分片）。本地每个 submix 只取 `part.0.parquet`（够看清结构+抽样）。
- **语言**：英文。
- **License**：见 repo（FLAN 系，研究用途）。
- **格式**：`inputs`(任务输入文本) / `targets`(目标输出) / `_task_source` / `_task_name` / `_template_type` / `_template_idx`。**纯 input→target 文本对，非 chat**。

## 14 个 submix（= 5 任务源 × 提示风格）

任务源 `_task_source`：**CoT / FLAN(2021) / T0 / NIV2(Natural-Instructions v2) / dialog**。
风格后缀：`fs`=few-shot / `zs`=zero-shot；`opt`=带选项 / `noopt`=不带选项。

| submix | 含义 |
|--------|------|
| cot_fsopt / cot_zsopt | 思维链任务（少/零样本） |
| dialog_fsopt / dialog_zsopt | 对话类任务 |
| flan_fsopt / flan_fsnoopt / flan_zsopt / flan_zsnoopt | FLAN 2021 任务集 |
| niv2_fsopt / niv2_zsopt | Natural Instructions v2（1600+ 任务） |
| t0_fsopt / t0_fsnoopt / t0_zsopt / t0_zsnoopt | T0 任务集 |

每个 submix 内 `_task_name` 有**成百上千个具体 NLP 任务**（如 `cot_esnli_ii`、各类 QA/NLI/摘要/翻译…）。

## 抽样

- `_task_source=CoT, _task_name=cot_esnli_ii, _template_type=fs_opt`
- inputs: `The man be showing his toys to adults... So what could be the question? Question followed by answer: If "A man we..."`
- targets: `Premise: "Woman skates in possession of puck." Based on this premise, can we conclude the hypothesis...`

## 对本项目的评估

- ⚠️ **与本阶段目标关系最弱**：FLAN 是"海量学术 NLP 任务泛化"语料，格式是 input→target 短文本、偏模板化，**不是对话、不是数学**；直接混入对"基本算术+数学方法"帮助有限，还可能引入生硬模板腔。
- ✅ 价值在于**任务多样性与指令泛化**：`cot_*` 子集含思维链、`niv2/t0` 覆盖极广的任务类型，少量混入可增强"听懂各种任务指令"的能力（防止只会数学）。
- ⚠️ **体量极大且质量参差**（社区重打包，部分 input/target 对略乱）；若用务必**只取小比例、优先 cot_* / zsopt**，并严格控量。
- 🔎 本阶段建议**最多极小比例混入 cot 子集**做指令泛化，其余暂不用。是 7 个通用集里**优先级最低**的。

## 审计补充（2026-06-10 复核）

**实际使用现状**：已下载 3.5G 但 adapter 从未注册——整库未接(计划"cot_zsopt 极小比例"未实现)。

- cot_zsopt 95,570 行:cot_gsm8k 9,577 + stream_aqua 3,528 是英文数学短 CoT(1-3 句,风格精简);cot_esnli/ecqa/strategyqa ~70k 英文推理。
- niv2 5M 行大杂烩,数学/算术任务约 2.2%(全量估 ~11万,addsub/数字操作与 Compute_Cot 目标同构);dialog 2.7M 合成 wiki 对话,机械,不建议。
- 建议:仅选择性接 cot_gsm8k/aqua(去污染后)+少量推理调味;fsopt/dialog/niv2 主体不接。
