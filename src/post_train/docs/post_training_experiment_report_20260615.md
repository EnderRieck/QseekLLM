# QseekLLM 后训练实验报告

生成日期: 2026-06-15  
覆盖范围: `src/post_train/` 下后训练数据、SFT、RL、评测、案例分析与工程日志。  
结论口径: 以已落盘文档、评测结果、val dump、case study 为准；仍在运行或只完成局部抽样的实验单独标注为“过程结果”。

## 1. 摘要

本轮后训练工作的主线是: 在自研 1.7B/1.6B Llama-like base 与 digit-split tokenizer 上，先重建可靠的 SFT/RL 数据，再通过 F2 Foundation SFT、S3-R1 难度课程 SFT、S4 退火 SFT，以及多轮 GRPO/RL 方案尝试，把基础数学、中文/英文应用题、合成算术和部分竞赛能力对齐出来。

截至 2026-06-15，最稳的可用模型档仍是 S3-R1 或 S4 SFT 档，而不是任何已完成 RL 档:

- F2 SFT 把 base 的近零可用能力显著拉起，格式率稳定到 100% 附近，并在 `cc-reserved`、`cmath`、`svamp`、`gsm8k` 上形成可用能力。
- S3-R1 相对 F2 有小幅增益，主要体现在 `gsm8k`、`gsmplus`、`cc-reserved` Pass@1；但整体提升有限。
- S4 退火相对 S3 基本横盘，六项终评都在小波动内，不能视作有效增益阶段。
- RL v1 明确退化，Pass@1/Pass@8 全线下滑，尤其 `cc-reserved` 72.9/83.9 退到 51.4/66.9。
- RL v2 修掉了 v1 的灾难性退化，但没有带来净收益: Pass@1 基本持平或局部下降，Pass@8 六项全线下降，表现为“分布变窄但没有把潜力兑现成贪心正确率”。
- v3 wrongpool、CC-only async GRPO 是正在探索的修复方向，已有过程日志，但还不足以作为最终模型结论。

能力层面，当前模型不是主要卡在输出格式，而是卡在低中阶数学核心: 量纲、变量/常量区分、等式不变量、状态跟踪、单位转换、符号操作和多步建模。继续堆高难题、长 CoT、退火或未过滤 RL，边际收益很低，并可能损伤已有能力。

## 2. 实验对象、硬件与评测口径

### 2.1 模型与 tokenizer

实验对象为 QseekLLM 自研 base:

- base checkpoint: `/data/zilu/fastrl/checkpoints/qseek_digitsplit_base`
- tokenizer: digit-split tokenizer
- 架构: Llama-like 1.7B/1.6B 量级
- 上下文: SFT 阶段使用 8k/16k，S3/S4 主打 16k

外部对照:

- `Qwen3-1.7B Base`
- `Qwen3-1.7B Instruct`

### 2.2 硬件与运行约束

后训练和评测主要运行在混合 GPU 环境:

- A800: 主要承担 actor/training。
- A4000: 承担 eval/ref/rollout。
- 必须使用 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 固定设备顺序。
- 混合 A800 + A4000 拓扑没有 NVLink，NCCL 默认 P2P 会在首次 broadcast 挂起；稳定设置是 `NCCL_P2P_DISABLE=1`。

工程日志显示，模型全参 bf16 同步约 3.4GB，实际参数同步约 4.5s/次；如果 fp32 传输则约 6.8GB，并会显著增加 A4000 OOM 风险。

### 2.3 评测集合

正式终评主表覆盖六组:

- `svamp`
- `math500`
- `cmath`
- `gsm8k`
- `gsmplus`
- `cc-reserved`

主要指标:

- Pass@1: 贪心或单样本准确率。
- Pass@8: 多采样潜力。
- format_rate: 是否满足 `<think>...</think><answer>...</answer>` 等指定输出格式。

注意事项:

- `Qwen3-1.7B Base` 不严格跟随本项目格式，format 通常较低或为 0，但 Pass@8 高，说明潜在能力强而对齐格式不匹配。
- `Qwen3-1.7B Instruct` 在部分集合上会因长输出被 2048 token 截断，尤其 `math500` 和 `gsmplus` 的可比性需要谨慎解释。
- `cc-reserved` 是本项目 compute_cot/合成基础能力最核心的保留评测集合。

## 3. 数据管线与审计修复

### 3.1 F1 数据被判定为不可继续依赖

早期 F1 Foundation SFT 后发现数据质量问题严重，F1 checkpoint 已被用户删除，只保留历史记录。后续所有可靠结论都从 v2 数据重建后的 F2 开始。

F1/旧数据的 P0/P1 问题包括:

- `orca` answer extraction bug: 约 99,268 行，占 orca 56.4%、训练总量 8.3%；大量 boxed/GT 抽到了中间数。
- `SVAMP` heldout 泄漏: 300 条 heldout 中有 63 条进入训练，占 21%。
- `numina` 中 `\boxed{proof}` 约 18,207 条，占 numina slice 20%。
- general pool 没有 eval isolation，跨池同题双格式约 7,256 条，主要来自 infinity。
- competition-math eval path 误用 full MATH 12.5k，包含训练项，导致约 129k 训练资产被过度排除。
- OpenR1 adapter 使用了错误字段，用 `solution` 而不是 R1 generations/messages + correctness，导致约 14.4% answer/boxed mismatch。
- Compute_Cot 旧数据存在重复、泄漏和渲染 bug。

这些问题决定了 F1 不能作为后续 RL 或能力分析的可信起点。

### 3.2 v2 数据重建结果

v2 rebuild 于 2026-06-10 完成，关键产物:

- SFT pool: 1,436,749
- RL pool: 1,557,905
- eval leaks removed: 4,159
- heldout v2: 3,881

F2 slice:

- 原始 1,265,064 行
- 8k 过滤 drops: 3,102
- 最终 1,261,962 行
- 组成: 约 74.7% math，25.3% general，compute_cot 31.1%

质量自检:

- 11 个 answer-bearing sources 的 gold selfcheck: 3,300/3,300 passed。
- machine-assembled sources 的 think/box 一致性: 99.8%-100%。
- R1 sources 的疑似问题多为 LaTeX 等价误判，不能直接等同于真实答案错误。

### 3.3 Compute_Cot 重建与定位

Compute_Cot 是本轮后训练最重要的基础能力资产，定位是补齐模型在算术、符号、模板化基础数学上的可执行步骤能力。

审计发现旧版 550k Compute_Cot 有严重问题:

- train unique 约 265k。
- s3 duplicate 约 73.5%。
- id_test leak 约 34.4%。
- val leak 约 29.4%。

已修复的生成器问题包括:

- `decimal_division_by_decimal`: 叙述 100% 错。
- `fraction_division`: 负号步骤缺口。
- `piecewise`: 出现 `+ -` 等脏格式。

后续 Compute_Cot 改为程序化验证，并重写了长乘、长除、小数乘除、分数运算、方程、函数、集合等基础家族，使训练样本更像“可执行解题过程”，而不是只给模板外壳。

### 3.4 数据结论

数据重建是本轮后训练中最确定有效的改动。相比继续在 F1/F旧数据上调参，v2 重建解决了更基础的问题: 答案错误、heldout 泄漏、重复、跨池污染和生成器叙述错误。F2 之后的模型能力跃迁，首先应归因于干净数据和 Compute_Cot 基础能力资产，而不是复杂训练技巧。

## 4. 训练路线与实验过程

### 4.1 阶段总览

| 阶段 | 路线 | 起点 | 数据/配置 | 产物 | 状态 |
| --- | --- | --- | --- | --- | --- |
| F1 | Foundation SFT | base | 旧数据 | checkpoint 已删除 | 废弃 |
| F2 | Foundation SFT | base | `train_sft_foundation_8k`，约 126 万，lr 1e-5，2 epoch，8k | `checkpoints/sft_foundation_v2/global_step_9858` | 已终评 |
| S3-R1 | 难度课程 SFT | F2-9858_hf | `train_sft_s3r1_16k`，约 99 万，lr 5e-6 cosine，1 epoch，16k | `checkpoints/sft_s3r1/global_step_3874` | 已终评，主要 RL 起点 |
| S4 | 退火 SFT | S3-3874_hf | `train_sft_s4_anneal_16k` | `checkpoints/sft_s4_anneal/global_step_1140_HFFIX` | 已终评 |
| RL v1 | GRPO | S3-3874_hf | async GRPO，norm_adv_by_std=True，KL 0.001，较大 shaping | `rlv1_gs50_hf` | 已终评，退化 |
| RL v2 | 修复版 GRPO | S3-3874_hf | norm_adv_by_std=False，KL 0.005，弱化 shaping，课程数据 | `v2_gs300_hf` | 已终评，轻度负向 |
| RL v3 | wrongpool + teacher replay | S3/S4 路线探索 | 全错/低正确组 gradient mask，teacher SFT 回放 | cycle1 过程日志 | 进行中 |
| CC-only async GRPO | Compute_Cot-only RL | S3 或 S4 | full CC train/val，async ref service | val dump 过程结果 | 进行中 |

### 4.2 F2 Foundation SFT

F2 是本轮可靠后训练的地基。数据由 v2 rebuild 之后的 126 万级混合 SFT 数据构成，包含 compute_cot/calc 基础算术、英文应用题、竞赛陪练、通用数据。

F2 终评:

- `svamp`: 39.3 / 73.0
- `math500`: 5.6 / 24.0
- `cmath`: 42.8 / 62.3
- `gsm8k`: 16.0 / 43.0
- `gsmplus`: 7.6 / 26.5
- `cc-reserved`: 71.8 / 84.4

F2 的意义不是达到强模型水平，而是把 base 从几乎不可用拉到稳定输出、可评测、可继续训练的状态。尤其 `cc-reserved` 说明合成基础数学与 digit-split tokenizer 的组合确实形成了局部优势。

### 4.3 S3-R1 难度课程 SFT

S3-R1 从 F2 热启，使用约 99 万条 16k 数据，难度分布约 easy 52%、medium 25%、hard 9.5%、general 13%，并提高 numina/竞赛样本权重。它被标记为主要 RL 起点。

S3-R1 终评:

- `svamp`: 41.7 / 73.3
- `math500`: 5.0 / 23.4
- `cmath`: 41.4 / 62.6
- `gsm8k`: 18.6 / 47.3
- `gsmplus`: 9.9 / 29.6
- `cc-reserved`: 72.9 / 83.9

相对 F2:

- `gsm8k` Pass@1 +2.6，Pass@8 +4.3。
- `gsmplus` Pass@1 +2.3，Pass@8 +3.1。
- `cc-reserved` Pass@1 +1.1，但 Pass@8 -0.5。
- `math500`、`cmath` 基本无增益或小幅下滑。

结论: S3 有轻微正向，尤其适合作为 RL 起点，但难度课程没有带来全局能力跃迁。

### 4.4 S4 退火 SFT

S4 从 S3 继续做退火 SFT，产物为 `global_step_1140_HFFIX`。

S4 终评:

- `svamp`: 40.7 / 74.3
- `math500`: 4.0 / 23.0
- `cmath`: 40.5 / 61.9
- `gsm8k`: 18.7 / 44.8
- `gsmplus`: 9.7 / 28.3
- `cc-reserved`: 73.3 / 84.4

相对 S3:

- `cc-reserved` Pass@1 +0.4，Pass@8 +0.5。
- `svamp` Pass@8 +1.0，但 Pass@1 -1.0。
- `gsm8k` Pass@1 基本不动，Pass@8 -2.5。
- `math500`、`cmath` 小幅下降。

结论: S4 约等于 S3，没有证明退火阶段产生了可靠收益。S4 可作为稳定 SFT 备选，但不能作为“明显优于 S3”的模型宣传。

## 5. 正式终评总表

表中格式为 Pass@1 / Pass@8。

| 模型 | svamp | math500 | cmath | gsm8k | gsmplus | cc-reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 2.0 / 6.7 | 0.4 / 1.6 | 0.0 / 0.18 | 0.4 / 2.4 | 约 0 | 约 0 |
| F2-9858 | 39.3 / 73.0 | 5.6 / 24.0 | 42.8 / 62.3 | 16.0 / 43.0 | 7.6 / 26.5 | 71.8 / 84.4 |
| S3-3874 | 41.7 / 73.3 | 5.0 / 23.4 | 41.4 / 62.6 | 18.6 / 47.3 | 9.9 / 29.6 | 72.9 / 83.9 |
| S4-1140 | 40.7 / 74.3 | 4.0 / 23.0 | 40.5 / 61.9 | 18.7 / 44.8 | 9.7 / 28.3 | 73.3 / 84.4 |
| RL v1 gs50 | 32.3 / 63.3 | 4.2 / 19.6 | 41.0 / 52.4 | 13.8 / 40.9 | 8.4 / 26.4 | 51.4 / 66.9 |
| RL v2 gs300 | 40.3 / 62.7 | 5.8 / 17.2 | 45.0 / 55.5 | 19.1 / 38.7 | 9.5 / 23.8 | 66.9 / 74.3 |
| Qwen3-1.7B Base | 28.7 / 90.7 | 52.6 / 78.6 | 34.5 / 86.0 | 48.1 / 92.0 | 34.7 / 74.1 | 50.4 / 82.1 |
| Qwen3-1.7B Instruct | 84.0 / 91.0 | 34.0 / 49.6 | 63.7 / 91.8 | 76.9 / 89.8 | 54.6 / 69.2 | 66.3 / - |

关键读法:

- 自研模型后训练能显著兑现 base 中很弱的格式和基础题能力，但整体预训练底座与 Qwen 仍有明显差距。
- 自研模型在 `cc-reserved` Pass@1 上超过 Qwen base 和 Qwen instruct，说明合成基础算术/符号任务存在局部优势。
- Qwen base 的 Pass@8 在多数集合极高，说明强底座有大量潜在正确样本，只是没有按本项目格式输出。
- Qwen instruct 的模式是把 base 的 Pass@8 潜力兑现到 Pass@1；当前自研 RL 没做到这一点。

## 6. 分阶段效果分析

### 6.1 base 到 F2: 后训练有效

base 在正式数学评测上几乎不可用:

- `svamp` 2.0 / 6.7
- `math500` 0.4 / 1.6
- `cmath` 0.0 / 0.18
- `gsm8k` 0.4 / 2.4

F2 后:

- `cc-reserved` 达到 71.8 / 84.4。
- `cmath` 达到 42.8 / 62.3。
- `svamp` 达到 39.3 / 73.0。
- `gsm8k` 达到 16.0 / 43.0。

这说明 F2 数据和 SFT 训练是有效的，且模型已学会稳定输出格式、基础算术和一部分应用题模式。

### 6.2 F2 到 S3/S4: 收益递减

S3 对 F2 的提升集中在少数集合:

- `gsm8k` 从 16.0 / 43.0 到 18.6 / 47.3。
- `gsmplus` 从 7.6 / 26.5 到 9.9 / 29.6。
- `cc-reserved` Pass@1 从 71.8 到 72.9。

但 `math500`、`cmath` 没有提升。S4 又基本没有超过 S3。

这说明在当前 base 能力与数据结构下，继续堆难度课程和退火不能自动补齐核心建模能力。S3/S4 之后的瓶颈更像是模型内部数学表征不足，而不是训练轮数不足。

### 6.3 SFT 的优势区

SFT 后模型最可靠的区域:

- 程序化生成的基础算术。
- 公式模板明确的 derivative/integral/function/property 类任务。
- 中文短应用题中的直接计算。
- 有清晰收口形式的 compute_cot 家族。

这些任务上，模型可以输出简洁、格式正确、步骤完整的答案。

### 6.4 SFT 的薄弱区

模型薄弱区:

- 真实多步应用题。
- 单位转换和量纲推理。
- 需要维护多个状态变量的题。
- 需要反向建模、分段讨论或条件枚举的题。
- `math500`/竞赛题中的符号和概念推理。

很多错误不是“没有写对格式”，而是题意理解、变量定义、等式构造、状态更新已经错了。

## 7. RL 实验分析

### 7.1 RL v1: 明确退化

RL v1 使用 S3-R1 作为起点，配置包括:

- async GRPO
- `norm_adv_by_std=True`
- KL 约 0.001
- format bonus 0.1
- think length shaping 最高约 0.2
- 数据中含大量难题/不可学组

终评显示 v1 全线退化:

| bench | S3 起点 | RL v1 | 变化 |
| --- | ---: | ---: | ---: |
| svamp | 41.7 / 73.3 | 32.3 / 63.3 | -9.4 / -10.0 |
| math500 | 5.0 / 23.4 | 4.2 / 19.6 | -0.8 / -3.8 |
| cmath | 41.4 / 62.6 | 41.0 / 52.4 | -0.4 / -10.2 |
| gsm8k | 18.6 / 47.3 | 13.8 / 40.9 | -4.8 / -6.4 |
| gsmplus | 9.9 / 29.6 | 8.4 / 26.4 | -1.5 / -3.2 |
| cc-reserved | 72.9 / 83.9 | 51.4 / 66.9 | -21.5 / -17.0 |

过程诊断也支持“能力被破坏”而不是“采样不稳定”:

- process eval total 从 28.9 降到 22.7。
- easy >90 band 从 99.6 降到 78.4。
- learnable band 从 36.2 降到 26.4。
- hard band 从 3.8 到 5.1，主要疑似随机 lucky。
- v0 -> v40 flip: right->wrong 498，wrong->right 181，破坏约为修复的 3 倍。

根因判断:

- `norm_adv_by_std=True` 在全错组会把 shaping 噪声放大成伪梯度。
- 过强 format/length shaping 让模型优化外形而不是答案。
- KL 太弱，不能保护 SFT 起点。
- 大量 p≈0 的难题组没有提供可学习信号。
- outcome-only reward 容易奖励 lucky/unfaithful 轨迹。

### 7.2 RL v2: 修掉灾难，但没有净增益

RL v2 做了关键修复:

- `norm_adv_by_std=False`
- KL 提高到 0.005
- format bonus 降到 0.05
- think length bonus 取消
- 使用 v2 curriculum pool

终评:

- `svamp`: 40.3 / 62.7
- `math500`: 5.8 / 17.2
- `cmath`: 45.0 / 55.5
- `gsm8k`: 19.1 / 38.7
- `gsmplus`: 9.5 / 23.8
- `cc-reserved`: 66.9 / 74.3

相对 S3:

| bench | S3 起点 | RL v2 | 变化 |
| --- | ---: | ---: | ---: |
| svamp | 41.7 / 73.3 | 40.3 / 62.7 | -1.4 / -10.6 |
| math500 | 5.0 / 23.4 | 5.8 / 17.2 | +0.8 / -6.2 |
| cmath | 41.4 / 62.6 | 45.0 / 55.5 | +3.6 / -7.1 |
| gsm8k | 18.6 / 47.3 | 19.1 / 38.7 | +0.5 / -8.6 |
| gsmplus | 9.9 / 29.6 | 9.5 / 23.8 | -0.4 / -5.8 |
| cc-reserved | 72.9 / 83.9 | 66.9 / 74.3 | -6.0 / -9.6 |

v2 的关键问题不是继续灾难性塌缩，而是 Pass@8 全线下降。它把分布变窄了，但没有把失去的多样性转化为 Pass@1 的确定正确。这与理想 RL 目标相反: 理想后训练应在基本保持 Pass@8 的同时提高 Pass@1。

### 7.3 RL v2 的 case-level 退化

`cc-reserved` 配对分析显示:

- paired: 4,372
- S4 wrong -> RL right: 135
- S4 right -> RL wrong: 414
- net: -279

退化集中在:

- 小数运算
- 解析几何模板
- 集合逻辑
- 方程
- 组合计数

RL 修复的少数区域包括部分数论、积分、二次方程等，但不足以抵消基础能力损失。

### 7.4 RL v3 wrongpool: 正在验证的修复方向

v3 wrongpool 的核心思想:

- 对全错组和低正确 lucky 组做分流。
- 对这些组 gradient mask，避免错误/噪声组直接影响 policy。
- 用 teacher 生成正确解法，再通过 SFT replay 回灌。
- 把 RL 用在“模型已有潜力但贪心不稳”的区域，而不是 p≈0 的区域。

过程记录:

- teacher smoke 约 5/6 correct。
- 2026-06-14 曾发生预算事故: 并发过高导致 22 分钟约 420 calls，消耗 5h bucket 和 29% weekly。
- 后续调整为 low、2 workers、60/hour。

截至当前，v3 wrongpool cycle1 只有过程 val dump，不能作为最终效果结论。已统计的 5k 抽样:

| dump | n | acc | format | score | think_len |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.jsonl | 5,000 | 29.10 | 82.18 | 0.2523 | 134.7 |
| 3289.jsonl | 5,000 | 28.62 | 84.18 | 0.2573 | 135.3 |
| 6160.jsonl | 5,000 | 27.96 | 86.70 | 0.2587 | 135.9 |
| 8811.jsonl | 5,000 | 27.28 | 87.04 | 0.2510 | 135.9 |

该结果只说明 cycle1 早期仍未展示正向趋势，不能替代正式终评。

### 7.5 CC-only async GRPO: 过程结果

2026-06-15 有两条 CC-only async GRPO 线:

1. `grpo_cconly_fullcc_async_refsvc_20260615_0339`
   - 起点: S3-R1
   - 数据: compute_cot-only train 392,553
   - val: full CC reserved 37,477
   - n=8，temp=1.0，NSTEP=300
   - 记录显示前一轮因 early eval degradation 停止

2. `grpo_cconly_fullcc_s4_temp13_async_refsvc_20260615`
   - 起点: S4-1140
   - temp=1.3，提高采样熵
   - 同样使用 full CC train/val

已统计 full CC reserved val dumps:

| 实验 | dump | n | acc | format | score | think_len |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S3 cc-only | 1084.jsonl | 37,477 | 70.65 | 99.79 | 0.7524 | 79.1 |
| S3 cc-only | 1802.jsonl | 37,477 | 70.65 | 99.79 | 0.7525 | 79.1 |
| S3 cc-only | 2442.jsonl | 37,477 | 70.51 | 99.79 | 0.7510 | 79.1 |
| S3 cc-only | 3082.jsonl | 37,477 | 70.30 | 99.79 | 0.7490 | 79.1 |
| S4 temp1.3 cc-only | 1104.jsonl | 37,477 | 70.62 | 99.78 | 0.7521 | 79.0 |
| S4 temp1.3 cc-only | 1806.jsonl | 37,477 | 70.53 | 99.79 | 0.7512 | 79.0 |

这些结果说明 CC-only RL 过程里格式稳定，但 accuracy 没有明显上行，甚至略微下降。它们应被视为“过程监控信号”，不是完整终评。后续若继续，应增加对 right->wrong / wrong->right 的配对分析，确认是否仍在损伤已有基础能力。

## 8. 与 Qwen3-1.7B 对照

### 8.1 预训练底座差距

Qwen3-1.7B Base 在多数集合 Pass@8 很高:

- `svamp`: 90.7
- `math500`: 78.6
- `cmath`: 86.0
- `gsm8k`: 92.0
- `gsmplus`: 74.1
- `cc-reserved`: 82.1

这说明 Qwen base 中已经存在大量可采样正确轨迹。Qwen instruct 的后训练主要作用是把这些潜力转成更高 Pass@1。

自研 base 的起点则明显低得多。F2/S3 能把格式和部分基础能力训练出来，但不能凭后训练补齐所有预训练阶段缺失的数学表征。

### 8.2 自研模型的局部优势

自研 SFT 模型在 `cc-reserved` 上很强:

- F2: 71.8 / 84.4
- S3: 72.9 / 83.9
- S4: 73.3 / 84.4
- Qwen base: 50.4 / 82.1
- Qwen instruct: 66.3 / -

按家族拆解，自研模型在一些合成基础数学族上超过 Qwen:

- `rational_inequality_schema`: ours 100，Qwen base 9.6，Qwen instruct 24.1。
- `absolute_value`: ours 90，Qwen base 0，Qwen instruct 25.7。
- `derivative_schema`: ours 100，Qwen base 20，Qwen instruct 61.5。
- `comparison`: ours 89.3，Qwen base 22，Qwen instruct 45.9。
- `set_logic`: ours 96.7，Qwen base 52，Qwen instruct 78.3。
- `function_property`: ours 89.8，Qwen base 40，Qwen instruct 52。

这些结果说明 Compute_Cot + digit-split 在目标化基础能力上有真实收益。

### 8.3 自研模型的全局劣势

在真实、多步、开放数学集合上，自研模型仍明显落后:

- `math500`: S3 5.0/23.4，Qwen base 52.6/78.6。
- `gsm8k`: S3 18.6/47.3，Qwen base 48.1/92.0，Qwen instruct 76.9/89.8。
- `gsmplus`: S3 9.9/29.6，Qwen base 34.7/74.1，Qwen instruct 54.6/69.2。

核心结论: 当前自研后训练能把某些窄域能力练强，但不能替代预训练底座的广义数学和语言建模能力。

## 9. 案例与能力诊断

### 9.1 正例模式

从 output showcase 和 case study 看，模型在以下场景表现稳定:

- 短算术题、整数/小数/分数计算。
- 直接求导、积分、比较、集合关系判断。
- 中文短题中运算路径直接的任务。
- Compute_Cot 家族中模板明确的题。

这些场景的输出通常具备:

- `<think>` 与 `<answer>` 格式稳定。
- 步骤较短，能收口。
- 不容易出现 Qwen thinking 模型那类长链非终止问题。

### 9.2 负例模式

典型失败包括:

- Kirill 身高题中把 “less/taller” 方向反了。
- Calc-APE 中把 `(2/5)x + 23 = 37` 错建模为 `37 / (2/5 + 23)`。
- Orca 单位换算中错误使用 `25.5 * 100`。
- 竞赛题中生成看似合理但无效的推理链，偶尔 lucky 命中答案。

错误类型:

- 题意解析错误。
- 量纲错配。
- 变量与常量混淆。
- 等式不变量破坏。
- 状态跟踪失败。
- 只套线性方程外壳，不理解运算语义。
- 写出“像在解题”的步骤，但每一步没有可靠约束。

### 9.3 格式不是主要瓶颈

F2/S3/S4 的格式率已接近稳定，CC-only val dump 也显示约 99.8% format。大量错误样本是格式正确但答案错误。因此继续加 format reward 或 length shaping 不仅收益有限，还可能像 RL v1 一样伤害实质能力。

### 9.4 当前能力画像

当前模型可以描述为:

- 会执行不少训练过的基础数学模板。
- 能在合成数据分布内保持高格式和较高正确率。
- 对真实多步应用题、单位和状态变化不稳。
- 对高难竞赛题缺乏底层表征，长推理经常只是形式链条。
- RL 暂时没有把 Pass@8 潜力兑现到 Pass@1。

## 10. RL 退化机理与修复原则

### 10.1 为什么 v1 会坏

v1 的配置把多个高风险因素叠加在一起:

- 全错组没有真实 outcome 区分，但仍被 std normalization 放大。
- format/length shaping 的 reward 可被模型轻易优化。
- 难题比例高，p≈0 的组太多。
- KL 不够强，保护不住 SFT 解题分布。
- outcome-only reward 对 lucky/unfaithful 轨迹敏感。

结果是模型学习了“更像 RL 想要的输出形状”，但破坏了 SFT 已经学到的基础解题过程。

### 10.2 为什么 v2 仍不够

v2 修掉了最危险的 std 放大和强 shaping，因此没有 v1 那样大幅崩盘。但它仍然没有解决两个核心问题:

- 对不可学组、全错组、低正确 lucky 组的处理不够彻底。
- RL 的优化目标仍不能区分“真正会了”与“采样碰巧对”。

Pass@8 全线下降说明 v2 更像在压窄输出分布，而不是把已有多样性中的正确路径蒸馏为贪心策略。

### 10.3 后续 RL 应遵守的原则

建议后续 RL 只在以下条件下继续:

- 从 S3-R1 或经过严格验证的 SFT 档开始，不默认使用退火档。
- 训练池按 Pass@8/Pass@1 分层，只选“有潜力但不稳”的样本做 policy RL。
- 全错组、低正确 lucky 组进入 wrongpool，不直接给 policy 梯度。
- wrongpool 由 teacher 或 verifier 产生可信解法，再做 SFT replay。
- 每轮必须做 paired flip analysis，监控 right->wrong 是否超过 wrong->right。
- 保持 Pass@8 不掉作为硬约束；若 Pass@8 连续下降，应停止该 RL 线。
- format reward 只保底，不作为主要 shaping。

## 11. 工程实验与效率结论

### 11.1 同步 GRPO 性能瓶颈

早期同步 GRPO smoke 的单步约 161.2s:

- generation: 100.3s，约 62%
- actor update: 30.7s，约 19%
- reward: 12.4s
- old_logprob: 10.1s
- ref: 7.6s
- reshard: 1.6s

采样是主要瓶颈。

纯 A4000 decode benchmark:

- 1 card: 512 seq，65.2s，约 3473 tok/s
- 2 cards: 33.3s，约 2x
- 4 cards: 22.7s，约 2.9x
- 7 cards: 19.5s，约 3.3x

结论: 多卡 rollout 有收益，但超过 4 卡后受 batch、同步和系统开销限制，需要更大 global batch 才能继续饱和。

### 11.2 split GRPO 与 rollout 数量

P2P-off split GRPO 测试:

| rollout 数 | step(s) | gen(s) | ref(s) | update_actor(s) | update_weights(s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 137.54 | 74.55 | 26.73 | 29.90 | 3.41 |
| 2 | 109.68 | 44.65 | 27.39 | 30.54 | 4.45 |
| 3 | 96.67 | 30.55 | 27.29 | 30.60 | 5.43 |
| 4 | 92.65 | 25.14 | 27.48 | 30.85 | 6.55 |

4 rollout 最快，但显存最紧；3 rollout 是更稳的成本/吞吐折中。

### 11.3 NCCL mixed GPU 问题

A800 + A4000 混合拓扑下，默认 NCCL P2P 会在首次 broadcast 卡住。最小复现:

- `nccl_broadcast_smoke.py`
- world=3
- 1MB broadcast 默认挂起
- 设置 `NCCL_P2P_DISABLE=1` 后通过
- 512MB broadcast 约 0.115s

所有 GRPO runner 应默认:

- 禁用 NCCL P2P。
- 重建通信 group。
- 使用 bf16 参数传输。

### 11.4 async/ref-service 架构

后续引入 split resource pool:

- actor: GPU1 A800
- ref: GPU3 A4000
- rollout: GPU4-7 A4000

新增 RefService:

- ref logprob 在独立 GPU microbatch 计算。
- trainer 从 ready_queue 取结果。
- trainer step 内 `timing_s/ref=0`。

还加入:

- actor update `asyncio.to_thread`
- rollout vLLM pause/resume eval
- EventLogger JSONL
- val dumps

这套工程改造使 RL 实验从“能跑”进入“能监控、能定位、能高频评估”的状态。

### 11.5 效率 bench

效率文档给出的最终 narrative:

- serial: 90.0s/step
- upstream/verl original fully async: 43.1s/step，约 2.09x
- our split async: 37.2s/step，约 2.42x

`metrics.json` 中还保留了 split-sync-ref 等中间 variant:

- `serial`: 90.0
- `verlsplit_sync_ref`: 61.0
- `ourasync`: 37.2

结论: 工程侧的 async/ref-service 改造是有效的；当前问题主要不是训练系统速度，而是 RL 目标和数据分层没有稳定带来模型收益。

## 12. 主要结论

### 12.1 已经成立的结论

1. v2 数据重建是必要且有效的。旧 F1 数据存在答案错误、泄漏、重复和生成器 bug，不能作为可信实验基础。
2. F2 SFT 是本轮最大收益阶段，把 base 从近零能力拉到可用能力。
3. S3-R1 有小幅收益，并且是当前更合理的 RL 起点。
4. S4 退火没有明显终评增益。
5. RL v1 明确退化，不能使用。
6. RL v2 修掉了 v1 的灾难，但没有正收益，尤其损失 Pass@8。
7. 当前错误主要是实质推理/建模错误，不是格式错误。
8. Compute_Cot 在目标基础能力上有效，并带来相对 Qwen 的局部优势。
9. 与 Qwen 的全局差距主要来自预训练底座和广义数学表征，不是简单后训练可完全补齐。
10. async/ref-service 工程路线有效，但模型效果瓶颈在 RL 信号设计和数据选择。

### 12.2 不应过度解读的部分

- S4 不能宣传为显著优于 S3。
- RL v2 不能宣传为成功 RL，只能说“止住 v1 退化但未带来净收益”。
- CC-only async 的 70% 左右 val acc 只是过程结果，并未显示持续提升。
- v3 wrongpool 当前还没有最终效果，只能作为合理方向和进行中实验。
- `cc-reserved` 的强表现不能代表真实多步数学能力已经解决。

## 13. 后续建议

### 13.1 模型选择建议

短期可用模型:

- 若要稳妥: 使用 S3-R1 `global_step_3874`。
- 若任务偏 Compute_Cot/基础合成题: S4 `global_step_1140_HFFIX` 可作为备选，因为 `cc-reserved` Pass@1 略高。
- 不建议使用 RL v1。
- 不建议默认使用 RL v2，除非具体任务偏向其少数提升区域，并通过任务内评测确认。

### 13.2 数据与训练建议

下一阶段不应继续简单堆难题。优先级应是:

1. 建一个 L1-L3 数学核心修复集:
   - 量纲
   - 单位转换
   - 变量/常量区分
   - 等式不变量
   - 状态更新
   - 比例/分数/小数互转
   - 简单方程建模
   - 反向条件题

2. 对真实错例做 minimal-pair 数据:
   - 同题不同措辞。
   - 正向/反向条件。
   - 多余信息/缺失信息。
   - 单位陷阱。
   - 变量命名干扰。

3. 对 Compute_Cot 扩展要继续程序化验证，禁止回到不可验证模板生成。

4. SFT replay 应覆盖 RL 期间 right->wrong 的基础能力族，特别是小数、集合、方程、解析几何模板。

### 13.3 RL 建议

继续 RL 前应先建立硬约束:

- 每个 checkpoint 都跑小型 paired eval，统计 right->wrong / wrong->right。
- Pass@8 不能持续下降。
- 全错组进入 wrongpool，不能直接 policy update。
- 低正确 lucky 组由 teacher/verifier 生成可学轨迹，再 replay。
- reward 以答案正确为主，format 仅保底。
- KL 不应低于 v2 的保护强度，且要监控 easy band 是否被破坏。

更合理的 RL 目标不是“让模型学会不会的难题”，而是“把已有 Pass@8 潜力稳定兑现为 Pass@1，同时不破坏基础族”。

### 13.4 评测建议

每轮训练至少保留三类评测:

- 正式六项终评: `svamp/math500/cmath/gsm8k/gsmplus/cc-reserved`。
- CC family breakdown: 监控基础能力族是否被 RL 破坏。
- 真实错例 suite: 专门覆盖量纲、建模、单位、变量状态跟踪。

对进行中 RL，建议每 1-2 个 eval dump 做:

- full val acc/format/score。
- family-level breakdown。
- paired flip matrix。
- 按 old correctness band 拆分: easy、learnable、hard、all-wrong。

## 14. 证据索引

核心文档:

- `src/post_train/docs/final_eval_all_models.md`
- `src/post_train/docs/eval_tracking.md`
- `src/post_train/docs/model_capability_analysis_20260612.md`
- `src/post_train/docs/model_output_showcase_20260614.md`
- `src/post_train/docs/data_audit_report_20260610.md`
- `src/post_train/docs/rebuild_v2_log.md`
- `src/post_train/docs/format_audit.md`
- `src/post_train/docs/s4_data_report_20260612.md`
- `src/post_train/docs/stage3_data_audit_and_plan.md`
- `src/post_train/docs/data_audit_and_architecture.md`

RL 与工程文档:

- `src/post_train/docs/rl_degradation_diagnosis_20260613.md`
- `src/post_train/docs/reward_verifier_fix_20260612.md`
- `src/post_train/docs/rl_wrongpool_sft_experiment_20260614.md`
- `src/post_train/docs/grpo_cconly_fullcc_async_20260615.md`
- `src/post_train/docs/grpo_cconly_fullcc_s4_temp13_20260615.md`
- `src/post_train/docs/rl_logging_and_process_eval_20260613.md`
- `src/post_train/docs/rl_async_grpo_architecture.md`
- `src/post_train/docs/rl_fully_async_split_ref3_20260613.md`
- `src/post_train/docs/rl_grpo_sync_a800_ref3_timing_20260613.md`
- `src/post_train/docs/rl_grpo_sync_issue_log_20260613.md`
- `src/post_train/docs/rl_nccl_weightsync_deadlock_20260613.md`
- `src/post_train/docs/nccl_mixed_gpu_p2p_hang.md`
- `src/post_train/docs/rl_async_env_upgrade_20260613.md`
- `src/post_train/docs/rl_sampling_bench_20260613.md`
- `src/post_train/docs/rl_grpo_smoke_timing_20260613.md`
- `src/post_train/docs/hybrid_sft_rl_design.md`
- `src/post_train/docs/effbench/efficiency_analysis.md`
- `src/post_train/docs/effbench/metrics.json`

Benchmark 与案例:

- `src/post_train/docs/benchmark/model_benchmark_comparison.md`
- `src/post_train/docs/benchmark/model_benchmark_scores.csv`
- `src/post_train/docs/benchmark/cc_reserved_breakdown.md`
- `src/post_train/docs/benchmark/cc_reserved_rl_case_study.md`
- `src/post_train/docs/benchmark/cc_reserved_qwenred_cases.md`
- `src/post_train/docs/benchmark/qwen_wrong_ours_right_cc_reserved_case_study.md`
- `src/post_train/docs/benchmark/cc_reserved_family_breakdown.csv`
- `src/post_train/docs/benchmark/cc_reserved_rl_flip_cases.csv`

过程日志与 val dumps:

- `src/post_train/logs/grpo_cconly_fullcc_async_refsvc_20260615_0339_rollout4_steps300_val_dumps/`
- `src/post_train/logs/grpo_cconly_fullcc_s4_temp13_async_refsvc_20260615_rollout4_steps300_val_dumps/`
- `src/post_train/logs/grpo_v3wrongpool_cycle1_rollout4_steps75_val_dumps/`

## 15. 最终判断

这轮后训练最有价值的成果是: 数据审计和 v2 重建把实验地基救回来，F2/S3 SFT 把自研 base 拉到了可用的基础数学模型，Compute_Cot 证明了 targeted synthetic data 对基础能力有效；同时，RL 实验明确暴露了当前方案的问题，避免继续在错误方向上消耗算力。

下一步最值得做的不是继续扩大 RL 或堆更难数据，而是围绕真实错误建立“基础数学核心修复集 + wrongpool teacher replay + 严格 paired eval”的闭环。目标应从“让模型挑战更难题”改为“先让模型在会做的基础题上不退化，并把 Pass@8 中已有的正确轨迹稳定变成 Pass@1”。
