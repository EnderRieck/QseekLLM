# 通用 SFT 数据集总览（阶段1 初步 SFT + 阶段2 防语言退化）

> 7 个数据集的信息卡片索引。下载脚本 `scripts/download_general_sft.py`（镜像直连为主）；
> 原始数据在 `/data/zilu/general_sft_raw/`。每张卡片含总体说明 + 全子类 + 抽样 + 可用性评估。

## 一览

| 数据集 | 语言 | 规模(本地) | 子类维度 | License | 质量 | 卡片 |
|--------|------|-----------|---------|---------|------|------|
| [no_robots](no_robots.md) | EN | 10k 全量 | category×10 | CC-BY-NC(非商用) | ⭐⭐⭐⭐⭐ 人工精写 | 精料锚 |
| [dolly-15k](dolly-15k.md) | EN | 15k 全量 | category×8 | CC-BY-SA(**可商用**) | ⭐⭐⭐ 人工但参差 | 商用量补 |
| [tulutalk-annotated](tulutalk-annotated.md) | EN | ~80万 全量 | task_category×10 + 难度/奖励/安全标注 | 见repo | ⭐⭐⭐ 合成+重标注 | 英文多轮主力 |
| [infinity-instruct](infinity-instruct.md) | EN+ZH | ~90万 采样(全量7M+) | config×6 + ability标签 | 见repo | ⭐⭐⭐⭐ BAAI清洗 | 双语量级主力 |
| [coig-cqia](coig-cqia.md) | ZH | 44.7k 全量 | 来源×13 | 见repo | ⭐⭐⭐⭐ 人工校验 | **中文主力** |
| [dynamics-instruction-tuning](dynamics-instruction-tuning.md) | ZH | ~82k 全量 | 能力×10 | 见repo | ⭐⭐⭐ 整齐 | 中文能力维度 |
| [flan](flan.md) | EN | 1.8G 采样(全量TB) | submix×14 | 见repo | ⭐⭐ 模板化参差 | 指令泛化(少量) |

## 配比建议（initial，留给 eval 驱动课程动态调）

- **英文指令侧**：以 `no_robots`(高权重精料) + `dolly`(商用量) 为质量锚，`tulutalk`(多轮，按 reward/难度筛) + `infinity-instruct`(量) 为主体。四者**题材有重叠，务必去重**。
- **中文侧**（防中文退化的关键）：`coig-cqia` + `dynamics` 为主力；`infinity-instruct` 的 zh-cn 子集补充。中文占比要足够（建议中英不要太悬殊，具体比例后续按评测定）。
- **指令泛化**：`flan` 只取小比例 `cot_*` 子集，防止"只会数学/对话不会通用任务"；优先级最低。
- **与推理协同**：`tulutalk` 的 Math/Reasoning/Coding、`coig-cqia` 的 logi_qa/ruozhiba、`infinity` 的"逻辑/数学推理"标签样本，可适度上采样以助"逻辑思维"目标。

## 类目均衡提醒

各集内部类目都极不均衡（no_robots 的 Generation 46%、coig 的 finance+wiki 近半、tulutalk 的 smol-magpie 占半）。按 category/source **均衡或上采样小类**，避免主体被单一风格主导。

## 优先级小结
中文：coig-cqia ≈ dynamics > infinity(zh) ｜ 英文：no_robots > dolly ≈ tulutalk > infinity(en) > flan

---

## ⚠️ 2026-06-10 全面审计补充（各卡片已附"审计补充"节,总报告见 `docs/data_audit_report_20260610.md`）

- **被浪费的质量信号 Top3**:chinese-r1 的 `reasoning_content`(11万 R1 think 全闲置,其中 3.4万验证过的中文数学) / tulutalk 的 st_instruct_reward+difficulty+task_category(随机抽 15万=放弃 80万行标注) / infinity 的 label.cate_ability(比 boxed 启发式准)。
- **结构问题**:build_general 无评测隔离(已实测漏 5 条评测题) / 与数学池 seen 独立 → 7,256 条同题双格式冲突(infinity 为主) / infinity 本地只下了各子集 shard 0。
- **新增卡片**:[chinese-deepseek-r1-distill](chinese-deepseek-r1-distill.md)(此前缺卡)。
- dynamics 的 reasoning_full(1.27万"选择题→裸字母")是 MCQ 空区病征的训练侧来源,建议剔除或重造。
