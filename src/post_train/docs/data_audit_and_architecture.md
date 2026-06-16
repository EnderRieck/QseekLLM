# 数据构建架构 + 审计发现（项目总结）

> 2026-06-09 对 `data_unified` 数据管线与 step 500/1000 eval 的深度审计沉淀。
> 关联：[training_plan.md](training_plan.md)（四阶段计划）、[eval_tracking.md](eval_tracking.md)（eval 指标演化）、[hybrid_sft_rl_design.md](hybrid_sft_rl_design.md)（SFT+RL 混合设计）。
> 代码：`data_pipeline/{adapters,build,reweight_sft,to_verl_parquet}.py`。

---

## 一、数据构建架构：「统一格式 → 两个全量弹药库 → 阶段课程子集」

```
原始 17 源
  │  ① adapters.py  每源一个适配器 → 统一 Record
  │     · 数学→ <think>worked</think> #### \boxed{答案}；通用→纯对话
  │     · 标签 ability(math/general)/source/difficulty
  │     · use 分流：验证过答案→both / 蒸馏未验证→sft / 只有答案→rl
  │     · 轻过滤：无题面、既无解又无答案 → 丢
  ▼
  ② build.py  建池（真过滤）
  │     · 评测泄漏剔除：qhash(题面)∈eval_hashes → 删
  │     · 全局去重：qhash 重复 → 删（先到先得，按 SOURCE_ORDER）
  │     · qhash = md5(题面去空白+小写)
  ├──────────────┬──────────────
  ▼              ▼
 train_sft.jsonl  train_rl.jsonl          ◀ 两个全量弹药库（建好不动）
 (194万, 有gold)  (160万, 有可验证答案)
  │  ③ reweight_sft.py  按阶段课程切子集（下采样, 非删除）
  │     · 地基(compute_cot/orca/gsm8k) cap=None 全留
  │     · 竞赛难题(numina 505k→90k / openr1 190k→50k) 随机降权"陪练"
  │     · 通用源限量 ~22% 防退化
  ▼  ④ 长度过滤 ≤8192 token
 train_sft_foundation_8k.parquet (118.9万)  ◀ 阶段1-2 当前在训
```

**核心认知**：不是「挑一批数据训完拉倒」，而是**建两个全量弹药库，每阶段按该阶段课程配方现切子集**。当前 `foundation_8k` 只是**阶段1-2 的第一轮配比子集**；阶段3 换 cap 表（调高难题比例）从同一弹药库重切，数据不变、变配比。

**减量两种性质**：
| 环节 | 性质 | 可逆 |
|---|---|---|
| 泄漏剔除 / 全局去重 / 超长过滤 | **真过滤**（该删） | 否，弹药库里就没了 |
| per-source cap 随机抽（reweight）| **下采样**（配比） | 是，全量在弹药库，下阶段能再多采 |

一道题可同时进 SFT 和 RL（`use=both`）——两池有交集，非切分。

---

## 二、难度分级现状：基本是「来源即难度」的常量先验

| 源 | difficulty 怎么来 | 值 |
|---|---|---|
| **compute_cot** | ✅ 真·逐题（生成器 metadata）| easy/medium/hard |
| **big-math** | ✅ 真·逐题（llama8b_solve_rate）| solve_rate=0.xx（仅RL）|
| metamath | 半真（type 前缀）| GSM*→easy，余→hard |
| orca/gsm8k/calc-ape | 🔸 一刀切 | easy |
| numina/openr1/deepscaler/openthoughts3/bespoke | 🔸 一刀切默认 | **hard** |
| infinity-math/dapo | 空 | "" |

- **大部分按数据集来源贴常量**，与题目实际难度无关；**仅 compute_cot、big-math 有真逐题难度**。
- 跨源不可比（compute_cot 的 easy ≠ orca 的 easy）。
- reweight 的「易重难轻」**约等于「按来源配比」**，非真实难度。
- 设计文档定位：静态标签仅用于「铺谱/初始分层」；**真正动态课程要靠训练中 eval 测「模型自评 solve_rate」**——现状标签撑不起可靠课程信号，阶段2 接动态课程前需先解决。

---

## 三、数学 SFT 各数据集（842,117 条，foundation_8k 版）

| 数据集 | 条数 | 内容 | 答案/风格 |
|---|---|---|---|
| compute_cot | 393,954 | 自产基础算术+方法(268子类) | 短 worked，骨干 |
| orca-math | 176,899 | 英文应用题 | 设变量列方程(45%代数) |
| metamathqa | 110,000 | 英文，大量「求未知变量x」 | 代数解方程(46%) |
| numinamath-1.5 | 89,995 | 竞赛题 | 表达式/整数/选择6%/proof |
| openr1-math | 50,000 | 竞赛/R1 trace | 干净逐步，选择16% |
| gsm8k | 7,473 | 小学应用题 | ✅ 直接逐步算 |
| deepscaler | 5,750 | 难题(对数/微积分) | 中长 |
| infinity-math | 3,918 | 教学式详解 | 偏长 |
| bespoke-stratos | 3,057 | ⚠️ 跨学科 R1(非纯数学) | 超长~6k字，选择73% |
| openthoughts3 | 1,071 | 数学，过度自纠 | ⚠️ 超长~1.4万字 |

全部英文（中文数学不在此桶）；统一 think+boxed 格式。

**numina 子集（原始896k，3维标签）**：source(cn_k12 30%/olympiads 22%/orca_math 17%/synthetic 16%/aops 8%…)；problem_type(Algebra47%/Geometry20%…，我们用它当 source 标签)；question_type(word-problem70%/MCQ16%/proof12%)。采样=均匀随机抽90k(非按子集分层)，保留原比例。

---

## 四、审计发现（按重要性）

### 🔴 orca boxed 答案错误是真 bug（2026-06-10 复核推翻早前"已澄清"）
**早前的"97.9% 一致、已澄清"结论作废——那是循环验证**：`adapt_orca` 里 `ground_truth` 和 SFT-boxed 用的是**同一次 `_num_from_text(sol)` 的同一个返回值**（adapters.py:102-107），拿 RL 池 ground_truth 对 SFT boxed 等于自己对自己，必然高一致。

**真问题**在 `_num_from_text` 的回退正则（adapters.py:76）：`re.search` 在解答**末 200 字符里取第一个 `=`/"answer is" 后的数**，而 orca 解答末尾常是连串中间等式 → 抓到**中间结果**而非最终答案。

**2026-06-10 严格口径复测**（boxed 数值是否出现在解答结论句中，`train_sft_foundation_8k.parquet` 实测）：
- orca 176,098 条可判定中 **99,268 条（56.4%）boxed 数值在结论句中完全不出现** = 硬伤下界，**占整个 foundation 训练集 8.3%**。
- 随机人检均为实锤：结论 8723→box 14696（题面数）、结论 66→box 8、结论 0.5→box 765.0000000000001、结论 7.5→box 0.8、结论 75.42→box 19007……
- 对照组同口径：metamathqa 不一致仅 0.7%（其 response 末尾是规范的 "The answer is: X"，regex 命中正确）→ bug 仅伤 orca（唯一走 `_num_from_text` 回退的源）。

**影响**：① SFT 教模型"推理一套、boxed 另抄一个数"（与 eval 看到的"经典陷阱"/boxed 脱节行为吻合）；② **RL 池 orca 的 ground_truth 同样错误** → 若进 RL 会奖励错答案，必须修。
**修法**：`_num_from_text` 回退改为取**最后一个**匹配（或优先取结论句"So/Therefore…is X"模式），重跑 orca 适配 → 重建弹药库受影响行 + 阶段2/3 配方重切时生效。当前在训 run 是否值得为此重启，单独决策（orca 占 14.9%，其中 ~56% 坏 ≈ 全集 8.3%）。

### ⚠️ 应用题解法「风格冲突」（值得盯，先不动）
模型对简单应用题套变量列方程而非直接算（bakery 题 `2/3x=1/2(x+1)` 列错又算崩）。溯源：
- 「直接逐步算」：gsm8k(96%直接) + compute_cot — 干净但量小(gsm8k仅7.5k)。
- 「设变量列方程」：**orca 45% + metamath 46% ≈ 13万条主导**。
- 判定：风格过度泛化，非数据错。触发式应对：step 2000-3000 若 compute_cot↑ 而 gsm8k/cmath 卡住，再给 metamath「求x」题降权 / 提 gsm8k 直接风格占比。

### ⚠️ 选择题训练「空区」
MCQ 训练分两档不连通：①简单/知识 MCQ ~2.6万(dynamics/coig/chinese-r1)是**通用prompt裸字母**作答(无think/box)；②带 `\boxed{字母}` 的 ~2万(numina/openr1/bespoke)**全是竞赛难题**。**缺「简单MCQ→简短CoT→单\boxed{字母}」这一档**。cmmlu eval 用数学prompt要boxed→落空区→枚举ABCD+空boxed。若要救 MCQ 需补简单 MCQ 的 boxed-CoT 样本（非泛泛单选数据）。

### ⚠️ 中文数学几乎缺席（真能力缺口）
数学 SFT 桶 ~100% 英文。chinese-r1(8万中文)被打 general 标签、calc-ape 仅RL；cn_k12 在 numina 里是英文文本。→ 模型靠跨语言迁移做 cmath/gaokao，解释其低分。要补中文数学需往数学桶加「中文题面+中文解+think+box」。

### ⚠️ cn_k12 被误标 hard + 双重错配
numina 原始无 difficulty 字段 → 管线一刀切全标 **hard**。cn_k12(中小学，实际偏易)被当竞赛难题降权砍掉。叠加「英文文本」→ 它既没补中文、又没作为易题地基留下。

### ⚠️ proof / 超长源（设计选择，下一版可清）
- numina proof 题 ~1.4万(question_type=proof) → 管线**有意**路由 SFT-only(`ans∈("","proof")→use=sft`)，`\boxed{proof}` 是副产物。保留②推理素材，但可能助长「难题吐空/占位boxed」。
- bespoke(跨学科R1) + openthoughts3(超长1.4万字)：与「基础阶段简短worked」冲突，量小，下一版可剔。

---

## 五、eval 基础设施（已重构）

`eval/async_eval.py`（2026-06-09 重写）：
- **vLLM 后端**（默认，比 HF generate ~10x；`.pt`→HF 转换后加载；`--backend hf` 兜底）。
- **多卡数据并行**（held-out 按长度均衡切片，每空闲 A4000 一实例；`--gpu-candidates 2,3 --max-gpus N`）。
- **贪心 Pass@1**（temperature=0，确定性可比、对GPU非确定性鲁棒；Pass@k 采样留阶段3/4）。
- **实时进度** + **增量落盘**（每块 flush，崩溃保结果；vLLM worker `os._exit` 避免引擎析构 hang）。
- 踩坑：原 batch_size=64 对16G A4000 必OOM(KV 1.75MB/token/seq) → 动态token预算分批。
- 产出（`<ckpt>/eval_dumps/`）：`metrics.jsonl` / `step_N/heldout.jsonl`(完整IO) / `step_N/heldout.md`(可读,`eval/dump_to_md.py`生成) / `tb/`。

权重位置：全部 `/data/zilu/fastrl/checkpoints/`（base=qseek_digitsplit_base；ckpt=sft_foundation/global_step_N，每个~21G）。

---

## 六、下一版数据修订 TODO

> **⚠️ 2026-06-10 全面审计后本节已被取代**：完整问题清单/未采纳资产盘点/修复行动清单见 **[data_audit_report_20260610.md](data_audit_report_20260610.md)**（P0:orca 抽取/svamp 泄漏/通用池隔离/numina valid 过滤;P1:中文数学池组建·openr1 换 R1 字段·MATH-train 解放;各数据集卡片均已附"审计补充"节）。下列旧条目仍有效的已并入该报告。

- [ ] **🔴 最高优先：修 orca 答案抽取 bug**（见四·第一条；`_num_from_text` 改取最后匹配，重抽 orca 的 ground_truth+boxed，重建受影响行——这条**动弹药库**，是真数据错不是配方问题；RL 池同步修，否则 RL 奖励错答案）。
- [ ] **中文数学补进数学桶**（calc-ape题面+造解 / chinese-r1 数学子集重打 think+box 格式）——最要紧的真缺口。
- [ ] **numina 难度别一刀切 hard**：按 problem_type / 题长 / 答案类型做粗难度分级，cn_k12 类不误标。
- [ ] 剔除 bespoke（跨学科）+ openthoughts3（超长）；清 numina proof/空答案 ~1.8万。
- [ ] （观察后定）应用题风格：若 gsm8k/cmath 卡住，给 metamath「求x」降权、提 gsm8k 直接风格。
- [ ] （若要救MCQ）补「简单MCQ→简短CoT→单\boxed{字母}」样本。
- [ ] **动态课程前置**：建「compute_cot val 每 source 采N条」的细粒度 eval + 探索模型自评 solve_rate，替代当前静态来源标签。
