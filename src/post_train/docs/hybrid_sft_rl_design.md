# 混合 SFT+RL 协同演化训练方案 — 可行性与设计报告

> 针对"每阶段联合 SFT+RL 损失、比例由 SFT 重渐变到 RL 重、数据由(算术+通用)渐变到(数学题)"的设想做可行性评估与初步设计。
> 基于现有数据资产（`docs/traindata/`）、v3 教学诊断、以及 `verl/` 框架的实际能力。编写日期 2026-06-09。

---

## 0. 方案复述

一条 **SFT→RL 连续过渡的混合课程**，不是"先 SFT 再 RL"的硬切，而是每阶段同时带两个损失项，逐步平移：

| 阶段 | 数据重心 | 数学题损失 | 通用 SFT |
|---|---|---|---|
| 1 | 基本算术 + 通用 SFT | 数学题 **SFT+RL**（让模型自演化更完整 CoT） | 重 |
| 2 | 基本算术(SFT+RL) + 数学题(SFT+RL) | SFT+RL | 渐减 |
| 3 | 逐减合成算术，全量数学题 | SFT+RL | 少 |
| 4 | 逐减 SFT 量、加大 RL 量 | 过渡到全量 GRPO | ~无 |

核心洞察（**成立**）：外部数学题的解答**运算过程没展开**（orca 口语解、numina 参考解都跳步）；纯 SFT 会把"跳步"教进去（正是 v3 诊断的病根），纯 RL 对弱 base 又**冷启动无信号**。联合损失两头取长：**SFT 项给地板/格式锚 + 梯度信号，RL 项让模型探索出被验证为对的更完整 CoT**。

---

## 1. 可行性评估

**结论：可行、动机扎实，但需要 ① Verl 改造 ② 谨慎的损失/课程调度 ③ 算力预算把控。**

- ✅ **方向有据**：这属于"示范增强的 RL / SFT-锚定 RL"一类（用金标准轨迹的 SFT 项稳住策略，用可验证奖励的 RL 项探索改进），对弱 base 冷启动尤其合理。
- ✅ **Verl 基本就位**：本仓库 `verl/` 是 PPO/GRPO 框架，自带 `reward_score`（math_verify / math_dapo / gsm8k / prime_math grader）、`tracking`（wandb/tb/swanlab）、checkpoint manager、KL 控制、entropy/clip 等监控。
- ⚠️ **需改造点**：标准 GRPO **没有 SFT 辅助项**——要在 actor 更新里**加一项金标准轨迹的 NLL 损失**（`L = λ_rl·L_GRPO(rollouts) + λ_sft·L_SFT(gold)`）。这与 CLAUDE.md 已计划的"改 Verl 实现 LoRA-only 权重同步"是同一类源码改造。
- ⚠️ **关键风险**：冷启动（pass@k≈0 时 RL 无梯度→靠 SFT 项兜底）、奖励黑客/格式坍塌（RL 退化成无 CoT 直接蒙答案→靠格式奖励+SFT 锚+KL 约束防）、算力（RL 的 rollout 采样很贵，1 周预算紧）、调度稳定性（λ 与 KL 系数要 warmup/anneal）。

### 1.1 SFT 与 RL 的分工：三能力分层（核心认知，决定成败）

> 这是整套方案的认知地基。它同时回答了两个看似矛盾的诉求——"基座弱，题目必须 SFT 训" 与 "外部解答跳步、又想要完整 CoT"——并解释为什么 v3 纯 SFT 会崩、而本方案不会。

把模型"会解数学题"拆成**三种能力**，明确各由谁负责：

| # | 能力 | 含义 | 谁负责 | 用什么数据 |
|---|---|---|---|---|
| ① | **算术执行力** | 多位加减乘除能逐位算对（不在最后一步瞎填） | **SFT** | Compute_Cot worked（39万） |
| ② | **解题理解力** | 读懂题意、抓条件、选方法/套路、组织解答结构 | **SFT** | 外部数学题（orca/metamath/numina/openr1/deepscaler，**跳步解照用**） |
| ③ | **格式/语言** | 输出成形：`<think>` 逐步 + `#### \boxed{}`、语言连贯 | **SFT** | 全部 + 通用 SFT |
| ④ | **展开行为 + 正确率** | 在难题上"该展开就展开"、并把答案做对 | **RL** | 可验证难题（big-math/dapo/deepscaler/gsm8k） |

**为什么三件都要 SFT，且题目必须进 SFT**：基座弱，连"读懂题、像样地搭出解题框架"都不会 → 纯 RL 的 pass@k≈0、没有梯度，学不动。所以 ②解题理解力**必须靠在题目上做 SFT** 先装进去（这是 RL 的冷启动底子），①算术执行力靠 Compute_Cot worked 装进去，③格式靠统一模板装进去。

**为什么外部解答的"跳步"不是问题**（这是和 v3 的关键和解）：
- v3 纯 SFT 崩，真正原因**不是"跳步数据有毒"**，而是 **模型既没有①算术执行力、又没有④RL 去奖励展开** → 它想展开也展不对（v3 实测：partial product 对了、最后求和瞎填）。
- 本方案里：①已由 Compute_Cot 装好（模型**有能力**把展开的每步算对）；④由 RL 提供——难题只有**展开**才能答对，RL 奖励"答对" → **自然 elicit 出"该展开就展开"的行为，且模型有能力展对**。
- 因此外部题的跳步解照样喂 SFT（教②解题方法/结构），跳步那部分由 ①+④ 兜住，**不需要把题目从 SFT 里排除，也不需要把每步加法都重写展开**（也不现实）。

**闭环**：SFT 装好 ①算术执行力 + ②解题理解力 + ③格式 → 做外部难题时，模型用②搭框架、用①算对展开的每一步、RL（④）奖励答对并把"展开"强化下来。**这正是用户最初设想的"数学题损失同时有 SFT 项 + RL 项"**——SFT 教理解/方法/格式，RL 逼展开/磨正确，二者并行。

**由此推出的两条硬约束**：
1. **Compute_Cot worked 数据必须贯穿、保持足量**——它是①算术执行力的唯一来源；一旦它在后期被稀释到模型"逐位算对"能力退化，RL 奖励展开、模型却展不对，整套就空转。
2. **SFT 和 RL 池都铺全难度谱（易+中+难），动态压向"前沿带"**——不是只挑某一段：
   - **易题不可少**：防遗忘基础；给 GRPO 提供"答对"的正样本（组内全错→优势全 0→没梯度）；配长度奖励教"简单题就简短、别啰嗦"。
   - **中难题最有信息量**：组内有对有错→reward 方差大→梯度强；也是"必须展开才能对"逼出④展开行为的主力。
   - **难题做拉伸**：少量，挑战上限。
   - **难度是相对当前模型的、且随训练右移**：故不静态挑段，而是用 eval 实时测**模型自己当前的 solve_rate**，把采样权重动态压向"信息量最大的前沿带"（当前 solve_rate≈0.3–0.7），易/难两端保留低比例。前沿带随模型变强自然右移（接 §6 eval 驱动动态课程）。
   - 静态难度信号（big-math `llama8b_solve_rate`、cmath grade/步数、Compute_Cot difficulty）用于**铺谱与初始分层**，动态权重靠 eval 模型自评。

---

## 2. 损失设计（核心）

每个数学 batch 的 actor 损失：
```
L = λ_rl · L_GRPO(rollouts, reward)  +  λ_sft · L_SFT(gold_trajectory)  +  β · KL(π‖π_ref)
```
- **L_GRPO**：对每题采 G 条 rollout，reward=答案验证(+格式/长度整形)，组内标准化优势，PPO-clip 更新（GRPO，无 critic）。
- **L_SFT**：金标准解答（统一格式包裹）的 teacher-forcing NLL。**对有解答的题都用**——既包括 Compute_Cot worked（装①算术执行力），也包括外部题的解答（装②解题理解力/方法/结构，跳步解照用，见 §1.1）。仅 big-math/dapo 这类**只有答案**的数据进不了 SFT 项、只走 RL。
- **β·KL**：对参考策略的 KL 约束，防漂移/坍塌（Verl 内置 `reward_kl_penalty`）。
- **不矛盾性**：外部解答跳步，但 SFT 项学的是②理解/方法/格式；"逐步算对"由贯穿的 Compute_Cot（①）保证、"该展开就展开"由 RL（④）逼出。三者并行不打架（§1.1）。

**λ 跨阶段调度（前期纯 SFT 打基础，GRPO 中后期才开）**：

| 阶段 | λ_sft : λ_rl（数学题） | 备注 |
|---|---|---|
| 1 | **1 : 0**（纯 SFT） | 打基础：弱 base 上 RL 没信号还不稳，先广泛 SFT 扎实①②③ |
| 2 | **1 : 0**（纯 SFT，末段可微量 warmup） | 继续 SFT 加深加广；模型像样后期可小权重试 RL warmup |
| 3 | **~0.5 : 0.5 → 0.3 : 0.7**（GRPO 正式开） | SFT+RL 联合，RL 渐主导磨锐 |
| 4 | ~0.1 : 0.9 → 0 | 过渡全量 GRPO |

> ① **RL 中后期才开**：前两阶段基本纯 SFT 打基础，阶段3 起正式开 GRPO（弱 base 早开 RL 浪费且不稳）。
> ② λ 连续 anneal（cosine），阶段间平滑过渡，不硬切。算术与题目可各自一套 λ。

### 2.1 实现效率：SFT/RL 反传"错开" + 梯度累积（省显存，关键）

**绝不能把 SFT 图和 RL 图同时堆在显存里再一起反传**（CoT 序列长、激活大，两图同堆易爆 80G）。改为**分开反传 + 梯度累积**：

```
# 数学上等价于 bp(λ_sft·L_SFT + λ_rl·L_RL)，但峰值显存 = max 而非 sum
loss_sft = λ_sft · L_SFT(gold_microbatch);   loss_sft.backward()   # 反传后立即释放该图激活
loss_rl  = λ_rl  · L_RL(rollout_microbatch); loss_rl.backward()    # 梯度累加到同一参数
optimizer.step(); optimizer.zero_grad()
```
- **峰值显存 = max(SFT 单图激活, RL 单图激活)，不是两者之和**（梯度线性可加，LayerNorm 无状态 → 累积安全）。
- **时间上错开**：RL 每步先在 A4000(card4-7) 异步生成 rollout（秒级），这段时间 A800(card1) 本空等 → **让它先跑 SFT 的 forward-backward**；rollout 回来再跑 RL 的；两者累完一次 step。**既降显存峰值，又把采样延迟藏在 SFT 计算后**，A800 不空转。
- **与已有设计契合**：LoRA 使梯度/优化器态极小、激活成大头 → 错开反传收益最大；叠 **gradient checkpointing** 再降激活；**异步 GRPO 的 rollout buffer** 正是解耦"生成/更新"的抓手，buffer 攒样本时 trainer 用 SFT micro-batch 填窗口。
- 注意：PPO 的 `old_log_probs` 在更新前算好（与 SFT 项独立）。

---

## 3. 阶段化课程：数据 × 比例 × 量（尽量用满，~1 周）

> **总原则（据反馈修正）**：① **数据广、量大、混杂**——各类数学题在 SFT 阶段就**充分混着上**，**不做"按难度/题型分阶段硬门控"**；阶段差异主要是 **SFT→RL 的比例**与**软性侧重**，不是"换数据集"。② **GRPO 中后期才开**，前期纯 SFT 广泛打基础。③ 难度课程是**软加权**（动态多采前沿带），底下始终是全谱大混合。

> SFT 项数据 = ①Compute_Cot worked(算术执行力,**贯穿保量**) + ②外部题解答(解题理解力,**各类题大量混杂、原解照用不重写**) + ③格式/通用。RL 项(中后期) = 可验证答案的难题池。

| 阶段 | 性质 | SFT 项（广混大：①算术 + ②各类数学题 + ③通用） | RL 项（④，中后期开） | 通用 SFT |
|---|---|---|---|---|
| 1 | **纯 SFT 打基础** | Compute_Cot worked + **各类数学题广混(orca/metamath/numina/deepscaler… 原解照用)** + 通用 7 件套 | — 关 | 重(中英) |
| 2 | **纯 SFT 加深** | 同上，**全量充分混杂**；难度软性向中段倾 | —（末段可微量 warmup） | 渐减 |
| 3 | **开 GRPO，SFT+RL** | 维持广混 SFT（①贯穿、各类题持续在） | big-math + gsm8k + dapo + deepscaler，动态压前沿带 | 少量防退化 |
| 4 | **RL 主导→全量 GRPO** | 少量 SFT 锚(保①②③不退化) | big-math(主) + dapo + deepscaler | ~无 |

- **不要把课程做死**：上表"阶段"主要标 SFT/RL 比例与侧重；**数据本身从阶段1 起就广泛混杂、量大铺满**，把我们手上各类数学题（中英、易难、应用/竞赛）尽量都用上，而不是按阶段切片喂。
- **①不能稀释**：Compute_Cot worked 每阶段保持足量（哪怕后期），否则算术执行力退化→RL 奖励展开但模型展不对（§1.1 硬约束1）。
- **难度课程（SFT 与 RL 都全谱铺底 + 动态压前沿）**：big-math `llama8b_solve_rate`、cmath grade/步数、Compute_Cot difficulty 用于铺谱/初始分层；训练中用 eval 测**模型自评 solve_rate**，把权重动态压向前沿带（≈0.3–0.7），易端保留(防遗忘+给正样本+教简洁)、难端保留(拉伸)。前沿随模型变强右移（§1.1 硬约束2、§6）。
- **去重/防泄漏**（贯穿）：题源高度重叠（orca/olympiads/math 系），SFT/RL 池**跨集去重**，且与 gsm8k/cmath/math-beyond 评测**题面隔离**。

---

## 4. 数据格式统一（务必先做）

**统一成 Verl 的 RLHF parquet + 单一对话/输出模板。** 每条样本字段：
```
prompt:        [{"role":"system","content": <统一指令: 先<think>逐步推演,再 #### \boxed{答案}>},
                {"role":"user","content": <题面>}]
reward_model:  {"ground_truth": <可验证答案>, "style": <验证器选择, 见 §5>}
gold_response: <统一格式的金标准解答>   # 仅 SFT 项需要; 仅答案的数据可空
data_source:   <来源, 用于选验证器 + per-source 统计>
extra_info:    {"difficulty": ..., "source": ...}   # 课程/监控用
```
输出统一为：`<think>\n逐步(worked)推演\n</think>\n#### \boxed{answer}`（与 Compute_Cot 一致）。

各类数据的归一：
- **Compute_Cot**：已是该格式（worked CoT），`gold_response` 直接用，答案精确。
- **orca/metamath/numina/openr1/deepscaler**：把解答**包进 `<think>`**、抽末答进 boxed 作 `gold_response`；**跳步解优先只做 RL（不做强 SFT 监督）**或重写为 worked。
- **big-math/dapo**：仅 `prompt`+`ground_truth`（无 gold_response）→ 纯 RL。DAPO 本就是这个格式（`reward_model{ground_truth,style}`），可作模板参照。
- **通用 SFT**：chat messages；本阶段轻量或不带 think。

---

## 5. 答案验证（奖励函数）

**直接复用 Verl `verl/utils/reward_score/` 现成模块，按数据源路由验证器**：

| 数据源 | 验证器 | 说明 |
|---|---|---|
| Compute_Cot | 自有精确(int/Fraction/Decimal) 或 math_verify | 答案我们可控 |
| gsm8k | `gsm8k`(解析 #### 数字) | 数值精确匹配 |
| big-math / numina / deepscaler | `math_verify`(sympy 等价) | 处理 LaTeX/分数/区间/集合 |
| dapo | `math_dapo`(MATH_v2 规则) | 与 DAPO `style` 对齐 |
| 竞赛/复杂 | `prime_math` grader | 更强的等价判定 |

- **奖励整形**：`reward = 1[答案正确]`（主）+ 小额 `格式奖励`(含完整 `<think>` 且有 boxed) + 有界 `thinking 长度奖励` - `循环重复惩罚`。当前默认 format +0.1、thinking 长度从 64 到 512 token-ish 线性给分且最多 +0.2、重复最多 -0.4；组件分开记录(§6)。
- **注意**：math_verify 会有**等价形式假阴**（如 `1/2` vs `0.5`、区间写法）→ 监控"验证器判负但人看是对"的比例；必要时多验证器投票。
- **去 proof/非数值**：RL 池只留可机器判分的题（big-math 已 closed-form；numina 的 `answer=proof` 剔除）。

---

## 6. 过程监控（监控什么 + 怎么监控）

**A. Verl 默认就记（经 `tracking`→wandb+tensorboard）——确认开启即可：**
- `actor/entropy`（**策略熵**，CLAUDE.md 要的，监控坍塌）、`actor/ppo_kl`、`actor/grad_norm`、`actor/pg_clipfrac`、`actor/reward_kl_penalty`、`actor/mfu`
- reward 的 mean/std、advantages 的 mean/max/min、`response_length`
- SFT 项加上后：`actor/sft_loss`、`actor/lambda_sft`（需在改造里手动 log）

**B. 必须自定义补的（训练前要确保被记录）：**
- **per-source / per-difficulty 准确率**（eval 驱动动态课程的信号；按 source/grade/solve_rate 切桶）
- **Pass@1 / Pass@8**（gsm8k 英、cmath 中、math-beyond hard 天花板）
- **格式合规率**（输出含 `<think>…</think>` + boxed 的比例）、**验证器通过率**
- **CoT 长度 × 正确率（按难度分层）**——**本方案是否生效的核心验证指标**：期望看到"难题上 CoT 变长 **且** 正确率上升"（RL 成功逼出展开，且模型有①算术能力把展开算对）。若"难题上短而错"=展开没发生；"长而错"=①算术执行力不足、需加 Compute_Cot；"易题上无谓变长"=过度展开/啰嗦，需长度整形。
- **reward 组件分解**（答案奖励 vs 格式奖励）、**两损失项的梯度范数**
- **三段耗时**（rollout 采样 / 训练 / 权重同步）—— 异步 GRPO 必看（CLAUDE.md）
- **动态采样权重历史**（哪些 source 权重随时间怎么变）

**C. 怎么监控（基础设施）：**
- **标量** → Verl Tracking 双写：`wandb`(主，远程可看) + `tensorboard`(本地兜底) + `console`。
- **定性 dump** → 每轮异步 eval(A4000) 写 `eval_dumps/step_N/heldout.jsonl`：`{prompt, generation, gold, reward, pass, source, difficulty}`（v3 诊断正是从这种 dump 正则解析出根因——**这套 dump 必须有，是事后追溯的命根**）。
- **评测异步、不阻塞训练**：A800 训、A4000 评/采样；卡号先 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 实测。

---

## 7. 本次训练会产出哪些文件/日志（**训练前必须对齐**）

> 目的：逐项确认"我们想要的信息"是否真被落盘。下表标注 Verl **默认有** / 需**自定义补**。

| 产出 | 位置/形式 | 默认? | 记录什么 |
|---|---|---|---|
| **模型检查点** | `<ckpt_dir>/global_step_N/`（actor 权重 / **LoRA adapter**）| ✅ | 按 save_freq；LoRA-only 同步是改造点 |
| **标量日志** | wandb run + `tensorboard/` event 文件 + console | ✅ | entropy/kl/reward/grad/clip/length… |
| **resolved config 快照** | ckpt 目录下的 config | ✅ | 超参/数据配比/λ 调度 |
| **eval 定性 dump** | `eval_dumps/step_N/heldout.jsonl` | ⚠️**需补** | prompt+生成+gold+reward+pass+source（追溯命根）|
| **per-source/难度 准确率** | metrics jsonl / wandb 自定义 panel | ⚠️**需补** | 动态课程信号 + 弱项诊断 |
| **Pass@k / 验证器通过率 / 格式合规率** | 同上 | ⚠**需补** | benchmark 主指标 |
| **动态采样权重历史** | jsonl | ⚠**需补** | 课程可追溯 |
| **三段耗时(采样/训练/同步)** | wandb 标量 | ⚠ 部分需补 | 异步 GRPO 调参依据 |
| **stdout/stderr** | 训练日志文件 | ✅ | 报错/进度 |

**训练前 checklist（防止"跑完发现没记")**：① 开 wandb+tensorboard 双写；② 接好 eval 回调写 heldout.jsonl；③ 自定义 reward manager 里把 per-source/格式/验证器结果 log 出来；④ 确认 ckpt 存的是 LoRA adapter（阶段4）；⑤ 跑 50 步 smoke 验证以上文件都真生成、字段齐全，再放量。

---

## 8. 风险与待确认

- **Verl SFT+RL 联合损失**需自行实现/验证（actor 加 SFT 项 + λ 调度）——优先级最高的工程项，与 LoRA-only 同步一起改。**实现务必用"错开反传 + 梯度累积"（§2.1），不要把两个损失图同时堆显存里**，并把 SFT 反传塞进 rollout 采样窗口。
- **冷启动（已规避）**：GRPO 中后期才开，前期纯 SFT 把基础打扎实，避免弱 base 早期 RL 无信号。
- **格式坍塌/奖励黑客**：格式奖励 + KL + SFT 项锚 + 监控 CoT 长度与 entropy。
- **算力/1 周**：RL 采样最贵且中后期才开；前期 SFT 吞吐高、可大量广混；RL 放量集中在动态课程选出的前沿带。
- **验证器可靠性**：math_verify 假阴；监控并必要时多验证器。
- **数据泄漏**：SFT/RL 池跨集去重 + 评测题面隔离（competition-math/MATH 系、math-beyond 来自 DAPO/DeepScaleR，须剔重）。

**待你拍板**：① 前期纯 SFT 跑多少 step / 何时切 GRPO（阶段3 的触发条件，如 gsm8k pass@1 达某阈值）；② 各阶段 λ_sft:λ_rl 的具体 anneal 曲线；③ Compute_Cot 后期保量下限（守①）；④ RL 难度前沿带的 solve_rate 区间。
> 注：§1.1 分工（①②③走 SFT、④走 RL）+ §3 总原则（广混大、不硬门控、GRPO 中后期）是本方案地基，非待定项。

---

## 9. 一周实施排期（建议）

- **D1**：数据格式统一管线（全部 → 统一模板/格式 + 验证器路由）；**跨集去重 + 评测隔离**；备好"广混大"的 SFT 数据池。
- **D2**：SFT 训练管线 + 监控/日志（per-source 准确率、格式合规、heldout.jsonl dump）；eval(card2,3) 异步评测接好。
- **D2–4**：**阶段1–2 纯 SFT 放量**（广混大：Compute_Cot worked + 各类数学题 + 通用），打基础；持续看 gsm8k/cmath 等是否成形。
- **D4**：Verl 改造——actor 加 SFT 辅助损失 + λ 调度 + LoRA-only 同步；50-step GRPO **smoke**（验证 §7 文件/字段、两损失项、卡分配/异步正确）。
- **D5–6**：**阶段3 开 GRPO**（SFT+RL 联合，RL 渐主导）；eval 动态调难度/source 软权重。
- **D7**：阶段4 过渡（加大 RL → 全量 GRPO）+ 汇总评测(gsm8k/cmath/competition-math/math-beyond, Pass@k, 策略熵)与 dump 复盘。

> 关键：**D1–D2 先把数据广混 + 格式 + 评测/日志打通**，前期靠高吞吐 SFT 把基础打满；**GRPO 到 D4 改造好、D5 才正式开**，弱 base 不早上 RL。

---

## 附：关键依据
- 数据资产与配比：`docs/traindata/{general_sft,math_sft}/README.md`、`docs/training_plan.md`
- 教学 CoT 必须 worked（否则 SFT 教坏）：`fastrl/outputs/sft/compute_cot_arithmetic_cot_diagnosis.md`、`docs/traindata/compute_cot_audit.md`
- Verl 现成能力：`verl/utils/reward_score/`（验证器）、`verl/utils/tracking.py`（日志）、`verl/trainer/ppo/`（指标）
