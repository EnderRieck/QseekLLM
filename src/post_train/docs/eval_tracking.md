# 阶段1-2 基础 SFT · 过程评测跟踪

> 活文档：记录 `sft_foundation` 每个 checkpoint 在 held-out 上的指标演化与诊断结论。
> 每出新 step 追加一行表 + 必要的诊断。最近更新 2026-06-10（v2 回填完成，记录至 step 5500；训练 Epoch 2/2 进行中，约 60%，预计 06-11 凌晨跑完）。

## 评测设施（怎么产生这些数）
- **脚本**：`eval/async_eval.py`（watch 模式监视 checkpoint，自动评新 step）。
  - 后端 **vLLM**（continuous batching，比 HF generate ~10x）；`--backend hf` 兜底。
  - **多卡数据并行**：held-out 按长度均衡切片，每空闲 A4000 一个独立实例（card2,3；忙时可加 4-7）。
  - 贪心 Pass@1（temperature=0），确定性可比、对 GPU 非确定性鲁棒。
  - 实时进度（每 15s 打印各卡完成度）+ **增量落盘**（每块 flush，崩溃保结果）。
- **held-out**：`eval/heldout.jsonl`，**v2 能力分解版**（2026-06-10 重建，`eval/build_heldout.py`，实测跑量 **3881 题 = 3864 可判分 + 17 自由问答**，训练外无泄漏）。
  设计转向：过程评测=按"在训能力"分解的【窄/对齐/低方差】信号，不是广覆盖 benchmark 拼盘（v1 拼盘见 `eval/heldout_v1.jsonl`，75% 是语言/格式错配的旁观噪声）。
  - ①算术(核心,in-dist)：**compute_cot 256 子源×~7=1848**（每子源 N=8，旧版仅 2 是噪声→现可做 per-source 课程信号种子）
  - ②应用题泛化(英文)：**gsm8k 500**(本域,旧 200 方差太大) + **SVAMP 300**(结构扰动) + **GSM-Plus 798**(7类对抗扰动,测模板过拟合)
  - ③中文迁移(次要旁观)：**cmath 200**(数值判分)
  - ④难度探针(观察,为阶段3)：**competition-math 200**(gold 重生成已修旧贪婪 bug→200/200 干净)
  - ⑤行为探针(观察,`eval/build_probe.py` → `eval/probe.jsonl`)：**probe-math 18**(手写陷阱/高考风题,有 gold 但每题 n=1,只看定性) + **probe-chat 17**(自由问答无 gold 不计 acc,人工看 `heldout.md` 可视化,含中英同题对照)
  - 砍掉：cmmlu/gaokao-mathqa/gaokao-mathcloze/bbh(中文/MCQ-boxed/通用推理错配,不可廉价修)。判分复用 reward.py 无改动。
- **产出**（均在 `/data/zilu/fastrl/checkpoints/sft_foundation/eval_dumps/`）：
  - `metrics.jsonl`：每 step 一行（overall/per-source/per-difficulty acc、format_rate、avg_gen_chars）。
  - `step_<N>/heldout.jsonl`：完整 IO dump（每条 prompt/generation/gold/correct/has_format）。
  - `step_<N>/heldout.md`：可读版（`python -m eval.dump_to_md --ckpt-dir … --step N`，支持 `--source/--only-wrong/--per-source`）。
  - `tb/`：tensorboard 标量。
- ⚠️ per-source **课程信号**（268 source 各采足量）尚未建——需从 `Compute_Cot/data/clean/val` 每 source 采 N 条，才能驱动动态采样（见 training_plan §4.3）。

## ⚠️ 评测集 v1→v2 重建 + 健康度读数口径（2026-06-10 定）

**背景**：v1（2018 题拼盘）选型时没和训练分布对齐——cmmlu/gaokao/bbh 中文或 MCQ-boxed 错配、competition-math 超纲，75% 算力花在旁观噪声上，真信号只剩 gsm8k(200,方差大)+compute_cot(每子源2,纯噪声)。**根因是用"终评广覆盖"思路干"过程监控"的活**（两者需求相反）。已重建为 v2 能力分解版（见上）。

**v2 健康度读数（按能力轴, ★=主看）**：
| 能力轴 | 源 | 判分 | 角色 |
|---|---|---|---|
| ★①算术执行力 | compute_cot 1848 | compute_cot | **主**, in-dist, 单调性最可靠; 可下钻 256 子源/类目 |
| ★②应用题泛化 | gsm8k 500 / SVAMP 300 / GSM-Plus 798 | gsm8k(数值) | **主**; 三者对比抓"真泛化 vs 记模板"(下) |
| ③中文迁移 | cmath 200 | math_verify | 次要旁观(中文几乎没训) |
| ④难度探针 | competition-math 200 | math_verify | 观察(本阶段≈0,阶段3才动) |

**关键交叉读法（v2 新增能力）**——直接量化诊断4的"模板过拟合"隐患：
- **gsm8k↑ 且 SVAMP/GSM-Plus 同步↑ = 真泛化**；**gsm8k↑ 而 GSM-Plus 卡住 = 在记模板**(被扰动打回原形)。GSM-Plus 按 7 类扰动分源,可看具体哪类扰动崩(数值替换/数位扩展/干扰项/批判性…)。

**✅ v1→v2 断点已消除（2026-06-10 凌晨回填）**：v2 建好后用 async_eval 把 step 500–2500 的旧 checkpoint 在 v2 上**全部重测回填**（02:16–02:44 落盘），step≥3000 起在线跟训。因此 `metrics.jsonl` 里 step 500–5500 全部 11 个点为**同一 v2 口径，可直接纵向比**。v1 原始记录（step 500–2000，n=2018）存档于 `metrics_v1.jsonl` + `eval/heldout_v1.jsonl`，仅作历史，勿与 v2 跨比。
**正经统一终评仍推迟到训练完**(届时加 MATH-500 分级 + 更广 bench 横向比)。

## 外部基线参考线（2026-06-11,Qwen3-1.7B 两档,同一 heldout v2）

> 跑法:`PYTHONPATH=. .venv/bin/python eval/baseline_heldout.py --model /data/zilu/fastrl/checkpoints/external/<dir> --device 4`(复用 async_eval 全套判分);样本级 dump 在各模型目录 `eval_dumps/step_0/heldout.{jsonl,md}`。

| source | F2@5000(我们) | Qwen3-1.7B-**Base** | Qwen3-1.7B-**Instruct**(thinking) |
|---|---|---|---|
| **overall** | ~38 | 45.8 | **75.7** |
| compute_cot | 63.0 | 51.3 | 72.5 |
| svamp | 27.7 | 30.0 | 90.7 |
| gsm8k | 11.0 | 48.4 | 85.4 |
| gsm-plus | 6.3 | 39.3 | 77.9 |
| cmath | 37.0 | 39.0 | 66.0 |
| comp-math | 4.5 | 44.5 | 58.5 |
| probe 探针 | 7/18 | 7/18 | 15/18 |
| 乘法切面 | 15.2 | 33.3 | 93.9 |
| format_rate | ~95 | 0(无 think 包裹,boxed 提取正常,acc 可信) | 90.6 |
| avg_chars | ~450 | 728(max_new=640) | 5,383(max_new=4096) |

**读法**:
- **Base 45.8** = 预训练语料差距的量化:解题类(gsm8k/gsm-plus/comp-math)落后 30-40pp 源自人家 36T token 预训练(或含污染),F2/S3 不必也不可能对标;但 compute_cot 我们 63>51.3、探针打平 → **digit-split+算术骨干路线在其瞄准的能力上赢了大厂 base**。
- **Instruct 75.7** = 成熟后训练管线天花板:base→instruct +30pp(gsm8k 48→85,乘法 33→94)即 S3+S4 的方向参照。注意它靠 5,400 字符 thinking 才到 72.5 cc(我们 450 字符到 63),**token 效率是我们的本钱**(S4 推理预算控制时用)。
- Instruct 探针仅错 3:"反模板·直接算即可别设x"+两道概率——模板病连它也犯。Level 5 仅 33.9%,难尾对 1.7B 尺寸普遍硬。

## 指标演化（★=本阶段主看）

### v2 能力分解版（全 step 统一口径，step 500–2500 为 06-10 回填）
| step | %训练 | ★compute_cot | ★gsm8k | ★SVAMP | ★GSM-Plus | cmath | comp-math | overall | format_rate | avg_chars |
|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 5.4% | 26.8% | 2.8% | 8.3% | 1.5% | 1.0% | 2.5% | 14.3% | 87.7% | 499 |
| 1000 | 10.8% | 35.2% | 2.8% | 11.3% | 2.1% | 4.5% | 4.5% | 19.0% | 90.1% | 500 |
| 1500 | 16.1% | 42.1% | 2.4% | 12.3% | 3.0% | 5.5% | 2.5% | 22.5% | 92.0% | 475 |
| 2000 | 21.5% | 46.2% | 3.8% | 17.0% | 2.9% | 10.0% | 2.5% | 25.2% | 92.6% | 479 |
| 2500 | 26.9% | 50.6% | 5.6% | 16.7% | 3.1% | 11.0% | 2.5% | 27.6% | 91.9% | 452 |
| 3000 | 32.3% | 53.0% | 5.8% | 21.3% | 2.6% | 11.0% | 2.0% | 29.0% | 91.3% | 480 |
| 3500 | 37.7% | 56.5% | 5.2% | 23.7% | 3.3% | 11.5% | 4.0% | 31.1% | 92.2% | 463 |
| 4000 | 43.1% | 60.7% | 6.2% | 25.7% | 4.0% | 11.0% | 3.5% | 33.4% | 92.5% | 440 |
| 4500 | 48.4% | 62.6% | 7.8% | 23.7% | 3.5% | 15.5% | 1.5% | 34.4% | 92.6% | 455 |
| 5000 | 53.8% | 64.1% | 8.2% | 23.3% | 3.8% | 16.5% | 3.5% | 35.4% | 93.1% | 432 |
| 5500 | 59.2% | 66.4% | 8.6% | 23.7% | 4.0% | 14.5% | 4.0% | 36.6% | 93.0% | 454 |

> 读法：compute_cot=①算术单调性；gsm8k vs SVAMP vs GSM-Plus 三者落差=②泛化质量(同步涨=真泛化,GSM-Plus 掉队=记模板)。probe-math（n=18，一题≈5.6pp）在 0%~16.7% 间跳动，纯噪声只看定性，不进此表。
> %训练按总步数 9288（4644/epoch × 2）折算。
>
> **🔴 2026-06-10 审计发现 SVAMP 列被污染**：heldout 的 300 题 svamp 中 **63 题(21%)逐字在训练集**(经 orca-math 携带,SVAMP 题源被 orca 语料吸收;build 时 EVAL_PATHS 没有 svamp)。**上表 SVAMP 列读数虚高,剔除重算前不要引用**;compute_cot 列另有 7 题 train/val 切分穿透(0.4%,影响微)。gsm8k/GSM-Plus/cmath/comp-math 列不受影响。修复方案见 `data_audit_report_20260610.md` §二/§五-P0。

### v1 拼盘版（step 500–2000 实测，历史存档 `metrics_v1.jsonl`/`heldout_v1.jsonl`；与 v2 不同尺，勿跨比）
| step | %训练 | gsm8k(200) | compute_cot(505) | format_rate | avg_gen_chars | overall(含旁观) |
|---|---|---|---|---|---|---|
| 500 | 5.4% | 0.5% | 30.1% | 54.7% | 606 | 11.8% |
| 1000 | 10.8% | 6.0% | 34.5% | 54.7% | 616 | 13.7% |
| 1500 | 16.1% | 2.5% | 39.0% | 56.0% | 536 | 14.1% |
| 2000 | 21.5% | 6.0% | 46.1% | 55.2% | 619 | 16.5% |

> 读法：**compute_cot 单调稳涨(30→34.5→39)=①算术在长**；gsm8k 抖(0.5→6→2.5，n=200 小样本噪声大)，等 step 2000 定性。step 1500 gsm8k 错题看：简单直算题已会，错在②读不懂题/列错式+误套代数（见 step_1500/gsm8k.md）。

<details><summary>旧的 overall/全 source 明细（旁观，保留备查）</summary>

| step | %训练 | overall acc | format_rate | avg_gen_chars |
|---|---|---|---|---|
| 500 | 5.4% | 11.8% | 54.7% | 606 |
| 1000 | 10.8% | 13.7% | 54.7% | 616 |
| 1500 | 16.1% | 14.1% | 56.0% | 536 |

</details>

### per-source 准确率（500/1000，v1 历史明细）
| source | 500 | 1000 | Δ | 备注 |
|---|---|---|---|---|
| compute_cot | 30.1% | **34.5%** | +4.4 | ①算术执行力在涨 |
| gsm8k | 0.5% | **6.0%** | +5.5 | ②英文应用题从≈0起飞 |
| cmath | 0.5% | **4.5%** | +4.0 | ②中文应用题起来了 |
| cmmlu | 15.8% | 15.8% | = | MCQ，低于随机(见诊断) |
| gaokao-mathqa | 10.8% | 10.0% | −0.8 | 噪声范围 |
| bbh | 5.1% | 4.6% | −0.5 | 通用推理被迫套 boxed |
| competition-math | 1.2% | 1.2% | = | 竞赛题≈0，超纲(正常) |
| gaokao-mathcloze | 0.0% | 0.0% | = | 难 |

### format_rate 按 source（500/1000，v1 历史明细；解释 v1 总 format_rate 为何"持平"）
| source | 500 | 1000 | Δ |
|---|---|---|---|
| compute_cot | 97.6% | 98.0% | +0.4 |
| gsm8k | 87.0% | 92.0% | +5.0 |
| cmath | 21.0% | 33.5% | +12.5 |
| competition-math | 44.0% | 48.4% | +4.4 |
| cmmlu | 23.3% | 23.3% | = |
| bbh | 67.1% | 61.1% | −6.0 |
| gaokao-mathcloze | 22.9% | 20.3% | −2.5 |
| gaokao-mathqa | 19.2% | 6.4% | −12.8 |

## 诊断结论（截至 step 5500，v2 口径，2026-06-10）

1. **训练健康，无需干预**：Epoch 2/2 约 20%（全局 step ~5575/9288），loss 0.83–1.07、grad_norm ~0.28、MFU ~0.41、lr 1e-5 恒定，A800(card1) 97% 利用率，已训 0.667B token，预计 06-11 凌晨 02:00 前后跑完。overall 36.6% 仍在涨，曲线未饱和，epoch 2 有收益。

2. **①算术执行力单调稳涨但增速放缓**：compute_cot 26.8%→66.4%，早期每 500 step +7~8pp，最近两段只 +1.5/+2.3pp。

3. **②泛化交叉读法：触发"记模板"警报（半实锤）**：
   - SVAMP 8.3%→23.7%，但 **step 4000 后进入 23~26% 平台**；gsm8k 2.8%→8.6% 缓涨未停；
   - **GSM-Plus 全程贴地 1.5%→4.0%**——同题对抗扰动即打回原形，7 类扰动全军覆没（逆运算 1.8%、整数-小数-分数转换 1.8%、干扰插入 3.5%、数值替换 4.4%、digit expansion 5.3%、加运算 3.5%、problem understanding 7.9%）。
   - 对照诊断旧4 的触发条件（compute_cot↑ 而应用题卡）：gsm8k/cmath 没卡死但增速远低于 compute_cot，且 GSM-Plus 实卡。**结论：阶段2 数据构建时直接落实 metamath"求x"题降权 + 提高 gsm8k 直接逐步算风格占比，不必再等。**

4. **运算类型短板：乘法**。svamp:Multiplication 仅 9.1%（减法 26.9%/除法 27.1%/加法 20.3%），probe"多位数乘法·逐位展开"0%。cmath 年级断崖：grade1 38.9% → grade3 11.3% → grade5/6 0%。→ 阶段2 Compute_Cot 数据优先加密多位数乘法/高年级覆盖。

5. **③中文：能算但表达退化**。cmath 1.0%→14.5% 在涨；但 probe-chat 中英同题对照（勾股定理）：英文版定义正确表达流畅，中文版循环复读+概念错误（"勾股定理是勾股定理…斜边上的高为c"）。**中文生成质量明显弱于英文** → 阶段2 通用 SFT 混入时加大中文对话语料比重。

6. **probe-math 全 0 区**（n=1/题只看定性）：高考风 18 连环题基本全错（偶有 1 题对属波动）、"经典陷阱·答案是3不是9"0%、"反模板·直接算即可别设x"0%（与诊断旧4 的风格冲突互证）。

## 诊断结论（截至 step 1000，v1 期，历史）

1. **训练健康，数学技能在涨且开始泛化**：compute_cot +4.4、gsm8k 0.5→6%、cmath 0.5→4.5%。①算术执行力提升，并外溢到应用题——正是 hybrid 设计要的「合成算术泛化到解题」早期苗头。才训 ~10%，趋势对。

2. **总 format_rate「持平 54.7%」是平均假象**：拆开看是两边对冲——
   - **训练分布(worked-CoT 数学)上 format 已基本解决**：compute_cot 98%、gsm8k 92%、cmath 升到 33.5%。**模型在它训练的格式上停得很好，不是"没学会收尾"**（更正了 step 500 时的初判）。
   - **拖后腿的是 MCQ/通用推理基准**：cmmlu 23%、gaokao 6-20%、bbh 61%——这些是「让模型 box 一个它没怎么训过的格式」，属 **train/eval 格式错配 + 题难**，不是模型缺陷。

3. **step 500 已识别的模型行为病征**（弱模型早期，预计随训练消退）：
   - 算术：结构对、多步大数计算错（`3^9=302`）——①还没装牢。
   - 选择题不会 commit 单一答案：枚举 A→B→C→D 各给 boxed，末尾 `#### \boxed{}` 空（8%空 boxed、12% 多 boxed）。**根因(2026-06-09 查训练集后更正)**：MCQ 训练分两档不连通——①通用数据 ~2.6万道(dynamics17k/coig/chinese-r1)是**通用prompt下裸字母**作答(无boxed)；②数学数据 ~2万个 `\boxed{字母}`(numina/openr1/bespoke)**全是竞赛难题**，step1000 解不出。**缺的是「简单选择题→简短推理→单个`\boxed{字母}`」这一档**(简单的在裸字母档、boxed的全在难题档)。cmmlu eval 用数学prompt要boxed→正落在能力空区。→ 若要救 MCQ，需补**简单/知识类 MCQ 的 boxed-CoT 格式样本**，而非泛泛"单选数据"。
   - 贪心退化复读（~11%），集中在难题/不收尾那批。

4. **应用题解法「风格冲突」**（2026-06-09 发现，由一道 bakery 题暴露）：模型对简单应用题**套变量/方程**（`2/3·x=1/2·(x+1)`）而非直接逐步算，列错又解崩。溯源训练数据**两种风格打架**：
   - 「直接逐步算」（无变量，gsm8k 风格）：gsm8k 仅 4% 用代数、compute_cot 9%——干净直接，但**应用题领域的量小**（gsm8k 才 7.5k）。
   - 「设变量列方程」：**orca-math 45% + metamathqa 46% ≈ 13 万条**主导（metamath 大量题本身就是"求未知变量 x"，代数解法对其是恰当的）。
   - 判定：这是**风格过度泛化**（模型把代数套到不需要的小学题上），**不是数据错**，靠更多训练+RL(④)纠偏，不急着洗。**触发式应对**：若 step 2000-3000 时 compute_cot 续涨而 gsm8k/cmath 卡住，再考虑 metamath"求x"题降权 / 提 gsm8k 直接风格占比。

## 待观察 / TODO
- [x] ~~format_rate 的 MCQ/gaokao 低分~~：v2 已砍掉 MCQ 源，问题随 v1 退役（定性为 eval 格式错配为主）。
- [ ] competition-math 长期≈0（5500 时 4.0%）：阶段3 难题课程才会动，本阶段不期望。
- [ ] 建 per-source 细粒度 eval（compute_cot val 每 source 采 N）以接通动态课程。
- [x] gsm8k/cmath 持续涨否 → **在涨但慢**（8.6%/14.5%），且 GSM-Plus 卡 4%——泛化"部分真"，见新诊断3。
- [x] **风格冲突触发监控** → 已到观察点并定性（新诊断3/6）：**阶段2 落实 metamath"求x"降权 + 直接风格增比**。
- [ ] 训练 06-11 凌晨跑完后：跑统一终评（MATH-500 分级 + 更广 bench），并对 epoch 末 checkpoint 选型。
- [ ] 阶段2 数据构建优先级（来自 5500 诊断）：多位数乘法/高年级覆盖加密、中文对话语料增比、扰动鲁棒性样本（GSM-Plus 七类）。

---

# F2 终评(2026-06-12,global_step_9858,8 卡,Pass@1/8,T=0.8)

> 产出:`<ckpt>/global_step_9858/final_eval/{summary.md,summary.json,<bench>.jsonl}`;dump 含贪心+8 采样全文(`gens` 字段)。
> 判分修复(本次评测中发现并修复,数字为重判后口径):①compute_cot 多解 gold("x=-4 or x=26")math_verify 只解析末解 → reward.py 拆解集逐项配对(cc-reserved pass@1 0.720→0.718,-10 条假阳性);②坐标点被当空开区间误判相等 → 逐分量比较。重判脚本 `eval/rescore_final_eval.py`(离线,不重生成)。

| benchmark | n | Pass@1 | Pass@8 | format |
|---|---|---|---|---|
| cc-reserved | 4372 | 71.8% | 84.4% | 100% |
| cmath | 1098 | 42.8% | 62.3% | 99.1% |
| svamp | 300 | 39.3% | 73.0% | 100% |
| gsm8k | 1319 | 16.0% | 43.0% | 97.4% |
| gsmplus(剔 critical thinking) | 2100 | 8.7% | 30.3% | 96.6% |
| math500 | 500 | 5.6% | 24.0% | 75.4% |

**评测盲区**:gsmplus critical thinking 类(300 题)gold=None(不可解题,期望模型指出条件不足),判分器遇 None 必判 0 → 该类构造性 0 分,已从口径剔除。模型行为上也确实无"条件不足"意识(自信硬算)。如需测此能力,S4 前给 reward.py 加 unanswerable 判定路径。

**诊断**:
1. **Pass@8/Pass@1 比值随领域难度递增**(cc 1.2x → svamp 1.9x → gsm8k 2.7x → math500 4.3x):"会而不稳"的空间大,RL(S4)红利区;svamp Pass@8=73% 说明简单应用题能力基本在分布里,缺的是稳定采出。
2. **math500 level1-2 pass@1 仅 11.6%/14.4%** → S3 R1 按原案 60:30:10,不压"易"。
3. **中文池见效**:cmath 42.8% 远超 gsm8k 16%,年级梯度单调(g1 66%→g6 22%);英文多步应用题(gsm8k/gsmplus)是主短板。
4. **cc-reserved 最弱子源**(S3 补数据靶子):复数方程/余数定理/裂项求和/等差等比求和 0%;三角(周期解集/象限符号)pass@1=0 但 pass@8=1.0(纯输出格式不稳,非不会);配方法/百分比变化/工程问题/3x3 行列式/期望 ≤6%。叠加 gsmplus 七类扰动:integer-decimal-fraction conversion 最脆(2%)。
5. **生成质量**:cc/cmath/svamp 干净(截断重复≈0);math500 截断 23.8%+重复退化 13.2%(难题 2048 token 不够+循环);gsm8k 典型错误模式=多步中途丢失题面状态(把中间结果当新条件)。
6. F1 比较仅作趋势参考(F1 数据脏+评测口径不同):F1 heldout acc 峰值 36.6%@5500,F2 heldout 45.2%@9500、终评如上,全面好转。

**S3 输入清单**:R1 配比 60:30:10 起步;补:中文指对计算(模板坍缩复测点)、数列求和/复数/余数定理/三角特殊值的基础 schema、小数分数互转扰动、多位数乘法续盯;16k 上下文解 math500 截断;贪心格式不稳子源观察是否随课程消退。

---

# Qwen3-1.7B 双基线对照(2026-06-12,与 F2 终评同口径:同判分器/Pass@1/8/T=0.8/max-new 2048)

> 产出:`/data/zilu/fastrl/checkpoints/external/qwen3_1_7b{_base,}/final_eval/`。
> ⚠️ instruct 的 math500 被 2048 token 截断显著压低(fmt 仅 37.6%,pass@8 49.6 反低于 base 78.6),其余项截断影响小。

| benchmark | 我们 F2 (p@1/p@8) | Qwen base (p@1/p@8) | Qwen instruct (p@1/p@8) |
|---|---|---|---|
| gsm8k | 16.0 / 43.0 | 48.1 / 92.0 | 76.9 / 89.8 |
| math500 | 5.6 / 24.0 | 52.6 / 78.6 | 34.0* / 49.6*(截断) |
| cmath | **42.8** / 62.3 | 34.5 / 86.0 | 63.8 / 91.8 |
| svamp | 39.3 / 73.0 | 28.7 / 90.7 | 84.0 / 91.0 |

**结论**:
1. **预训练底子差距是主轴**:Qwen base 的 pass@8 全线 79-92%(分布里"什么都有"),我们 24-73%。多步推理/竞赛(gsm8k/math500)差距最大,后训练补不齐,属预训练战场。
2. **对齐兑现度我们不差**:贪心口径我们在 cmath(+8.3)和 svamp(+10.6)赢 Qwen base——F2 的中文池和应用题对齐有效;Qwen base 没对齐,能力睡在分布里。
3. **instruct 展示了后训练的标准模式**:gsm8k 上 pass@8 几乎不动(92→90),pass@1 翻倍(48→77)——把分布能力兑现为稳定输出,天花板不变。这正是我们 S3+S4 在自己 pass@8 天花板内要复刻的过程,佐证"RL 只放大不创造"。
4. **路线含义**:我们的合理目标=在自己 pass@8 天花板内最大化 pass@1(如 gsm8k 16→40 区间);对标 Qwen instruct 的绝对值需要预训练语料升级,不在后训练范围。
