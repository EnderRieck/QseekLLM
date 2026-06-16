# QseekLLM 后训练计划 v2（2026-06-10 重构）

> v1（06-09）写于数据全面审计之前。本版吸收审计结论（[data_audit_report_20260610.md](data_audit_report_20260610.md)），把「计划 / 当前位置 / 修复门禁 / 阶段配方」合并为单一事实源。
> 数据集明细一律看 `docs/traindata/*`（22 张卡片均带 2026-06-10 审计补充节），评测明细看 [eval_tracking.md](eval_tracking.md)。

---

## 0. 总目标与贯穿原则

把 **1.7B 弱 base** 经「Foundation SFT → 数学 CoT SFT（动态课程）→ 难度课程 SFT → 异步 GRPO(LoRA)」训成具备**基本算术与数学方法推理**能力、可被 RL 进一步放大的模型。

四条贯穿原则：
1. **训练/评测物理隔离、异步不阻塞**：训练 A800(card1)，评测 A4000(card2,3)，RL 采样 A4000(card4-7)。
2. **eval 驱动动态课程**：per-source 准确率 → EMA → 反比软加权采样（弱项升权、强项降权但有 floor 保底）。
3. **可学习 > 答案对**：教学 CoT 逐步 worked、绝不单行断言；可验证 > 自然语言花样。
4. **（审计新增）质量信号优先、先审后用**：每个源先用自带标注（valid 旗标/reward/难度/correctness）过滤分层，再谈配比；任何新评测源先对训练池做 qhash 隔离。

---

## 1. 硬件与 base（已定，不变）

- `card1`=A800 80G(训练)；`card2,3`=A4000(异步 eval)；`card4-7`=A4000(RL rollout，DP 4 副本)；`card0` 备用。**必须 `export CUDA_DEVICE_ORDER=PCI_BUS_ID`**。
- 环境：`post_train/.venv`(uv, torch2.6/vllm0.8.5)；缓存与大文件在 `/data/zilu`。
- base：`/data/zilu/fastrl/checkpoints/qseek_digitsplit_base`（1.7B Llama，数字切分 tokenizer，=ctx16k 权重）。**不用** `latest_infer_*_hf_vllm`（合并 tokenizer，与逐位推演设计冲突）。

---

## 2. 路线图与当前位置

| 阶段 | 内容 | 状态 | 进入门禁 |
|---|---|---|---|
| **F1**·Foundation SFT v1（原阶段1+2 合并第一轮） | 118.9万混合池(含已知毒样本) | ⚫ **已废弃**(06-10 用户删除 checkpoint,重训为准;评测历史保留在 eval_tracking.md,acc 曾到 36.6%@5500) | — |
| **修复期**·数据重建 | 审计 P0/P1 修复 + 弹药库/heldout 重建 | ✅ **完成**(06-10 16:12,对账全中,见 rebuild_v2_log.md §三) | — |
| **F2**·Foundation SFT v2（原阶段2） | 干净数据 + 中文数学池,从 base 重训(**静态配比**,见 §4.2 注) | 🟢 **训练中**(06-10 16:25 起,126 万×2epoch≈9858 步/1.24B token,预计 06-11 晚收敛) | **P0 全清**（§4.1）✅ |
| **S3**·难度课程 SFT（原阶段3） | 易→难多轮课程，解题能力在此建立 | ⬜(超参草案见 §4.3,待 F2 终评定稿) | F2 收敛 + 终评像样 |
| **S4**·GRPO RL（原阶段4） | 同步 smoke→异步→LoRA-only 权重同步(改 verl) | ⬜ | S3 收敛 + RL 池修复 |

**F1 定性（2026-06-10 终版）**：F1 数据含已知毒样本（orca 错 boxed 8.3%、numina boxed{proof} 1.8万、svamp 泄漏），决策为 **F2 从 base 重训**；用户已直接删除 F1 checkpoint(基线对照取消)。F1 的价值留存为:评测流水线验证 + eval_tracking.md 的曲线/诊断(乘法弱/中文退化/扰动脆弱三短板,F2 数据已针对性补强)。

---

## 3. 数据资产（审计后口径）

**架构不变**：原始 17+ 源 → adapters 统一格式 → 两个全量弹药库（SFT 池/RL 池）→ 按阶段配方重切子集。修复动弹药库（真数据错），配比只动切片。

| 资产块 | 审计后可用量 | 关键点 |
|---|---|---|
| Compute_Cot 自产 | 39.4万(重切分后) | 逐题难度标签;train∩val/test 113 条穿透需重切 |
| 英文应用题(易) | orca 16.8万(修复抽取+丢5%低置信) + metamath GSM 段 24万 + gsm8k 0.7万 | metamath 抽取 100% 可靠 |
| 英文中段 | metamath MATH 段 15.5万(解放) + hendrycks MATH-train 0.75万(带解) + numina 易中源 ~50万(valid 过滤后) + tulutalk Math 29.6万(5档难度) | MATH test 边界已下载就位 |
| 英文难段 | numina 奥数系 ~28万(valid 过滤) + openr1 18.8万(**改用 R1 messages+correctness**) + deepscaler 4万 | openr1 13.8% 超 24k 字符需分桶 |
| **中文数学池(新建)** | **~25万**：ape210k chain→worked CoT 15-18万 + chinese-r1 math repos 3.4万(reasoning_content 包 think) + dynamics math_full 0.9万(去污染后) + dapo 中文 0.3万 | 中文缺口的解;高难段(~4千)先空着,不强求 |
| 通用防退化 | tulutalk(过滤后)/infinity/coig/dynamics(剔 reasoning_full)/no_robots/dolly(降权)/chinese-r1 通用段 | 上质量过滤器,中文占比 ≥40% |
| RL 池 | big-math 25万(solve_rate 连续难度) + dapo 1.7万 + deepscaler + gsm8k + compute_cot + ape210k 19万(中文) | orca gt 修复 + math_verify.parse 过滤后可用 |

**难度梯度结论（2026-06-10 已核实）**：英文 易~81万/中~95万/难~58万,梯度连续、标签齐全,**不再下载训练数据**；评测边界件 math-500 + math-hendrycks(train7.5k/test5k) 已就位。

---

## 4. 阶段详案

### 4.1 修复期（F1 跑完立刻做,预计 1-2 天）——**F2 的硬门禁**

P0（全部完成才能切 F2 数据）：
1. 修 orca 抽取（"末含数句 is/= 后数+分数整捕",验证 90%+）→ 重抽 boxed+ground_truth,重建双池受影响行。
2. heldout 剔 63 条 svamp 泄漏 + 7 条 compute_cot 穿透,svamp 曲线重算;`build_heldout.py` 加"对训练池 qhash 过滤"。
3. `build.py` EVAL_PATHS：+svamp、competition-math 行换 **math-hendrycks test(5k)**;`build_general.py` 接入 eval_hashes + 复用数学池 seen（消 7,256 条跨池双格式）。
4. numina 适配器：`problem_is_valid==Yes ∧ solution_is_valid==Yes` 过滤、剔 answer∈{proof,null,notfound}、source 字段纠错（难度按 source 分层,不再一刀切 hard）。
5. Compute_Cot 归一化 qhash 重切。

P1（F2 数据构建时一并落实）：中文数学池组建（§3 表）/ openr1 换 R1 字段 / MATH-train 解放入池 / 通用池过滤器（tulutalk safe∧st_reward≥0.75、剔 dynamics reasoning_full、dolly 降权）/ RL gt 过 math_verify.parse。

### 4.2 F2 · Foundation SFT v2（数学 CoT + 动态课程）

**目标**：算术执行力＋向应用题初步泛化＋中文不瘸腿。针对 F1 诊断的短板显式加密：**多位数乘法、中文表达、扰动鲁棒性**。

**配方（目标比例,总量 ~120-140万,≤8192 token）**：
| 块 | 占比 | 内容 |
|---|---|---|
| Compute_Cot | ~30% | 全留,乘法/高位数子源上采样 |
| 英文应用题(易中) | ~25% | orca(修复后)+metamath(GSM 优先,"求x"类 FOBAR/SV 降权)+gsm8k(<<>>转 worked 步骤)+MATH-train |
| **中文数学** | **~15%** | 新建池 ~25万全投 |
| 竞赛陪练 | ~5% | numina(过滤后)/deepscaler,少量见难题不学崩 |
| 通用防退化 | ~25% | 过滤后通用池,中文 ≥40%,类目配比压 Math/Code |

**动态课程循环（机制不变,v1 §4.3）**：训练 N step → A4000 测 per-source acc → EMA → `w[s]=max(floor, g(1-acc_ema[s]))` → 喂采样器。补两条审计教训：①小空间 source（119 个唯一题<1500）设最大贡献上限;②**课程信号前置条件 = per-source 细粒度 eval 建好**（compute_cot val 每 source 采 N,修复期一并做）。

> **06-10 实际执行注**:F2 以**静态配比**跑全程(固定 parquet,reweight 切片,seed 复现)。动态课程未启用——用户拍板**不建** per-source 细粒度集(heldout 每子源 ~7 条作过程评测够用);子源级弱项诊断改由阶段终评的 **cc-reserved**(Compute_Cot 保留集每子源 20 条,`eval/final_eval.py`)承担。超参:lr 1e-5 恒定/wd 0/clip 1.0/batch 256(动态打包 24,576 tok/卡)/2 epoch=9,858 步≈1.24B token/A800 单卡,~30h。

**过程评测**：heldout v2(修复版) 每 500 step + probe 中英对照（中文退化监控加密）+ 通用对话抽查。

### 4.3 S3 · 难度课程 SFT（解题能力在这里建立,RL 只放大不创造）

**难度信号（全部现成,不再自评）**：big-math `llama8b_solve_rate`（连续值,主轴）/ MATH level 1-5（metamath 可用 original_question 回填）/ tulutalk difficulty 5 档 / openr1 correctness_count / numina source 分层。

**课程轮次（示意,按过程评测调）**：R1 易:中:难=60:30:10 → R2 35:40:25 → R3 20:40:40;Compute_Cot+中文池小比例贯穿防遗忘。难题一律带解答：openr1 用 correctness 过滤的 R1 trace（≤16k 分桶,F2 后上下文可放宽）、numina/deepscaler 带 solution 子集、hendrycks train。

**⚠️ 配比口径(06-11 定,勿按条数直配)**：池内 token 量极度偏斜——ot3 均长 46k 字符/openr1 9.9k vs numina 969/ape210k 123;ot3+openr1 条数仅占 16% 但 token 占 80%+。"易:中:难"若按**条数**配 60:30:10,token 口径实际坡度可能已是 ~25:35:40(难题全是长轨迹,权重自动放大)。**S3 reweight 时必须打印 token 口径分布人工确认坡度**,或直接按 token 预算配比。另:长 CoT 几乎全英文(中文长思考仅 zhr1 2.6 万条、均长 2k,为英文轨迹的 1/5)→ S3 盯 cmath 曲线,若涨幅明显落后 gsm8k = 中文思维深度不足,对策:zhr1 拉满贯穿 / 用阶段模型自蒸馏中文长轨迹。

**评测**：策略熵 + Pass@1/Pass@8 + 完整 IO dump（A4000 异步）;终评标尺切 **MATH-500 分级** + gsm8k/cmath/svamp(修复版)/GSM-Plus。

**超参与训练量(06-10 草案,F2 终评后定稿)**：
- **热启 F2 最终 checkpoint**(不从 base);上下文 8k→**16k**(openthoughts3 38k 条/openr1 全量轨迹进场)。
- **lr 5e-6 + cosine 衰减**(热启精调,步幅减半防冲掉 F2;顺带解决 F2 恒定 lr 尾段晃动)。
- **每轮 30-40 万样本 × 1 epoch**,三轮合计 ~100 万样本 / 1.5-2B token(长 CoT 进场,均长大涨)。单轮不刷多 epoch(防背题),轮间按过程评测调比例再开下一轮。
- **轮次推进判据**:heldout acc 涨 + 策略熵未塌 → 按计划加难;某轮 acc 掉头 → 难度比例回撤。
- **F2 终评对定稿的影响**:MATH-500 level1-2 已好 → R1 的"易"压到 <50% 直接加难;否则按 60:30:10。

> **06-12 R1 实际执行注**:F2 终评(见 eval_tracking.md F2 终评节)后用户拍板**不重做数据**,直接用现有弹药库切片;生成器扩 schema(数列求和/三角/中文指对)降级为触发式——R1 跑完看 cc-reserved 对应子源动没动再决定。切片 = 新脚本 `data_pipeline/reweight_s3.py`(难度归一:easy/medium/hard ∪ hendrycks Level ∪ openr1 cc=N;**token 预算配比**,单源 ≤40%/桶,compute_cot ≥10% + 中文数学 ≥15% 贯穿,通用 15% 内中文 ≥62%)。产出 `train_sft_s3r1_16k.parquet`:99.2 万条 / 0.451B token;**easy 桶供给见顶**(全池易题只有 155M token),实际坡度=条数口径 60:29:11、token 口径 43:44:13(可接受,cap 压住了长轨迹)。超参:热启 `global_step_9858_hf` / 16k ctx / lr 5e-6 cosine(min_ratio 0.1, warmup 2%) / 1 epoch=3874 步 / save_freq 200 / A800。06-12 03:45 起训,11.7s/it,ETA ~12.5h(预算 20h);step1-3 显存 51G/MFU 0.49/loss 0.72-0.78 健康。启动=`SFT/run_s3r1_sft.sh`,评测 watcher 已挂 card2,3(`logs/async_eval_s3r1.log`)。

**S3 数据补洞(06-10 发现,勿忘)**:全池**没有中文"简单计算+干净 CoT"的指对题**——裸算 log/指数只有 compute_cot exp_log 657 条且 100% 英文;中文源(zhr1)的 log 题全是函数/定义域/取值范围长题。实证后果:F2 step500 对中文探针"计算 log_2(8)+2^3"产生模板坍缩输出(内容借英文 logarithm_laws 模板借串成 8×8=64,语体借 zhr1 口癖空转),heldout `高考风·指对运算·zh` 得 0。**S3 动作**:compute_cot 生成器补①exp_log 等偏高中 schema 的**中文变体**②**跨规则混合运算模板**(log+幂/log+根式);起训前先盘全 schema 中英分布定补量。复测探针:log_2(8)+2^3(中文问)在 step 2000/5000/终评跟踪模板坍缩是否消退。

### 4.4 S4 · 异步 GRPO（Verl + LoRA）

- **数据**：修复后 RL 池（§3 表）;solve_rate 选带（如 0.1-0.7 区间,太易无梯度太难全零）;判分 reward.py 已审计通过。
- **奖励判定器(06-10 定)**:**不用 verl 内置** reward_score(只认其已知数据集),走 `custom_reward_function` 钩子接**我们自己的** `data_pipeline/reward.py::compute_reward`(RL 池每行带 `reward_model:{ground_truth,style}` 即为此设计;底层数学等价用 math-verify/sympy)——保证 SFT 自检/过程评测/终评/RL 奖励四方同一套判定。接线只需写薄 wrapper 适配 verl 签名 `(data_source, solution_str, ground_truth, extra_info)`。
- **⚠️ S4 起训前判定器加固三件事(06-10 自检结论,勿忘)**:
  1. **RL 池 gold 自检**:155 万 RL 池还没跑过 `scripts/selfcheck_gold.py`(06-10 只验了 SFT 池,11 源 3,300/3,300 全过);
  2. **野生采样压测**:用 F2/S3 模型在 RL 题上 T=0.8 采一批,人工抽读"判 0 但像对"(假阴性→浪费采样)与"判 1 但可疑"(假阳性→**reward hacking 入口,重点防**)两端——gold 自检只证明标准写法判得对,RL 面对的是野生写法;**另统计"判 1 但推理不自洽"(unfaithful CoT)率**——实例:F2 step1000 在 gsm-plus 上 `48/11=4` 硬接对答案(应为 48/12,已查实非背题:训练池无此题及近亲,gsm8k test 黑名单封死),outcome-only reward 会奖励这种胡编过程,若占比高需考虑过程一致性辅助项或单列跟踪指标;
  3. **已知短板处置**:mcq 字母提取 `\b([A-D])\b` 偏松、exact_match 小写化+去空白有撞对空间——压测露馅则收紧(只认 boxed 内字母)或把这两类 style 从 RL 配方剔除只留 SFT(占比小,主力 compute_cot/gsm8k/math_verify 数值路径是硬的)。
- **路线不变**：同步版 smoke（A800 训练 + card4-7 vllm 采样,跟踪采样/训练/权重同步三段耗时）→ 定超参 → 异步若干轮验证 → 改 verl 源码 **LoRA-only 权重同步**。
- **防泄漏**：math-beyond 与 DAPO/DeepScaleR 重合题在评测侧剔除。

---

## 5. 评测体系（汇总）

| 用途 | 标尺 | 备注 |
|---|---|---|
| 过程监控(F2/S3) | heldout v2 修复版：compute_cot 1848 / gsm8k 500 / svamp 237 / GSM-Plus 798 / cmath 200 / comp-math(改抽自 math-500) + probe | 窄/对齐/低方差;svamp 修复前读数作废 |
| 课程信号 | ~~per-source 细粒度集~~(06-10 用户拍板不建) → 终评 cc-reserved 的 per-source 读数 | 阶段间调配方,不做训中动态 |
| 终评(每阶段末) | `eval/final_eval.py`:**cc-reserved**(CC 保留集每子源20,基本功) + MATH-500(level 分级) + gsm8k + cmath(三维) + GSM-Plus(7 类扰动) + svamp;mgsm-zh 未下载暂缺 | Pass@1/8;完整 dump;已冒烟 |
| 中文监控 | cmath + probe 中英对照 + (评测侧)mgsm-zh 250 | mgsm 是 gsm8k test 翻译,**永不进训练** |

**隔离铁律**：EVAL_PATHS = 上表全部题面 ∪ math-hendrycks test;数学池与通用池共用 seen;新评测源先过训练池 qhash。

---

## 6. 里程碑

① F1 完训+终评（06-11）→ ② 修复期 P0 清零+弹药库/heldout 重建 → ③ per-source 细粒度 eval 接通 → ④ F2 重训(动态课程首跑) → ⑤ S3 课程两轮+Pass@k 跟踪 → ⑥ S4 同步 smoke → 异步 → LoRA-only 同步改造。

**风险与对策**：泄漏复发(新源先隔离,铁律§5) / 小 source 过采样(上限) / 长 CoT 过早混入学崩(S3 才放开 16k,F2 维持 8k) / 中文池质量(chain→CoT 转换器先抽检 200 条人审再批量) / 卡号错配(PCI_BUS_ID)。

---

## 附：关键接口与路径（交接用）

- **启动训练**：F2 = `bash SFT/run_foundation_v2_sft.sh [SAVE_PATH]`（默认存 `/data/zilu/fastrl/checkpoints/sft_foundation_v2`;参数详见 `SFT/README.md`）。
- **启动异步评测**：`.venv/bin/python -u -m eval.async_eval --ckpt-dir <ckpt> --gpu-candidates 2,3 --watch`（产出 `<ckpt>/eval_dumps/`：metrics.jsonl / step_N/heldout.{jsonl,md} / tb）。
- **阶段终评**：`.venv/bin/python -m eval.final_eval --ckpt <ckpt>/global_step_N --gpus 2,3`（6 benchmark,Pass@1/8,产出 `<ckpt>/final_eval/summary.{md,json}`;冒烟 `--benchmarks svamp --limit 8 --k 2`）。
- **数据管线**：`python -m data_pipeline.build`（数学弹药库）→ `python -m data_pipeline.build_general`（通用池）→ `python -m data_pipeline.reweight_sft`（阶段配方切片）→ `python -m data_pipeline.to_verl_parquet`。
- **进度查询**：tmux 会话 `sft`（训练 console）;`tail -f <ckpt>/eval_dumps/async_eval.log`（评测）;tensorboard `tensorboard_log/`+wandb 项目 `qseek-posttrain`。
- **checkpoint**：base 与各阶段 ckpt 全在 `/data/zilu/fastrl/checkpoints/`（每 500 step 一存,每个 ~21G,⚠️ 18 个 ≈380G 注意清理）。
- **文档导航**：审计报告 `docs/data_audit_report_20260610.md` / 架构 `docs/data_audit_and_architecture.md` / 评测追踪 `docs/eval_tracking.md` / 卡片 `docs/traindata/`。
