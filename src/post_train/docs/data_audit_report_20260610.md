# 数据全面审计报告（2026-06-10）

> 触发：foundation SFT 训练中抽样发现 orca boxed 错误 → 全面重审「数据处理 / 数据集使用 / 原始质量 / 未采纳资产」。
> 方法：管线代码逐行审 + 全量池/训练 parquet 量化核查 + 4 路并行子代理逐源抽样原始数据。
> 关联：[data_audit_and_architecture.md](data_audit_and_architecture.md)（架构）、[eval_tracking.md](eval_tracking.md)（评测）、`docs/traindata/*`（数据集卡片,本次已逐卡更新审计节）。

---

## 〇、结论速览（按严重度）

| # | 级别 | 问题 | 量级 | 影响面 |
|---|---|---|---|---|
| 1 | 🔴 P0 | orca 答案抽取 bug：boxed/ground_truth 取到中间结果 | ~9.9万条(orca 的 56%) = 训练集 8.3% | SFT 教"推理与答案脱节"；RL 池奖励错答案 |
| 2 | 🔴 P0 | SVAMP 评测泄漏：heldout 300 题中 63 题(21%)逐字在训练集(经 orca) | 63/300 | v2"泛化"读数虚高，svamp 曲线不可信 |
| 3 | 🔴 P1 | numinamath `\boxed{proof}` 占位答案进入当前训练集 | 18,207 条(其切片的 20%) | 教模型"难题→吐 proof 占位" |
| 4 | 🟠 P1 | build_general **无评测隔离**(结构洞) | 已实测漏 4 条 MATH+1 条 cmath | 通用池任何重切都可能引入泄漏 |
| 5 | 🟠 P1 | competition-math 基准目录=全量 MATH 12.5k(含 train) → 过度排除 | ~12.9万条被误杀(metamath 7.5万为主) | MATH-train 难度阶梯资产全损失 |
| 6 | 🟠 P2 | Compute_Cot 切分穿透(归一化 qhash) | train∩val 63 / train∩test 50 | heldout 7 条已命中；"无泄漏"声明不实 |
| 7 | 🟠 P2 | 跨池同题格式冲突：同一题数学池教 think+boxed、通用池教纯对话 | 7,256 条(infinity 7,247) | 与"风格冲突"诊断同根 |
| 8 | 🟡 P2 | openr1 解答结论与 boxed 脱节 14.4% + 763 条空 think；correctness 字段未用 | 5,409/37,567 | SFT 教学价值打折 |
| 9 | 🟡 P2 | numina/infinity-math RL ground_truth 形态脏(`=b==\frac{1}{3}`、截断括号) | 抽样即见 | math_verify 解析失败→奖励噪声 |
| 10 | 🟡 P3 | 计划承诺未实现：tulutalk 按 reward 筛/类目均衡、openr1 按 correctness 筛 | — | 配比质量低于计划描述 |

（子代理逐源审计与未采纳资产盘点结论见 §三/§四,卡片已同步更新）

---

## 一、管线代码审计（data_pipeline/）

### 1.1 🔴 orca 答案抽取 bug（adapters.py:70-82, `_num_from_text`）
- 回退正则 `re.search(r"(?:answer is|答案是|equals?|=)\s*...", t[-200:])` 在**末 200 字符取第一个匹配**；orca 解答末尾常是连串中间等式 → 抓中间结果。
- 实测(train_sft_foundation_8k)：176,098 条 orca 中 **99,268 条(56.4%) boxed 数值不出现在解答结论句**；按"末两行"口径 42.9%。随机人检全部实锤(结论 8723→box 14696 等)。
- `ground_truth` 与 boxed 同源同错 → **RL 池 orca 答案同样错**。早前"97.9% 一致已澄清"系循环验证(同一抽取自比),已在 architecture 文档作废。
- 对照组：metamathqa 同口径仅 0.7% 不一致(规范 "The answer is: X" 结尾) → bug 仅伤 orca(唯一走回退正则的源)。
- **修法**：回退取**最后一个**匹配 + 优先 "So/Therefore … is X" 结论句模式；修后重抽 orca 行,SFT/RL 池同步重建。

### 1.2 评测隔离审计（build.py EVAL_PATHS + build_general.py）
- ✅ gsm8k-test / cmath / gaokao / math-beyond / competition-math 隔离生效(数学池 leak 计 129,018)。
- 🔴 **SVAMP 不在隔离清单**(v2 heldout 2026-06-10 新增源,建池时未回补)：svamp 基准全量 300 题中 **63 题逐字在 orca 训练行里**。GSM-Plus 0 重合(扰动改写,天然安全)。
  - 短期修：heldout 剔除这 63 题(svamp 余 237);长期修：EVAL_PATHS 增加 svamp(+今后 heldout 任何新源),重建时丢 63 条 orca。
- 🟠 **build_general.py 完全没有 eval_hashes 检查**：通用池实测含 3 条 MATH(tulutalk)+1 条 MATH(infinity)+1 条 cmath(chinese-r1)。量小但是结构洞,通用源里混着大量数学题(见 1.4),任何重切都可能再漏。
- 🟠 **competition-math 基准目录是全量 MATH(12,500,单 train split)**：隔离按它做 → MATH **train** 7.5k 题及其增强后代全被排除(metamath leak 75,081、infinity 22,137、big-math 9,367、deepscaler 7,715、bespoke 7,644、ot3 6,133...合计 ~12.9万)。评测只用其中 200 题。**建议**：基准侧换成标准 MATH test(或 MATH-500),把 MATH-train 题面从禁用名单解放,阶段3 难度阶梯急需这批数据。
- 🟠 **Compute_Cot "无泄漏切分"不实**：归一化 qhash 下 train∩val=63、train∩test=50、val∩test=6(切分用的精确去重,没做空白/大小写归一)。heldout 已命中 7 条。修：clean 数据重切或 heldout 剔除。

### 1.3 去重审计
- 数学池内 first-wins 全局去重 ✅(SOURCE_ORDER 优质源优先;metamath dup 18.7万/numina 35.9万/ot3 27.7万属预期的跨源同题)。
- 🟠 数学池与通用池 **seen 集合独立** → 跨池同题 7,256 条(infinity 7,247/tulutalk 7/chinese-r1 2)：同题两种 gold 格式(think+boxed vs 纯对话)同时在训,直接制造"答数学题要不要 think"的格式冲突。修：build_general 复用数学池 qhash 集合(数学池优先)。

### 1.4 格式与字段审计
- `wrap_think_boxed` 清洗逻辑总体可靠(R1 标记/gadget/<<>>/#### 均剥);⚠️ `^#{2,}` 行删除会误伤解答内的 markdown 小标题(轻微)。
- `extract_boxed` 平衡括号 ✅;`make_prompt` 中英 system 按题面自动选 ✅。
- RL 池 ground_truth 形态抽检：compute_cot/gsm8k/orca(除错值)/metamath/dapo 干净;🟡 numina 见 `=b==\frac{1}{3}`、`3,1,=2` 等脏答案;infinity-math 见截断 LaTeX(`\frac{m_{\text{house}}}{52` 缺右括号)。建议入池时跑一遍 math_verify.parse 可解析性过滤。
- reward.py 验证器逻辑审计通过(boxed→####回退、数值容差、sympy 等价、MCQ 字母归一)。

### 1.5 配比执行与计划落差
- FOUNDATION_CAPS 之外混入两个源未注释：chinese-deepseek-r1-distill 8万(默认全留)、infinity-math 3,918(默认全留)。
- openthoughts3 名义 cap 5万,实际 8k 长度过滤后仅 **1,071** 条在训(38,436→1,071;超长 trace)。bespoke 3,494→3,057。**当前训练集里"长推理风格"事实上缺席**(设计上留给阶段2-3,但卡片/注释易误读)。
- 计划承诺未实现:① tulutalk "按 mt_instruct_reward≥阈值筛、按 task_category 均衡"(实际纯随机 8万);② openr1 "筛 correctness_count≥2"(实际全收,适配器没读 correctness 字段);③ "各集类目均衡/上采样小类"。

### 1.6 池→训练集对账（manifest vs parquet）
| 源 | 入池(SFT) | cap | 8k 过滤后在训 | 损耗主因 |
|---|---|---|---|---|
| compute_cot | 393,954 | 全留 | 393,954 | — |
| orca | 176,899 | 全留 | 176,899 | — |
| metamathqa | 132,770 | 110,000 | 110,000 | cap |
| numinamath-1.5 | 504,136 | 90,000 | 89,995 | cap+长度 |
| openr1 | 188,685 | 50,000 | 50,000 | cap |
| openthoughts3 | 38,436 | 50,000 | **1,071** | **长度** |
| bespoke | 3,494 | 全留 | 3,057 | 长度 |
| deepscaler | 5,750 | 全留 | 5,750 | — |
| infinity-math | 3,918 | (未列,全留) | 3,918 | — |
| gsm8k | 7,473 | 全留 | 7,473 | — |
| 通用 7 源 | 486,975 | 各 cap | 346,791 | cap |
| **合计** | — | — | **1,188,908** | — |

---

## 二、评测侧连带影响（heldout v2 读数校正）

- svamp 曲线(8.3%→23.7%)含 21% 记忆题,**真实泛化水平需剔除 63 题重算**(预计读数下修)。
- compute_cot 曲线影响微(7/1848=0.4%)。
- gsm8k/GSM-Plus/cmath/competition-math 读数不受本次泄漏影响。
- 行动:eval/build_heldout.py 增加「对训练池 qhash 过滤」步骤,任何未来 heldout 源默认过一遍。

---

## 三、逐源原始数据审计（4 路子代理全量抽样,细节已写入各数据集卡片"审计补充"节）

### 3.1 数学源

| 源 | 原始规模 | 核心发现 | 判定 |
|---|---|---|---|
| orca-math-200k | 200,035 | 只有 question/answer 两字段,无独立 gold;"answer is"收尾仅 0.3% → 管线正则必然失效;**修复策略已验证**(末含数句取 is/= 后数+分数整捕,90%+ 准确) | 🔴 修抽取 |
| metamathqa | 395,000 | "The answer is:" 100% 覆盖,抽取零风险;`original_question` 100% 存在未用(可回填 MATH level+防泄漏);MATH 系 39.2% 被全量隔离误杀 75,081 | ✅ 干净,解放 MATH-train |
| gsm8k | 7,473+1,319 | 干净;`<<a+b=c>>` 步骤标注 98.7% 覆盖被管线洗掉——对"乘法弱"短板是对症监督素材 | ✅ 可加值 |
| numinamath-1.5 | 896,215(全量,卡片旧注 1/3 过时) | **`problem_is_valid` 非 Yes 4.1%(36,859,含波兰残题)、`solution_is_valid` 非 Yes 另 ~46k 全未过滤**;answer 无效 16.4%;字段误用(problem_type 当 source);MCQ 16.3%;aops_forum 32,429 行空解答 | 🔴 加过滤+字段纠错 |
| openr1-math-220k | 225,129 | **管线错用人写 `solution`(52.6% 无 boxed)而非 R1 `generations/messages`+correctness 标注**;87.3% 题有验证通过的 R1 轨迹;13.8% 轨迹超 24k 字符 | 🔴 换字段 |
| openthoughts3-1.2m | 1,200,000(math 850k) | math 子集 **~70% 回答截断**(think 闭合 31.9%);中位 48k 字符,8k/16k 上下文都难用;difficulty 全 None | ⚠️ 阶段3 前不投入 |
| bespoke-stratos | 16,710 | **32.3% Python 代码题 + 4.9% 非数学 QA**;markup 是 `<\|begin_of_thought\|>` 非 think;无截断 | ⚠️ 按前缀分流 |
| calc-ape210k | 195,179(+val/test 各 4,867) | **equation 100% 可 eval(98.6% 与答案吻合)、chain 95% 自带分步计算链 → 19.5万中文 worked CoT 唾手可得**;当前只当 RL 题面用 | 🌟 最大中文资产 |
| big-math-rl-verified | 251,122 | `llama8b_solve_rate` 双峰(易/难各 20%)=现成课程难度标签未用;orca_math 重叠 33%;cn_k12 已英译 | 🌟 课程标签 |
| dapo-math-17k | 17,398 | gt 100% 纯整数;**中文题 3,282(18.9%)**;模板前缀已正确剥离 | ✅+中文子集 |
| deepscaler | 40,315 | 18.3%(7,391) 有 worked solution 已正确入 SFT;928 重复 | ✅ 基本正确 |
| cmath / math-beyond / calc-gsm8k | 1,698 / 181 / 8.7k | cmath 无 train split,维持纯评测(grade/步数/位数标注 eval 未用全);math-beyond 极难探针;calc-gsm8k 冗余 | ✅ 定位确认 |

### 3.2 通用源(质量信号浪费是主旋律)

| 源 | 原始规模 | 核心发现 | 判定 |
|---|---|---|---|
| tulutalk-annotated | 808,322 | = tulu-3 子样本+SmolTalk 的重标注版;**全部标注未用**(st_reward/difficulty/task_category/llama_guard);Math 类占 36.6%(29.6万,带 5 档难度);unsafe 1.9% 在训 | 🔴 上过滤器 |
| infinity-instruct | 本地 90万(**仅各子集 shard 0,全集 1/8~1/35**) | label.cate_ability 比 boxed 启发式准;7M_domains/math 本地 10万纯数学;**zh boxed 仅 9 条,填不了中文**;跨池同题 7,247 条 | 🔴 补下载+换过滤 |
| chinese-r1-distill | 110,000 | **`reasoning_content`(R1 think,100% 非空)全量闲置;math repos 36,945,score≥8 验证过的 34,327 条=全场最优中文数学 CoT**;score<7 过滤仅筛 2.9%;repo_name 领域标签未用 | 🌟 移数学池 |
| dynamics | 82,396 | math_full 11,501(79% 中文带步骤,MATH/GSM 翻译,**需跨语言去污染**);**reasoning_full 12,751 全是裸字母 MCQ=空区病征来源,应剔除** | ⚠️ 分流处理 |
| coig-cqia | 44,694 | 子集质量复核 OK;exam 数学仅 57 条;chinese_traditional 有脏数据 | ✅ 维持 |
| no_robots / dolly | 9,948 / 15,011 | no_robots 优;dolly 中等(短答/过时),"过滤后用"未实现 | ✅ / ⚠️ 降权 |
| flan | 已下载 3.5G 未接 | cot_gsm8k 9.6k+aqua 3.5k 可选;dialog/niv2 主体陈旧不接 | 可选小块 |

## 四、已下载未采纳资产盘点(/data/zilu/fastrl/data/train)

| 数据集 | 规模 | 判定 | 理由 |
|---|---|---|---|
| numinamath-cot | 859,494 | **部分接** | 题面 70% 与 1.5 重合,但 solution 97.3% 含 boxed 且全短(p99 3.3k 字符)——按题面 hash join **回填 1.5 的缺/坏解答**;独有 synthetic_amc 62k;cn_k12 同为英文翻译 |
| tulu-3-sft-mixture | 939,343 | **部分接** | 数学子集 ~33.4万;`open_math_2_gsm8k` 增量 ~2.9万(boxed 全有/短/干净,最值得接);numinamath_tir 是代码穿插风格(冲突,慎);通用段已被 tulutalk 覆盖 |
| openthoughts-114k | 113,957 | 不接 | 被 ot3-1.2m 覆盖;p50 16k 字符超长;无 domain 列需自分类 |
| mgsm | en/zh 各 258 | **评测侧接** | zh=gsm8k **test** 前 250 题人工中文翻译 → 作中文小学评测(与 cmath 互补)+中英同题探针;**绝不能进训练** |
| opus-100-enzh | 1,000,000 | 不接 | 碎句翻译对,非指令格式 |
| sharegpt / logicot / finemath | — | 未下载 | download 脚本注册了但本体不在本地 |

**评测资产**(fastrl/data/benchmark,13 项):gsm8k/gsm-plus/competition-math(全量 12.5k!)/svamp/cmath/gaokao×2/cmmlu/mmlu-stem/bbh/gpqa/ifeval/zebralogic。中文数学评测可再加 mgsm-zh。

## 五、修复行动清单(按优先级)

**P0(阶段2 数据构建前必须)**
1. 修 orca 抽取(`_num_from_text`→"末含数句 is/= 后数+分数整捕",低置信 ~5% 丢弃);重抽 orca 全部 ground_truth+boxed,重建 SFT/RL 池受影响行。
2. heldout 剔除 63 条 svamp 泄漏题 + 7 条 compute_cot 穿透题,svamp 曲线重算;`build_heldout.py` 增加"对训练池 qhash 过滤"步骤。
3. build.py EVAL_PATHS 增加 svamp;build_general.py 接入 eval_hashes 隔离 + 复用数学池 seen(消跨池同题双格式)。
4. numinamath 适配器:按 `problem_is_valid==Yes ∧ solution_is_valid==Yes` 过滤、剔 answer∈{proof,null,notfound}(或单列 sft-only 并不再拼 boxed{proof})、source 字段纠错。

**P1(阶段2 构建时落实)**
5. 中文数学池组建(架构文档 P0 缺口,本次找到的真实库存):**ape210k chain→中文 worked CoT(15-18万)** + **chinese-r1 math repos 34,327(用 reasoning_content 包 think)** + dynamics math_full ~9k(跨语言去污染后) + dapo 中文 3,282。
6. openr1 适配器改用 `messages`/`correctness_math_verify=True` 的 generation;correctness_count 当难度代理。
7. competition-math 基准侧换 MATH test-only,解放 MATH-train 增强样本 ~12.9万(metamath 7.5万为主)——阶段3 难度阶梯的主力补给。**(2026-06-10 资产已就位**:已下载 `benchmark/math-500`(500,评测用,带 level/subject)与 `benchmark/math-hendrycks`(官方 train 7,500/test 5,000 分 split);实测 MATH-500 ⊆ 官方 test、旧 competition-math 12.5k 中 7,498 条属 train 段可解放。执行时:EVAL_PATHS 的 competition-math 行换成 math-hendrycks 的 **test** 段(5,000)+heldout 的 comp-math 抽样源改为 math-500;hendrycks train 段 7.5k 带 solution 可直接当干净 SFT 源入池。)
8. 通用池上过滤器:tulutalk(safe ∧ st_reward≥0.75,类目配比压 Math+Code)、剔 dynamics reasoning_full 裸字母 MCQ、dolly 降权。
9. Compute_Cot 重切(归一化 qhash 全局去重,消 train∩val/test 113 条)。

**P2(阶段3 前)**
10. 接入 tulu-3 open_math_2_gsm8k(~2.9万)+numinamath-CoT 解答回填/synthetic_amc;big-math solve_rate + tulutalk difficulty 接入课程调度;补齐 infinity 下载(若需扩英文数学)。
11. RL 池入池前过 math_verify.parse 可解析性检查(numina/infinity-math 脏 gt)。
12. mgsm-zh 接入评测侧;cmath eval 增加按 reasoning_step/num_digits 切片。

**当前在训 run 的处置建议**:不中断(epoch 2 还剩 ~11h,本轮当格式对齐+初步能力);**阶段2 必须在修完 P0 后用重建的数据重训**,届时 orca 毒样本/proof 占位/泄漏全部出清。
