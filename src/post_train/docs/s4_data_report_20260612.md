# S4(最终阶段难题退火 SFT)数据构建报告 · 2026-06-12

> 任务来源:整合 MATH/高考/考研难题做最后一轮定向 SFT。本报告覆盖:池清洗、
> S4 训练集构建、heldout v2(高考档观测)、外部补源结论、RL 保留题库覆盖审计。
> 前置:判分器修复见 `docs/reward_verifier_fix_20260612.md`(本轮所有验证均用修复后版本)。

## 1. 池清洗(clean_s4.py)—— 只去"错",不去"杂"

三条机械规则:R1 证明题(Prove/证明/求证)、R2 gold 损坏(未知 LaTeX 命令/
花括号不配平/多字母变量糊/超长;compute_cot 为自家明文 DSL 豁免)、
R3 gold_response 与 ground_truth 经 verifier 判不一致。
探测器在已知坏样本 8/8 召回、26 例合法答案零误杀(校准过程在脚本 docstring)。

| 池 | 清洗前 | 清洗后 | 丢弃 |
|---|---|---|---|
| SFT 池 → `train_sft_s4clean.jsonl` | 1,436,749 | 1,389,982 | 3.3% |
| RL 池 → `train_rl_s4clean.jsonl` | 1,557,905 | 1,500,915 | 3.7% |

重点源丢弃率:numinamath **10.5%**(garbage 2.2 万 + proof 1.5 万)、openr1 5.2-6.8%、
big-math 8.5%、chinese-r1 1.8%;orca/ape210k/metamathqa/gsm8k ≈ 0%(干净)。
分源明细见 `*.stats.json`。

## 2. S4 训练集:`parquet/train_sft_s4_anneal_16k.parquet`

**291,929 条 / 0.133B token**,配方 `reweight_s3 --ratio 12:38:50 --general-frac 0.18
--math-pool <清洗池> --exclude-hashes eval/heldout_v2_exclude.txt`(已验证与 heldout 0 重合)。

与 S3 对比(token 口径):hard 10.4% → **38.0%**,medium 34.9% → 32.9%,
easy 34.4% → **10.4%**,general 20.3% → 18.8%。hard 配额(50%)受池内供给限制
实际到 38%——已是清洗池 hard 全量吃进(高考档主力 numinamath-hard 9.8M tok、
openr1-hard 8.7M、bespoke 5.6M、MATH 原题 0.8M、deepscaler 1.6M)。
规模约为 S3(451M tok)的 30%,按 S3 速度估训练 **约 4 小时**(A800 单卡)。
基座建议:s3r1 final(当前 77%,预计今晚完成)。

## 3. 过程评测 heldout v2:`eval/heldout_v2.jsonl`(4,804 条)

= v1(3,881,能力分解版,指标连续可比)+ **高考/考研观测层 923 条**:

| source | n | 来源 | 判分 |
|---|---|---|---|
| gaokao-mathqa | 347 | AGIEval 高考选择(外部基准) | mcq |
| gaokao-cloze | 116 | AGIEval 高考填空(外部基准) | math_verify |
| cnk12-heldout | 200 | 清洗池 cn_k12 留出(in-dist 吸收度) | math_verify |
| zhmath-heldout | 160 | EduChat-Math 120 + applied_math 40 | math_verify |
| kaoyan-heldout | 60 | zhr1:kaoyan(池中仅 338,余 278 进训练) | math_verify |
| advmath-heldout | 40 | zhr1:Advanced-Math hard | math_verify |

用法:`python -m eval.async_eval --ckpt-dir <S4目录> --heldout eval/heldout_v2.jsonl`。
注意:①高考档建议 `--max-new-tokens 1024`(默认 640 对难题可能截断);
②留出题 hash 在 `eval/heldout_v2_exclude.txt`,**后续任何训练切片都必须带
--exclude-hashes 隔离**;③v2 与 v1 的 v1 部分完全相同,历史曲线可续接,但判分器
修复(06-12)前后的数字不可直接对比;④kaoyan 子集混有选择题(gold 为字母,可判)。

## 4. 外部补源结论:无合格公开源,缺口如实记录

- 搜索 HF(gaokao/kaoyan/chinese math 等多组关键词 + 直探已知 ID):
  无"高考/考研难度 + 中文 + 带可验证解答"的合格 SFT 源(候选多为小学题翻译、
  院校政策 QA、预训练语料,GAOKAO-Bench 属评测不宜入训)。
- `Azure99/blossom-math-v4`(1 万中文题)已下载到 `/data/zilu/math_sft_raw/blossom-math-v4/`
  但**未采用**:小学应用题为主,与 ape210k/chinese-r1 重叠,抽样见解答自相矛盾样本。
- 本轮实际最大的增量来自**池内回收**:openr1 全量在清洗池中有 16.1 万条
  (S3 配方只吃进 1.77 万),S4 的 medium/hard 桶已显著加大其占比。
- **考研档供给仍是硬缺口**(全池仅 338 条):要补只能自建(爬取+蒸馏)或等内部渠道。
- **决策(2026-06-12,zilu)**:英文质量够用即可接受——高考档以英文 cn_k12(清洗后
  14.4 万,题解质量抽查合格)为主力,不阻塞 S4/RL;中文缺口不再专门补源。
  验证手段:heldout v2 同时有英文层(cnk12-heldout)和中文层(gaokao-cloze/mathqa),
  S4 训完直接对比两层曲线,看英文训练能否迁移到中文高考题;若迁移差,再回头议。

## 5. RL 保留题库覆盖审计(train_rl_s4clean,150 万条)

**题型/语言覆盖:广。** 算术 DSL(compute_cot 39 万)→ 小学应用题(orca/ape/gsm8k 34 万)
→ 中学(metamathqa/cn_k12 等 ~40 万)→ 竞赛(deepscaler/dapo/olympiads ~6 万);
判分 style 三种全有;中文占比 14%(ape210k 18 万 + chinese-r1 2.6 万)。

**难度覆盖:标签全,但"模型可用带宽"集中在中低段。** 结合终评 pass@8 实测:
gsm8k 档(p1 16%→p8 43%)与 cmath/svamp 档在带宽内;竞赛 hard(MATH L3-5/
AIME/dapo)对当前 1.6B 基本是全错组,GRPO 零梯度,白费采样。
big-math 自带 solve_rate 标签(Llama-8B 口径):0.1-0.9 带宽存量 **3.6 万条**,
是现成的 RL 课程骨架(需用自家模型重标定,口径会整体左移)。

**两个需要处理的风险:**
1. **可猜 gold 比例高**:gold 为 0-10 小整数或单字母的占比——chinese-r1 40.7%、
   openr1 38.6%、metamathqa 36.1%、dapo 30.0%(compute_cot 仅 7.1%)。
   math500 审计已证明这类题会奖励"猜数"轨迹,RL 训练集须过滤或降权。
2. **中文高考档 RL 供给薄**:中文 RL 题几乎全是 ape210k(小学难度);
   高考风格的 cn_k12 是英文。若 RL 目标包含中文高考档,建议从 chinese-r1(2.6 万,
   先滤可猜)+ EduChat 转 RL 用(题面+gold 齐备)。

**结论:覆盖广度够,但直接全量上 RL 不可取。建议构建 RL 课程子集:**
compute_cot hard + gsm8k + 滤可猜后的 cn_k12/metamathqa/big-math(0.1-0.9 带)
≈ 25-40 万条;先用 S4 产出的 checkpoint 在 A4000 上对各源抽样跑 pass@8 标定
"自家口径"的带宽,再定终配比。

## 6. 产物与接口清单

| 产物 | 路径 |
|---|---|
| 清洗脚本 | `data_pipeline/clean_s4.py`(`python -m data_pipeline.clean_s4 --pool ... --out ...`) |
| 清洗池 + 统计 | `/data/zilu/data_unified_v2/train_{sft,rl}_s4clean.jsonl{,.stats.json}` |
| S4 训练集 | `/data/zilu/data_unified_v2/parquet/train_sft_s4_anneal_16k.parquet`(+manifest) |
| heldout v2 | `eval/heldout_v2.jsonl` + `eval/heldout_v2_exclude.txt`(构建:`eval/build_heldout_v2.py`) |
| reweight 新参数 | `--math-pool / --general-pool / --exclude-hashes`(向后兼容) |
| 构建日志 | `logs/clean_s4_{sft,rl}.log`、`logs/reweight_s4.log` |

## 7. 建议的下一步

1. s3r1 训完(今晚)→ 修复版判分器跑 final_eval 拿基线;
2. 从 s3r1 final 起 S4 退火(同 sft_trainer 配置,换 parquet,save_freq 200);
3. async_eval 用 heldout_v2 跟踪,重点看 gaokao 层各 source 曲线与熵探针(防塌缩);
4. 并行:RL 课程子集构建 + 用 S4 checkpoint 标定各源 pass@8 带宽。

## 8. 数据源缺陷:zhr1:kaoyan 几乎不是数学(2026-06-12 晚,两次更正)

用户两轮审阅戳穿:`zhr1:kaoyan` **不是考研数学,是考研全科混题**(史/政/化/生/医/
计算机/心理学为主)。

**更正(勿信第一版的"41%含数学")**:第一版用松正则(含 $/\mathrm/%/数字即算数学)
估 41%——但非数学题恰恰满是 LaTeX(化学式、CS 伪代码、百分比),全部绕过该判据。
人工核验 heldout 60 题抽样:**真数学仅约 6 题(10%)**,其余皆冒充。故 zhr1:kaoyan
作为数学源基本不可用,真实可用率约一成,远非 41%。

**影响**:
1. **评测层作废**:kaoyan-heldout 60 题真数学仅 ~6,撑不起观测层 → **整层删除**
   (eval/heldout_v2.jsonl 现 4744 题;此前"kaoyan acc 上涨"全是史政猜对率波动,作废)。
2. **训练轻度污染**:约数百条非数学题混入 S4 hard 桶(占 29 万 <0.2%,可忽略,不重训)。
3. **教训(同 06-12 早 compute_cot 那次)**:看内容不看标签/符号;LaTeX 富集 ≠ 数学。

**RL 待办**:zhr1:kaoyan 直接弃用(可用率仅一成,过滤不划算);zhr1:Advanced-Math
需重新人工抽检(第一版"79%"同样可能被松正则高估,勿直接采信)。
