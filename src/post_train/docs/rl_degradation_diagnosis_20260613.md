# RL v1 退化诊断 + v2 修复重跑 · 2026-06-13

> 第一版正式异步 GRPO(`grpo_formal_fullepoch_s3r1_sync4_stale05_n16_t12`,见
> [`rl_logging_and_process_eval_20260613.md`](rl_logging_and_process_eval_20260613.md))
> 跑到 param_version 56 时,过程评测暴露**系统性能力退化**。本文记录:观察→根因→修复→重跑。
> 一句话:**奖励/配置让"侵蚀已有能力"跑赢了"学会新东西",净效果是模型变笨。已停 v1,按修复包重跑 v2。**

## 1. 观察(v0→v50 全量过程评测,5000 题贪心 pass@1)

按 v0 基线 per-source acc 分难度带追踪:

| 难度带 | v0 | v20 | v40 | v50 | 结论 |
|---|---|---|---|---|---|
| 总 acc | 28.9% | 22.2 | 22.5 | 22.7 | 横在 22.7,−6.2,上不去 |
| 太简单(>90%,546) | 99.6% | 73.1 | 77.1 | 78.4 | **−21,本来稳对的丢一大块** |
| **可学习带(10–90%,2256)** | **36.2%** | 26.9 | 26.6 | **26.4** | **−9.8,到 v50 仍单调下滑** |
| 太难(<10%,2198) | 3.8% | 4.7 | 4.8 | 5.1 | +1.3(多为运气,见下) |
| format | 82% | 92 | 93 | 94 | 饱和(先薅满便宜分) |

**翻转矩阵(v0→v40)**:对→错(搞坏)498 vs 错→对(教会)181,**破坏是建设的 ~3 倍**。其中:
- easy 纯损(123 丢、0 得);mid 真教会 97 但砸坏 313;hard 净 +22 **但 67% 是个位数/字母蒙对**,真 miracle 上限 ~28。
- 即"赢"里掺大量 **unfaithful 运气**(outcome-only 判分把胡乱蒙对记成进步)。

**退化的真实形态(原文实证,样本见评测 dump `*_val_dumps/{1,6100,11355,13941}.jsonl`)**:
1. **渐进崩坏**:绝对值不等式 v0/v10 一字不差稳对 → v20 概念坏 → v40 稳定成"自信胡编"(把会的 procedure 换成编的解释)。
2. **基础技能丢失**:运算顺序 `((4+1)×2−2)×4` v0 对 → v40 不会解析括号(还"提前收手忘最后一步""凭空发明除法")。
3. **常识丢失**:Gavin 题 v0 知道"一年=365天" → v40 当成"12个月",自信稳定算错。
4. **涌现式放弃**(目前仅 1 例,萌芽):Kirill 题代入丢 `−14` 得欠定假象,模型"宣布条件不足"box 0。

## 2. 根因(证据闭环)

真信号(correct 优势)被四样东西淹没:

1. **GRPO std 归一(`norm_adv_by_std_in_grpo=True`)× 全错组**:44% 竞赛集采 16 个全错,correct 零方差,但 format/长度 shaping 仍有连续差异 → 除以极小 std **把 shaping 噪声放大成"更长/更套格式"的伪梯度** → 顺共享权重污染中等带(头号主犯)。
2. **shaping 过重**:`format_bonus=0.1 + think_len_bonus≤0.2 = 0.3`,在难题(correct 期望 0.14)上喧宾夺主,先把便宜分薅满(format 82→94%)再说。
3. **KL 太松(`kl_loss_coef=0.001`)**:锚不住 SFT 已会的简洁正确,策略漂走丢知识。
4. **数据**:40% 竞赛集 p≈0,零有效梯度 + 上述伪梯度来源;高温(1.2,本身利于探索)同时也产出被错误奖励的退化样本。

**注**:引擎本身没坏——hard 区求导 `12→24` 是 RL 真修对的执行错误,155 个数值错→对里有真学习。是**配置让破坏跑赢**,不是路线错。继续训不会自愈(v50 mid 仍在跌)。

## 3. v2 修复包(本次重跑)

原则:**关掉破坏的闸,留住有效的探索**。一条没动高温(探索是 hard 区赢的来源)。

| 改动 | v1 → v2 | 治什么 | 改在哪 |
|---|---|---|---|
| `algorithm.norm_adv_by_std_in_grpo` | True → **False** | 全错/低方差组不再放大 shaping 伪梯度(头号) | runner(`NORM_ADV_BY_STD`) |
| `actor.kl_loss_coef` | 0.001 → **0.005** | 锚住 SFT 简洁正确,刹住丢知识 | runner(`KL_LOSS_COEF`) |
| `REWARD_FORMAT_BONUS` | 0.1 → **0.05** | 降薅格式空间,correct 主导 | `RL/reward_verl.py`(env) |
| `REWARD_THINK_LEN_MAX_BONUS` | 0.2 → **0** | 去掉低收益、添方差的长度分 | `RL/reward_verl.py`(env) |
| 训练数据 | 全量混合 → **v2 课程池** | 剔 dapo/big-math 死签,留中等带主干 + 10% 难题彩票 | `RL/reweight_rl_v2curric.py` |
| 温度 / group | **1.2 / 16 不变** | 保留高温探索(hard +真miracle 的来源) | — |

**数据池 `train_rl_s4_v2curric.parquet`(100.5万条)**:compute_cot 39%(170+ 题型广度全留)+ 应用题 ~51%(calc/orca/metamath/gsm8k/chinese,中等带主力)+ 难题彩票 9.9%(numina/openr1 抽 10万,p~0.04、G=16 中奖率~48%)。剔除 dapo(0%)、big-math(2%)、deepscaler。

**为什么 v1 不接 filter_groups**:norm_adv=False + 砍 shaping 后,全错组 ≈ 零方差零梯度,即便不在线过滤也基本无害(只剩浪费采样)。`filter_groups`(在线按 correct 方差过滤 + 自动移动难度前沿)留作 **v2.1 的效率/课程升级**(异步路径继承自 `separation/ray_trainer.py`,需补接线)。

**未做但记下的候选(需仔细设计,勿轻上)**:判分取首个 `\boxed` / 罚"出答案后续写"——直觉对但有反作用风险(worked-CoT 常 box 中间量),留待 v2.1。

## 4. 验证方法(怎么知道修好没)

复用同一套诊断脚本,重跑到 v20–30 即可判:
- **修好**:easy 不再纯损、**mid 净转正**、hard 续 +。
- **没修好**:同脚本立刻暴露,不白跑。

诊断脚本(本次现写,均在 `RL/` 下临时跑,核心逻辑:dump 的逐条 `correct` 对齐 val parquet 的 `data_source`):
- 难度带轨迹 / 翻转矩阵 / reward 分项 / MCQ 运气拆分 / 同题跨版本原文 diff。

## 5. 关键文档与文件

| 类别 | 路径 |
|---|---|
| 本诊断 | `docs/rl_degradation_diagnosis_20260613.md` |
| v1 基建说明 | `docs/rl_logging_and_process_eval_20260613.md` |
| 奖励函数(判分同口径) | `data_pipeline/reward.py::compute_reward` / RL wrapper `RL/reward_verl.py` |
| 启动脚本 | `RL/run_grpo_fully_async_split_a800_ref3_a4000.sh` |
| 数据重配 | `RL/reweight_rl_v2curric.py` → `parquet/train_rl_s4_v2curric.parquet` |
| 事件日志分析 | `RL/analyze_events.py` |
| 优势计算(norm_adv 生效点) | `verl/.../separation/ray_trainer.py:566 _fit_compute_advantage` |
| v1 退化过程评测 dump | `logs/grpo_formal_fullepoch_..._n16_t12_..._val_dumps/{1,3274,6100,8810,11355,13941}.jsonl` |
| v1 最后 checkpoint(退化样本,反面参照) | `logs/ckpt_grpo_formal_fullepoch_..._n16_t12_.../global_step_50` |

## 6. v2 重跑配置(2026-06-13 22:xx 启动)
- RUN_GROUP=`grpo_v2curric_s3r1_normadvF_kl005_noshaping_n16_t12`
- 起点同 v1:`sft_s3r1/global_step_3874_hf`;数据 `train_rl_s4_v2curric.parquet`;val 仍 `val_rl_s4clean_fix.parquet`(对比可比)。
- sync4 / stale0.5 / ref_service / n16 / temp1.2 / TEST_FREQ=10 / SAVE_FREQ=50 / val_before_train。
