# 全模型终评总表(统一登记册)

> 维护:每训出/评出一个关键档就更新此表。细节分析见 [`eval_tracking.md`](eval_tracking.md)(F2 终评节/Qwen 基线节)。
> 终评口径:`eval/final_eval.py`,6 benchmark,贪心 Pass@1 + 采样 Pass@8(k=8,T=0.8,max_new_tokens=2048),判分=训练同口径 `data_pipeline.reward`。
> 产出:每档 `<ckpt>/final_eval/{summary.json,summary.md,<bench>.jsonl}`(逐样本含贪心+8采样全文)。

最后更新:2026-06-14。

## 1. 模型谱系 · 从哪训 · 训了什么 · 文件索引

| 模型 | 链路位置 | 从哪训(起点) | 训了什么(数据/超参) | checkpoint | 终评状态 |
|---|---|---|---|---|---|
| **base** | 预训练基座 | — | 1.7B Llama,数字切分 tokenizer,预训练(ctx16k 权重) | `checkpoints/qseek_digitsplit_base` | ✅ 4/6(gsmplus/cc 主动停评⁶) |
| ~~no-math SFT(F1代理)~~ | base 代理 | base | `qseek_foundation_v1` 通用SFT(无数学,33.7k)。**checkpoint 已删,仅剩数据;用户决定不重训** | — | ❌ 跳过(用 base 裸评代替) |
| **F2** 地基SFT | base→F2 | base(不热启F1,F1数据脏) | `train_sft_foundation_8k`(126万):算术地基43%(compute_cot+calc)/英文应用题22%/竞赛陪练7%/通用26%。lr1e-5/2ep/8k | `checkpoints/sft_foundation_v2/global_step_9858`(+_hf) | ✅ 已评(6/6) |
| **S3-R1** 难度课程SFT | F2→S3 | F2-9858_hf | `train_sft_s3r1_16k`(99万):难度坡度 easy52/med25/hard9.5/通用13,竞赛numina翻倍(11%)。lr5e-6cos/1ep/**16k**。**=RL起点** | `checkpoints/sft_s3r1/global_step_3874`(+_hf) | ✅ 已评(6/6) |
| **S4** 退火SFT | S3→S4 | S3-3874_hf | `train_sft_s4_anneal_16k` 退火。**RL未用此档** | `checkpoints/sft_s4_anneal/global_step_1140_HFFIX`(+_hf) | ✅ 已评(6/6) |
| **RL v1**(退化) | S3→RL | S3-3874_hf | 异步GRPO,sync4/n16/t1.2,norm_adv=True(后诊断退化)。停在v56 | `logs/ckpt_grpo_formal_fullepoch_.../global_step_50`(merge→`fastrl/v3eval_hf/rlv1_gs50_hf`) | ✅ 已评(6/6) |
| **RL v2**(收敛) | S3→RL | S3-3874_hf | 退化修复:norm_adv=False/kl0.005/砍shaping/课程数据。收敛于基线下 | `logs/ckpt_grpo_v2curric_.../global_step_300`(merge→`fastrl/v3eval_hf/v2_gs300_hf`) | ✅ 已评(6/6)⁵ |
| **Qwen3-1.7B base** | 外部基线 | — | 大厂预训练 base | `checkpoints/external/qwen3_1_7b_base` | ✅ 6/6 |
| **Qwen3-1.7B instruct** | 外部基线 | — | 大厂后训练(thinking) | `checkpoints/external/qwen3_1_7b` | ✅ 6/6(cc仅P@1) |

## 2. 终评准确率总表(Pass@1 / Pass@8,%)

> 空=未评;🔄=运行中待回填。format_rate 见各档 summary.md。

| 模型 | svamp | math500 | cmath | gsm8k | gsmplus | cc-reserved |
|---|---|---|---|---|---|---|
| base | 2.0/6.7 | 0.4/1.6 | 0.0/0.18 | 0.4/2.4 | ⏹️停评⁶ | ⏹️停评⁶ |
| no-math SFT | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| **F2-9858** | 39.3/73.0 | 5.6/24.0 | 42.8/62.3 | 16.0/43.0 | 7.6/26.5 | 71.8/84.4 |
| **S3-3874** | 41.7/73.3 | 5.0/23.4 | 41.4/62.6 | 18.6/47.3 | 9.9/29.6 | 72.9/83.9 |
| **S4-1140** | 40.7/74.3 | 4.0/23.0 | 40.5/61.9 | 18.7/44.8 | 9.7/28.3 | 73.3/84.4 |
| **RL v1-gs50** | 32.3/63.3 | 4.2/19.6 | 41.0/52.4 | 13.8/40.9 | 8.4/26.4 | 51.4/66.9 |
| **RL v2-gs300** | 40.3/62.7 | 5.8/17.2 | 45.0/55.5 | 19.1/38.7 | 9.5/23.8 | 66.9/74.3 |
| Qwen base | 28.7/90.7 | 52.6/78.6 | 34.5/86.0 | 48.1/92.0 | 34.7/74.1⁴ | 50.4/82.1² |
| Qwen instruct | 84.0/91.0 | 34.0/49.6¹ | 63.7/91.8 | 76.9/89.8 | 54.6/69.2⁷ | 66.3/—³ |

¹ Qwen instruct 的 math500 被 2048 token 截断压低(fmt 仅 37.6%)。Qwen base format_rate≈0(不跟随格式,故 p@1 低但 p@8 高=潜力睡在分布里)。
² Qwen base cc-reserved 用默认 prompt 跑完整 4372 题；标准判分 50.4/82.1、format=0。严格按 `final_eval.py` / `data_pipeline.reward`，不做格式放宽。产出见 `checkpoints/external/qwen3_1_7b_base/final_eval_cc_reserved_default/`。
³ Qwen instruct cc-reserved 用默认 prompt 跑完整 4372 题 Pass@1；标准判分 66.3%、format=84.7%。完整 Pass@8 因输出过长预计耗时过高，本轮中止，暂不填。产出见 `checkpoints/external/qwen3_1_7b/final_eval_cc_reserved_p1_default/`。
⁴ Qwen base gsmplus format_rate=0(不跟随格式),p@1 34.7 低、p@8 74.1 高=潜力睡在分布里(与其余项一致)。
⁵ **RL v2 终评踩坑订正(2026-06-14)**:首次直接把 verl `global_step_300` 喂 `final_eval`,其内部 `convert_to_hf` 的 `find_ckpt_pt` 只搜顶层、漏看 `actor/` 子目录 → 找不到权重却**静默退回纯 base** → 评出 svamp 1.3/8% 假数据。已修 `eval/async_eval.py`(搜 `actor/` + 找不到即报错,不再静默退 base),并改用 `model_merger` 正经合并到 `/data/zilu/fastrl/v3eval_hf/v2_gs300_hf` 重评。表中为订正后真值(svamp fmt=100%)。
⁷ Qwen instruct gsmplus(2400 题,5 卡并行补齐 2026-06-14):Pass@1 54.6 / Pass@8 69.2,format=58.7%。按扰动看,critical thinking 子类 0.0%(thinking 模型对"陷阱题"输出超长被 2048 截断),problem understanding 80.3% 最高。产出见 `checkpoints/external/qwen3_1_7b/final_eval_gsmplus/`。

⁶ **base 的 gsmplus / cc-reserved 主动停评(2026-06-14)**:裸预训练基座前 4 项已全趴地板(svamp 2.0、math500 0.4、cmath **0.0**、gsm8k 0.4,均不跟随格式、不输出 EOS),"SFT 之前 ≈ 0"的起点基准已钉死,剩两项必为同样地板水平,跑完仅归档无信息增益且占 6 卡又慢(每条顶满 2048 token),故停评省算力。已完成的 4 项逐样本 dump 见 `checkpoints/qseek_digitsplit_base/final_eval/{svamp,math500,cmath,gsm8k}.jsonl`。

## 3. 已知结论(截至 F2/S3,详见 eval_tracking.md)

- **预训练底子差距是主轴**:Qwen base 的 pass@8 多数项在 79-92%,我们 24-84%。多步推理/竞赛差距最大,后训练补不齐。
- **我们的对齐兑现不差**:贪心口径 cmath/svamp 赢 Qwen base(中文池+应用题对齐有效);compute_cot(cc-reserved)我们 72>Qwen base 50.4、Qwen instruct 66.3,**digit-split+算术骨干在其瞄准能力上仍强**。
- **F2→S3 变化很小**:gsm8k 16→18.6、cc 持平、math500 持平——S3 难度课程没带来明显终评提升(与后续 RL 收敛于基线下一致)。
- **RL 只放大不创造**:Qwen instruct 把 base 的 pass@8 兑现为 pass@1(gsm8k p@1 48→77,p@8 不动),是我们 RL 该复刻的模式;天花板由预训练定。
- **RL v1 终评量化确认退化(2026-06-14)**:相对 S3-3874 起点**全线下滑**,cc-reserved p@1 72.9→**51.4(−21.5)**、svamp 41.7→32.3(−9.4)、gsm8k 18.6→13.8(−4.8);**连 pass@8 都掉**(cc 83.9→66.9、cmath 62.6→52.4)——不只是"会而不稳"的稳定性损失,而是**潜在能力本身被侵蚀**。佐证 v1 退化诊断(norm_adv std 归一×全错组把 shaping 噪声放大成伪梯度)。这正是 v2 修复(norm_adv=False)要止住的灾难。
- **S4 退火 ≈ S3(几乎不动)**:六项相对 S3 全在 ±1 噪声内(cc 72.9→73.3、gsm8k 18.6→18.7、svamp p@8 73.3→74.3)。退火没带来终评增益——和"S3 之后各阶段终评都没明显提升"一致,再次指向预训练天花板。
- **RL v2(收敛版)终评:止住了退化,但没兑现增益,反而损失多样性(2026-06-14)**。相对 S3-3874 起点(均 p@1/p@8):

  | bench | S3 起点 | v2-gs300 | Δp@1 | Δp@8 |
  |---|---|---|---|---|
  | svamp | 41.7/73.3 | 40.3/62.7 | −1.4 | **−10.6** |
  | math500 | 5.0/23.4 | 5.8/17.2 | +0.8 | −6.2 |
  | cmath | 41.4/62.6 | 45.0/55.5 | +3.6 | **−7.1** |
  | gsm8k | 18.6/47.3 | 19.1/38.7 | +0.5 | **−8.6** |
  | gsmplus | 9.9/29.6 | 9.5/23.8 | −0.4 | −5.8 |
  | cc-reserved | 72.9/83.9 | 66.9/74.3 | **−6.0** | −9.6 |

  - **关键模式:pass@1 基本持平(±1,个别 cmath+3.6/cc−6.0),但 pass@8 六项全线下滑 −5.8~−10.6。** 这是典型 RL "收窄分布"特征——v2 把采样多样性磨掉了,**却没把丢掉的 pass@8 转化成 pass@1 增益**(理想 RL 应像 Qwen instruct:p@8 不动、p@1 顶上去)。净效果是**轻度负向**:贪心没涨、潜力变窄。
  - **vs v1:norm_adv=False 的修复确实生效**——v1 是灾难性侵蚀(cc p@1 −21.5、连 p@8 崩),v2 温和得多(cc p@1 −6.0);**没有能力侵蚀,只是没学到东西 + 丢多样性**。
  - **结论坐实"收敛于基线之下"**:这套薄底子 1.7B 上,纯 GRPO 既没创造新能力(pass@8 天花板由预训练定、还被磨低),也没把已有潜力兑现到贪心。**印证 v3 错题教师SFT 实验的动机**:RL 自己补不上的,得靠外部强教师注入。

## 4. 待办

- [ ] 回填 base / RL v1 / S4(全6)终评(运行中,GPU 0,3/4,5/6,7,setsid 后台)。
- [x] ~~no-math SFT~~:checkpoint 已删、用户决定不重训,跳过(base 裸评代替)。
- [ ] (可选)RL v2 各档终评,作"纯RL"对照。
- [ ] 补齐 Qwen 基线的 gsmplus(可比性)。
