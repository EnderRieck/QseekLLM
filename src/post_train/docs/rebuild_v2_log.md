# 弹药库 v2 重建执行记录（2026-06-10）

> 对应 [data_audit_report_20260610.md](data_audit_report_20260610.md) §五 行动清单的落地。决策:F1 跑完不中断当基线;**F2 从 base 用 v2 数据重训**。
> 产出目录:`/data/zilu/data_unified_v2/`(不动 v1,在训 F1 不受影响)。

## 一、代码修改清单(全部已完成,文件:改了什么)

| 文件 | 修改 | 验证 |
|---|---|---|
| `data_pipeline/adapters.py` | ① `_num_from_text` 重写(orca 修复:末含数句 is/= 引导数+分数整捕,多数字无引导→丢弃) | 8千抽样:保留 89.1%,抽出值 100% 在结论句,人检 8/8 对 |
| 同上 | ② 新增 `adapt_numina`(valid 双过滤+剔 proof/notfound+source 纠错+按源分难度) | 冒烟 3000:2998 格式合格 |
| 同上 | ③ 新增 `adapt_openr1`(取 correctness_math_verify=True 最短 R1 轨迹,cc=N 当难度) | 冒烟 3000:2594 SFT/406 仅RL |
| 同上 | ④ `adapt_calc_ape` 重写(chain→中文 worked CoT,逐步 Python 复算校验,洗 float/around 噪声,损坏列式不展示,退化题丢弃) | 3万抽样:78.5% SFT 化;200 样张 v3 零可疑(`outputs/ape210k_cot_samples.md`) |
| 同上 | ⑤ 新增 `adapt_chinese_r1_math`(math repos score≥8,reasoning_content 包 think;**GSM8K_zh 故意排除**防跨语言泄漏) | 产出 26,467 |
| 同上 | ⑥ 新增 `adapt_math_hendrycks` + `iter_hendrycks_train`(MATH-train 7.5k 官方解,difficulty=level) | — |
| `data_pipeline/build.py` | EVAL_PATHS:+svamp(两种拼法)/+math-500/+CC val·test;competition-math→math-hendrycks **test-only**(新 fmt `arrow_test`/`cc_jsonl`);SOURCE_ORDER 插入 math-hendrycks/chinese-r1-math | eval_hashes 18k→**84,147** |
| `data_pipeline/build_general.py` | 接入 eval_hashes 隔离 + seen 以数学池题面打底(`--math-pool`),manifest 加 leak 列 | — |
| `data_pipeline/general_adapters.py` | tulutalk 过滤器(unsafe 剔/单轮 st_reward<0.75 剔);dynamics 剔 type=reasoning(裸字母 MCQ 1.27万) | — |
| `data_pipeline/reweight_sft.py` | F2 配方 caps(见下);UNIFIED→v2;cap=0 语义=剔除(ot3) | — |
| `data_pipeline/filter_length.py` | **新增**:8k token 过滤正式脚本(F1 时是临时手工,不可复现) | — |
| `eval/build_heldout.py` | +训练池 qhash 过滤(`--train-pools`,默认 v2 三池);comp-math 抽样源→**math-500** | 待 v2 池建完执行 |
| `SFT/run_foundation_v2_sft.sh` | **新增**:F2 启动脚本(data→v2,实验名 sft_foundation_v2,超参沿用 F1) | 待数据就绪 |
| `scripts/rebuild_v2.sh` | **新增**:四步全链(build→build_general→reweight→filter+to_verl) | 运行中 |

## 二、F2 配方(reweight FOUNDATION_CAPS)

compute_cot 全留 / orca 全留(修复后) / metamathqa 11万 easy 优先 / math-hendrycks 全留 / **chinese-r1-math 全留 + calc-ape210k 全留(中文 ~18万)** / numina 6万 / openr1 2万 / **ot3 剔除** / bespoke·deepscaler·infinity-math 全留 / tulutalk·infinity·chinese-r1(通用) 各 8万 / coig·dynamics·no_robots 全留 / dolly 降到 8千。

## 三、重建运行(scripts/rebuild_v2.sh → logs/rebuild_v2.log)

- 启动 2026-06-10 14:12。中途读数:ape210k kept 180,283(SFT 148,299✓);**metamathqa leak 75,081→17**(MATH-train 解放生效;其 AnsAug 原题面 dup 给了更早入池的 hendrycks 官方版,符合预期)。
- **三次启动才跑完**:14:12 首启与 14:52 重启均被静默杀死(无 traceback/无 OOM 证据,死点不同,疑似外部信号;单源复现正常)。15:15 第三次由 Claude 会话托管后台运行,16:12 全链完成。
- **修一个 bug**:reweight 写 parquet 时批间 struct 键序不一致(`{role,content}` vs `{content,role}`)导致 schema 冲突崩溃 → `reweight_sft.py` rows_iter 中显式归一化键序后重跑 3-4 步通过。
- **manifest 对账(16:12,全部命中预期)**:SFT 池 1,436,749 / RL 池 1,557,905 / 评测隔离剔除 4,159。关键:calc-ape210k sft=148,299✓ chinese-r1-math=26,414✓ metamathqa leak=17✓ orca=157,318✓。
- **F2 切片**:1,265,064 行(数学 74.7%/通用 25.3%,compute_cot 31.1%);8k 过滤丢 3,102(主要 openr1 超长轨迹)→ 最终 **1,261,962 行**。
- **抽检**:40 条(随机 30+定向 ape210k/chinese-r1 各 5)通过,proof/notfound 抽样扫描 3.4 万条零命中 → `outputs/v2_spot_check_30.md`。
- **heldout v2**:3,881 条(svamp 300 全保留——泄漏已从训练侧根治,训练池过滤剔除 0 条)→ `eval/heldout.jsonl`。
- **gold box 自检(06-10 训中补做,`scripts/selfcheck_gold.py`)**:每源抽 300,用 reward.verify_answer 解析 gold_response 与 ground_truth 比对——**11 个带答案源 3,300/3,300 全过(100%)**;bespoke-stratos/ot3 为 use=sft、style=none、gt 空(蒸馏轨迹无独立标答,不参与判分/RL,符合设计),box 提取本身正常。失败样例 dump:`outputs/gold_selfcheck_fails.jsonl`。
- **think↔box 自洽抽检(06-10,`scripts/selfcheck_think_box.py`)**:每源抽 400,启发式比对 think 末段结论与 boxed——机器组装源(orca/ape210k/metamathqa/compute_cot/gsm8k,F1 出事的地方)**99.8~100% 一致**;R1 轨迹源(openr1/zhr1/deepscaler/numina)启发式标 4~24% 可疑,人工抽读 18 条**全为 LaTeX 写法差异误报**(如 think `-41/38` vs box `\frac{41}{38}`,`25√2/√17`≡`50/√34`),零真实不一致。可疑 dump:`outputs/think_box_suspects.jsonl`。
- **F2 起训 16:25 左右**(tmux `sft`,A800/card1,4,929 step×2epoch,save_freq=500)+ async_eval 已挂监视(tmux `eval`,card2,3)。

## 四、待办(v2 池建成后)

- [x] manifest 对账 + 抽样人检(06-10 16:20 完成,见 §三)
- [x] `python -m eval.build_heldout --out eval/heldout.jsonl`(06-10 完成,3,881 条)
- [x] `bash SFT/run_foundation_v2_sft.sh` 启动 F2 + `eval.async_eval --watch` 跟评(06-10 16:25 起训)
- [x] ~~compute_cot per-source 细粒度课程 eval 集~~(**06-10 用户拍板不建**:heldout 每子源 ~7 条作过程评测够用;S3 动态课程信号改用终评 cc-reserved 的 per-source 读数)
- [x] 终评脚本 `eval/final_eval.py`(06-10 新增,已冒烟):6 benchmark(含 **cc-reserved**=Compute_Cot 保留集每子源采20,数学基本功标尺),Pass@1/8,用法见 `eval/README.md`

## 五、F2 训完后的 checklist(预计 06-11 晚收敛)

1. **收敛判定**:`<ckpt>/eval_dumps/metrics.jsonl` 曲线进平台期、epoch2 无过拟合(train loss 降但 heldout acc 跌=信号)→ 选最终 checkpoint。
2. **终评**:`.venv/bin/python -m eval.final_eval --ckpt <ckpt>/global_step_N --gpus 2,3`(~1-2h)。
3. **判读 F1 三短板**:乘法(cc-reserved 最弱子源表)/ 中文(cmath 三维)/ 扰动(gsmplus 7 类)→ 记入 eval_tracking.md 作 S3 基线。
4. **⚠️ 磁盘提醒(只提醒,清理须用户确认)**:~18 ckpt×21G≈380G。用户明确要求**先不清理**——可能要回看中间 checkpoint 观察训练过程;S3 起训前空间不够时再向用户提出。
5. **S3 准备**(可与 F2 训练并行,纯 CPU):难度切桶(big-math solve_rate/MATH level/openr1 cc/numina source)→ S3 配方 reweight(R1 易:中:难=60:30:10,openthoughts3 进场)→ 训练脚本改热启 F2 + ctx 16k → 评测加策略熵+Pass@k。
- ~~F1 终评~~(06-10 用户删除 F1 checkpoint,重训为准;A800 已空闲,F2 可即起)

> ⚠️ 交接看这份就够:**[HANDOVER_20260610.md](HANDOVER_20260610.md)**(命令级接手步骤)。
