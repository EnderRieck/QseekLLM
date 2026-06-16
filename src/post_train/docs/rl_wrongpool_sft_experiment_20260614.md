# 对照实验:错题教师SFT 是否帮助 RL(v3-wrongpool)

> 2026-06-14 启动。对照的是 v2 纯 RL(`grpo_v2curric_s3r1_normadvF_kl005_noshaping_n16_t12`,已收敛:easy 94 守住、mid ~32 卡基线下、hard ~6 蒙对噪声)。

## 1. 假设与动机

v2 证据:纯 GRPO 在这个薄底子 1.7B 上**收敛于基线之下**,mid 读题瓶颈救不动,hard pass@k≈0 无可放大。判断:更多同款 RL/SFT 无效,真瓶颈是预训练容量。**本实验问一个具体问题:把"模型当前不会的题"交给一个强教师(GPT-5.3-Codex-Spark)生成解法、周期性 SFT 重放,能否补上 RL 自己学不会的部分?**

教师可用性已验证(`codex exec -m gpt-5.3-codex-spark`):对我们 val 的 mid/hard 题真实正确率 5/6,轨迹简洁、原生 `<think>+\boxed` 格式;唯一错的是 1 处配方算术滑误。**按用户决定:教师输出不过滤,直接用("好模型的能力就算是错的也很帅")。**

## 2. 算法修改(相对 v2 的唯一变化)

RL 配置与 v2 完全一致(norm_adv=False / kl=0.005 / 无 shaping / n16 / t1.2 / 课程数据 `train_rl_s4_v2curric.parquet`)。**只改一处算法 + 加一条支路:**

### 2.1 梯度屏蔽 + 入池(组件①,trainer 内)
每个 prompt 的 n=16 rollout 组,按正确数 `c` 分类:
- `c == 0`(**全错**):入错题池。注:norm_adv=False 下全错组 advantage 本就≈0(天然零梯度)。
- `0 < c <= LUCKY_MAX`(**蒙对/侥幸**,默认 LUCKY_MAX=2):入错题池 **且梯度屏蔽**(response_mask 置零,不进策略梯度也不进 KL)。理由:侥幸对的解大概率不忠实,不该被 RL 强化;交给教师补正解。
- `c > LUCKY_MAX`:正常参与 RL 梯度(可学习前沿)。

效果:RL 梯度只在"有正确率方差的可学习前沿"上更新;全错+蒙对都转入教师SFT。这相当于 DAPO dynamic-sampling 的思想,但被过滤的组不丢弃而是**导流到教师SFT**。

### 2.2 教师生成(组件②,独立进程,无GPU)
`RL/teacher_worker.py` 消费 `pending.jsonl` → 调 spark 生成解法 → 写 `sft_ready.jsonl`。不过滤。详见 §3。

### 2.3 错题重放SFT(组件③,检查点循环)
每隔固定 RL 版本数,暂停/续接做一轮 SFT,消费 `sft_ready` 的一部分。走检查点循环(RL档→SFT→续RL),复用已验证 SFT 栈,风险低、产出可对照。详见 §4。

### 2.4 codex 限流/异常鲁棒性
- **教师worker**:codex 整批全失败(典型限流)→ **不推进 offset、指数退避(30s→封顶30min)重试同批**,恢复后接着做,不丢错题;零散失败才丢弃。
- **重放SFT阶段**:`build_replay_sft_parquet.py` 取不到 ≥REPLAY_MIN 条新样本 → exit 3 → 编排器 **"无新答案样本,本轮 SFT 直接跳过,继续 RL"**(`continue from RL_HF`)。即 codex 不可用时实验自动退化为纯 RL,绝不卡死/崩。

## 3. 错题池目录与数据流

`WRONGPOOL_DIR`(默认 `/data/zilu/fastrl/wrongpool_v3exp/`,数据盘、跨重启持久):
- `pending.jsonl` — trainer 追加:`{data_source, ground_truth, prompt:[sys,user], label, correct_count, n, pver, ts}`
- `sft_ready.jsonl` — 教师worker追加:`{messages:[sys,user,assistant], data_source, ability, extra_info}`(直接 SFT 格式)
- `.teacher_offset` — 教师worker已处理的 pending 行数
- `.sft_consumed_offset` — 重放SFT已消费的 sft_ready 行数
- `teacher_worker.log` — 教师worker日志

## 4. 关键接口(脚手架)

| 动作 | 命令 |
|---|---|
| 起教师worker | `python3 RL/teacher_worker.py --pool-dir $WRONGPOOL_DIR --workers 8 &` |
| 看池子规模 | `wc -l $WRONGPOOL_DIR/pending.jsonl $WRONGPOOL_DIR/sft_ready.jsonl` |
| 起RL(v3) | `RUN_GROUP=grpo_v3wrongpool_... WRONGPOOL_DIR=... bash RL/run_grpo_fully_async_split_a800_ref3_a4000.sh 4 5850` |

## 5. 评测/对照口径

与 v2 同口径:每 10 版过程评测(贪心 pass@1,按子任务分带),全量 I/O dump。**对照看点:mid 带能否在 v2 的 ~32 平台上被教师SFT顶起来 / hard 是否出现 v2 没有的真增益。** 因每轮多了 SFT 阶段,横轴用"RL版本"对齐时需扣除 SFT 阶段。

## 6. 关键代码与接口清单

| 组件 | 文件 | 说明 |
|---|---|---|
| ① 梯度屏蔽+入池 | `RL/wrongpool_hook.py` + `verl/.../separation/ray_trainer.py`(_fit_compute_advantage 末尾接入) | env `WRONGPOOL_ENABLE=1` 打开;全错/蒙对组 advantages 置零 + 写 pending.jsonl |
| ② 教师worker | `RL/teacher_worker.py` | 消费 pending→调 spark→规整格式→写 sft_ready.jsonl,不过滤 |
| ③ 重放SFT数据 | `RL/build_replay_sft_parquet.py` | sft_ready→messages-parquet,推进 .sft_consumed_offset |
| ③ 重放SFT训练 | `SFT/run_replay_sft.sh` | 参数化 verl sft_trainer,从HF热启 |
| 编排器 | `RL/run_v3_wrongpool_experiment.sh` | RL限版→merge→重放SFT→merge→续;教师worker后台 |

merge 命令: `python -m verl.model_merger merge --backend fsdp --local_dir <step>/actor --target_dir <hf>`。

## 7. 进度日志

- 2026-06-14: 设计锁定。教师可用性验证通过(对 val mid/hard 真实 5/6)。
- 2026-06-14: 用户追加——蒙对组也踢出当前轮次梯度(更干净:RL只在可学习前沿更新)。
- 2026-06-14: 组件①②③全部写完并验证(不动 v2,用空闲 GPU0/2 + v2 的 global_step_300 验证 C 的 merge/SFT 往返):
  - ① hook 逻辑单测通过(全错+蒙对组 adv 归零、正常组保留、入池正确);已接入 trainer(try/except 兜底,默认关)。
  - ② 教师worker 真实 codex 冒烟 3/3,格式规整(<think>+#### \boxed)通过。
  - ③ merge(v2档→HF)✓、replay parquet 构建 ✓、SFT 管线加载+进训练步 ✓(A4000 OOM 仅因测试选卡,A800 不会)。
- 2026-06-14 12:06: 掐断 v2(终态 v325/6档/33dumps 全留),12:07 启动 `RL/run_v3_wrongpool_experiment.sh`(RL_VERSIONS_PER_CYCLE=75 即~50真实版本/cycle、N_CYCLES=6,从 sft_s3r1/global_step_3874_hf 起)。12:25 进入训练,三组件协同正常(pending 填充、教师消费、无 hook skip)。
- **2026-06-14 12:47 预算事故:codex-spark 烧太快**。教师worker 8并发/medium 在 ~22min 打了 ~420 次调用(330成功+90失败),**烧光整个 5h 桶(97%→0%,resets 16:30)+ 29% 周额度(76%→47%)**。根因:spark 是重推理模型,单次烧几千 token。鲁棒性按设计生效(ALL-FAILED→退避→不丢 pending)。
  - **节流决策(用户拍板:中等)**:教师worker 改 **low档 + 2并发 + 60次/小时限速**(`RateLimiter` + `--max-per-hour`/`--batch-size`,已加进 `RL/teacher_worker.py`;编排器默认也改成限速)。把一个 5h 桶摊到 5h 用完。weekly 剩 ~680 次是更硬天花板。预计全程教师样本规模~千条级(每cycle 几百),对针对性补错题够用。cycle1 SFT 用已生成的 330 条即可跑通。
- **当前:cycle1 RL 训练中;限速教师worker(pid 由 nohup 起)<16:30 会限流退避、16:30 后按 60/h 续扫。待盯 cycle1 边界首次 merge→SFT→续。**
