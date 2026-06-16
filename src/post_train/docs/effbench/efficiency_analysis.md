# 三变体 GRPO 效率对比分析(交接文档)

> 目的:统一配置下对比三种 GRPO 训练架构的效率,量化并解释"我们的全异步(ref 独立卡解耦)到底快在哪、为什么"。
> 状态:2026-06-15 完成 3 变体各 5 步 smoke + 绘图 + 归因。**本文是交接文档,下一个 agent 照此可复现/续跑。**
> ⚠️ 读之前先看 §7「已知局限 & 下一步」——当前数字有小样本噪声,且 trigger=1 压制了我们的优势。

---

## 0. 一句话结论(修正版)

**串行 90.0s/步 → verl 原版全异步 43.1s/步(2.09×)→ 我们的(ref 独立卡)37.2s/步(2.42×)。**
- 2.1× 来自**全异步流水线**(rollout↔train 重叠),这部分是 verl 的、②③ 共享。
- 我们 split 的 **ref 解耦再贡献 ~1.16×**(43.1→37.2),即使在最不利的 trigger=1 下也是正收益。

> 历史教训:初版我用 `mean(step span 时长)` 当 sec/step,被**预热步 + 一个 0.0 bug span** 污染,误得出"③≈②(都 41s)中性"的**错误结论**。已改用稳态步周期(见 §3 指标定义)。早期文档/记忆里"比 verl 原生快 1.68×(基于 one-step-off)"的说法**作废**。

---

## 1. 统一配置(只留"架构"一个自变量)

| 旋钮 | 取值 | 说明 |
|---|---|---|
| group `rollout.n` | 8 | 每题 8 条 |
| 一个训练步 | 64 题 = 2 次 optimizer.step(ppo_mini=32×2) | 三者一致 |
| 几步一同步 | **每 1 步同步一次**(trigger_parameter_sync_step=1) | 三者一致;但见 §7,这恰好压制了异步红利 |
| actor 卡 | card1(A800) | 三者相同 |
| rollout 卡 | card4,5,6,7(A4000×4) | 三者相同 |
| 起点权重/数据 | `sft_s3r1/global_step_3874_hf` / `data_unified_v2/rl_smoke` | 三者相同 |
| max_prompt/resp | 1024 / 1024 | 三者相同 |
| 步数 | 5(smoke) | |

---

## 2. 三种架构 · ref 放哪 · 重叠到哪

| | ① 串行同步 | ② verl 原版全异步 | ③ 我们的全异步 split |
|---|---|---|---|
| 入口 | `RL/main_grpo_sync_split.py` | `verl.experimental.fully_async_policy.fully_async_main` | `RL/main_grpo_fully_async_split.py` |
| 启动器 | `RL/run_grpo_sync_a800_ref3_a4000_one.sh` | `RL/run_grpo_fully_async_nosplit_a800_a4000.sh` | `RL/run_grpo_fully_async_split_a800_ref3_a4000.sh` |
| **ref 放哪/怎么算** | card3 独立卡,但**同步阻塞** | **colocate 在 card1**,fit_step 内**同步 inline** 算(`_fit_compute_ref_log_prob`);card3 空闲 | **card3 独立卡 + 异步 RefService**:card3 上持续微批算 ref,trainer 从 ready_queue 取算好的批 |
| rollout | card4-7 | card4-7 | card4-7 |
| 重叠了什么 | **什么都不重叠** | rollout↔train 重叠(MessageQueue 流水线);ref 仍在 trainer 关键路径 | rollout↔train↔ref 三重叠;ref 脱离 trainer 关键路径 |

> 注:one-step-off(`verl.experimental.one_step_off_policy`)是 verl 另一个更早的异步实现,本对比**按用户要求已剔除**,只留上面三个。

---

## 3. 关键结果(稳态)

> **指标定义(重要,别再用 step-span mean)**:`sec/step = 相邻 step 起点间隔的中位数,丢掉第 1 个(预热)间隔`。
> 代码见 `eval/plot_effbench.py::analyze`。预热步(队列/ref-service 预热)是稳态的 2 倍且 ours 预热更久,必须排除。

| 指标 | ① 串行 | ② verl 原版异步 | ③ 我们的 |
|---|---|---|---|
| **秒/步(稳态)** | 90.0 | **43.1** | **37.2** |
| **vs 串行** | 1.00× | 2.09× | **2.42×** |
| **vs verl 原版** | — | 1.00× | **1.16×** |
| card1(actor/A800)util | 33% | **74%**(actor+ref) | 59%(actor only) |
| card3(ref/A4000)util | 27% | **0%(空闲浪费)** | **64%(ref-service)** |
| card4-7(rollout)util | 23% | 49% | 56% |

**单步阶段耗时拆解(解释 ②③ 差异从哪来)**:

| 阶段(mean) | ② verl 原版 | ③ 我们的 |
|---|---|---|
| gen(出队等待) | 8.4s | 29.6s |
| **ref** | **9.7s(同步 inline,在关键路径)** | (在 card3,不在 driver) |
| update_actor | 30.1s | 31.0s |
| param_sync | 3.5s | 5.3s |
| prefetch_wait | 5.4s | 11.9s |

读法:② 的 ref 9.7s 实打实串行压在 trainer 上;③ 把 ref 挪到 card3,trainer 的 ref 这段消失。代价是 ③ 的 gen 出队等待变长(8.4→29.6,在等 ref-service 把批备好),但净账 ③ 仍快 ~6s/步。**card3 从 0%→64%、card1 从 74%→59%**,就是 ref 工作被搬走的直接证据。

---

## 4. 图(`docs/effbench/`)& 怎么读

- `serial_timeline.png` —— 甘特 `gen→ref→actor-update→sync` 严格首尾相接、零重叠;三角色 util 接力棒式一个接一个亮。
- `verlorig_timeline.png` —— 甘特里 **ref 条存在**(同步 inline);card1 红+绿都满(actor+ref),card3 全 0。
- `ourasync_timeline.png` —— 甘特里**没有 ref 条**(ref 是 card3 上的 service,不是 driver 阶段),ref 工作只在 card3 绿色 util 栏;三卡稳态同时忙。
- `compare_cards.png` —— 左:按物理卡 util(card3:② 空闲 vs ③ 64%);右:秒/步 90.0 / 43.1 / 37.2。

> 各 timeline 的 x 窗口是**按各自 3 步自动定位的**,绝对跨度不同,**不要跨图直接目测条间距**(会被 x 缩放误导);要比就看 §3 的稳态秒/步,或看 verlorig 红 bar 间距(43)确实 > ours(37)。

---

## 5. "我们的" vs "verl 原版" —— 代码级归因

- `verl/` 是独立 vendored git repo。`fully_async_policy/fully_async_trainer.py` 是**我们改过的未提交版**(`git -C verl diff --stat HEAD`:**+528 / -14**)。
- **upstream(HEAD)原版**:`fit_step` 里 `batch = self._fit_compute_ref_log_prob(batch)` —— ref **同步**;`separation/utils.py:37` 把 `Role.RefPolicy` 放进 training 池 → **ref colocate 在训练卡**。upstream **没有** RefService / `_fit_compute_ref_log_prob_async` / async_ref(`git -C verl log -S ...` 搜不到)。
- **我们加的(都挂 `fully_async_split.*` 命名空间)**:
  1. `RL/main_grpo_fully_async_split.py`:把 `Role.RefPolicy` 路由到独立 `ref_pool`(card3)。
  2. `fully_async_trainer.py` 里加:`RefService`(异步微批 ref 服务)、`_fit_compute_ref_log_prob_async`(ref_future 预取)、`async_ref`(默认 True)、`ref_service`(默认 False)。
- ④ baseline 怎么变回"真 verl 原版":`run_grpo_fully_async_nosplit_a800_a4000.sh` 里 `ASYNC_REF=False` → `_submit_ref_log_prob` 返回 None(trainer.py:884)→ ref 走同步 inline(trainer.py:912)。**不传 async_ref=False 会默认 True,那不是纯原版**(那是"我们的预取但 ref 仍 colocate",作为 ablation 数据存在 `logs/effbench_verlfullyasync_*`,实测也≈37s)。

---

## 6. 统一埋点机制(本次新增,可复用)

- `verl/utils/profiler/performance.py`:给共享 `_timer` 加了 phase 事件钩子。driver 进程调 `enable_phase_events(os.environ["PHASE_EVENT_LOG_PATH"])` 后,该进程所有 `marked_timer(...)` 阶段自动按统一 schema 落 `*_phase.jsonl`(`{ts,t,ev:phase_start/phase_end,phase}`);worker 不调用 enable → 零开销零污染。
- 已接入:`fully_async_trainer.__init__`、`main_ppo_sync.PPOTrainer.fit`、`one_step_off/ray_trainer.fit`(后者本轮没用,无害)。
- GPU util 由各 launcher 内 `nvidia-smi -lms 1000` 采到 `*_gpu.csv`(物理卡真值,与 trainer 无关)。
- 绘图:`python3 eval/plot_effbench.py`(**必须 python3,.venv 没装 matplotlib;.venv 有 vllm**)。变体定义在脚本顶部 `VARIANTS`(run 名前缀 + 角色→卡映射)。

**数据文件**(`logs/`):`effbench_{serial_rollout4, verlorig_rollout4_steps5, ourasync_rollout4_steps5}_{gpu.csv, phase.jsonl, timing.txt, meta.env}`。

---

## 7. 已知局限 & 下一步(交给下一个 agent)

1. **小样本噪声**:5 步去预热只剩 2-3 个稳态间隔(ours steady gap=[39.4, 35.0] → 37.2;verlorig=[43.1,46.4,40.4] → 43.1)。**建议三者各重跑 ~15 步**(launcher 第 2 个参数:`bash RL/run_..._split_....sh 4 15`)拿稳数再定稿。
2. **trigger=1 压制了我们的优势**:每步同步 → 流水线跑不到前面(提前采的样本超 staleness 被砍)→ ref 解耦红利没完全释放。**建议补一组 trigger=4**(三者统一)再比,③ 对 ② 的差距预期更大。注意 serial 无 trigger 概念、本就每步同步,这一项是架构固有差异,需标注。
3. **绘图 x 窗口未统一尺度**,跨图目测会误导(见 §4 注)。可选:改 `plot_effbench.py` 让三图共用同一秒/格尺度。

**复现/续跑的坑(务必照做)**:
- ③ smoke 的 launcher 默认 `STALENESS_THRESHOLD=0.1` 配 trigger=1 会**死锁**(旧样本立刻超阈、rollouter 不补生成、trainer 永等)。本轮用 `STALENESS_THRESHOLD=2.0` 跑通(只影响样本新鲜度,不影响时间结构)。
- ② launcher 已内置修复:权重传输 bucket 默认 2GB 在 A4000 OOM → 设 `update_weights_bucket_megabytes=512` + `gpu_memory_utilization=0.45`;hydra searchpath 是相对路径 → CLI 绝对覆盖;它不跑 reward 迁移 → 显式传 `reward.custom_reward_function.*`。
- 启动:`export RUN_GROUP=effbench_xxx; export PHASE_EVENT_LOG_PATH=logs/<run>_phase.jsonl;`(②launcher 自带 PHASE 路径)然后 `nohup bash RL/<launcher>.sh 4 <steps> > logs/xxx.out 2>&1 &`。
- **清场只按精确 PID 杀主进程**(ray driver 死会带下 actor);**严禁广义 `pkill -f`**——共享机器,**card0 / card2 是别人的(refusal 项目等),绝不能碰**。
- 跑前 `nvidia-smi -i 1,3,4,5,6,7` 确认我们这 6 张卡空闲。
