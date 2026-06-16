# RL 详细日志 + 过程评测(复用采样侧)· 2026-06-13

> 目标(用户 /goal):①完善训练日志,详细到训练后能复盘各种过程现象;②过程评测**不开新卡**,周期性暂停采样侧让其跑 val;③训练前检查数据/参数;④无误即开训。
> 本文档记录实现、接口、验证结果,供交接与手动接管。

## 1. 详细事件日志(可事后复盘)

### 做了什么
TensorBoard 只记录**每个 logged step 的聚合标量**,在 `sync>1` 下一个 step 聚合了多次 actor update,**per-update 粒度被抹平**;ref service 的逐 micro-batch 轨迹此前只在 stdout 飘过、不可结构化分析。

新增 `EventLogger`(`fully_async_trainer.py`),把关键事件写成**带绝对时间戳的 JSONL**:

| 事件 `ev` | 含义 | 关键字段 |
|---|---|---|
| `ref_micro_batch` | ref service 每个 micro-batch 完成 | `batch_id,samples,seqs,assemble_s,submit_s,compute_s,ready_q` |
| `update_actor_start/end` | 每次 actor update 起止(**还原 sync>1 抹平的 per-update 粒度**) | `update_s,local_trigger_step,pver` |
| `param_sync_start/end` | 每次权重同步起止 | `param_sync_s` |
| `validate_start/end` | 每次过程评测起止 | `validate_s,val_before_train` |
| `step` | 每个 logged step 的全 timing 快照 | `timing_s/*` |

每条记录都有 `ts`(epoch 秒)+ `t`(相对启动偏移),**绝不因写日志失败中断训练**(全程 try/except)。

### 接口
- 产物路径:`logs/<RUN_NAME>_events.jsonl`(runner 经 `EVENT_LOG_PATH` 注入)。
- 分析器:`python3 RL/analyze_events.py logs/<RUN_NAME>_events.jsonl`
  输出:per-update 墙钟分布、param_sync、ref 吞吐 + ready_q 深度(0=trainer 饿死)、评测耗时、**ref-compute 被 update 隐藏的并发比例**。每条记录带 `ts` 可用 pandas 自画时间线。

### 验证
smoke `grpo_async_evalsmoke` 事件日志经 `analyze_events.py` 解析正常,成功还原 per-update 粒度(pver=0 两次 [31.8,31.0]s、pver=1 一次 [31.7]s)、ref ready_q min=1(从未饿死)、ref-hidden-ratio。

## 2. 过程评测复用采样侧(不开新卡)

### 做了什么
复用现成的 `use_trainer_do_validate=False` 路径:trainer 的 `_fit_validate` 委托 `rollouter.do_validate.remote()`,用**采样侧自己的 vLLM 引擎**(GPU4-7)在 held-out val 上评,**不额外起卡**。

为避免评测与训练采样抢引擎,新增一对防御式 RPC(`fully_async_rollouter.py`),复用与权重同步相同的 `paused`/`_resume_event` 机制:
- `pause_for_eval(drain_timeout_s=180)`:暂停采样、等在途请求 drain(有超时,**绝不无限挂起**),返回 `drain_remaining`。
- `resume_after_eval()`:恢复采样。

trainer 侧:`pause_for_eval → do_validate →(finally)resume_after_eval`,即使评测抛异常也保证恢复采样。

### 评测成本
val 用 **n=1 贪心**(`do_sample=False`,`@1`),非训练的 n=8。全量 5000 条 4 卡批处理很快(200 条 smoke 仅 ~20-26s)。

### 验证
smoke 中途(active sampling 状态)评测实测:`paused...in-flight remaining=0 → validate 19.3s → resumed`,训练随后继续到正常结束,**无死锁、无报错**。

## 3. 过程评测输入/输出全量落盘(透明可追溯)

### 做了什么
启用 verl 内置 `trainer.validation_data_dir`:每次评测把**全部** val 样本写 `<dump_dir>/<step>.jsonl`,每行含
`input`(题目)、`output`(模型完整生成,含 `<think>`)、`gts`(标准答案)、`score`、`reward`、`correct`、`has_format`、`think_len_tokens`、`think_len_bonus`、`repeat_penalty`。
另设 `log_val_generations=20` 抽样进 TB 便于快速肉眼看。

### 接口
- 产物:`logs/<RUN_NAME>_val_dumps/<step>.jsonl`(`<step>` = rollouter 已喂样本计数,单调、各次评测不覆盖)。
- 与 param_version 的对应:同一时刻 trainer 把 `val-core/*` 指标按 param_version 写 TB,可按时间/顺序对齐。

### 验证
smoke 两次评测 dump 出 `1.jsonl`(step0)+ `272.jsonl`(中途),**各 200 行=全量 val、互不覆盖**,字段齐全(input/output/gts/score/correct/has_format/think_len/repeat_penalty)。文件名 `<step>` = rollouter 已喂样本计数(单调递增,故不覆盖)。

## 4. runner 新增可配置项(`run_grpo_fully_async_split_a800_ref3_a4000.sh`)

| env | 默认 | 作用 |
|---|---|---|
| `TRAIN_FILES` / `VAL_FILES` | smoke 路径 | 正式跑指向 `parquet/train_rl_s4clean`、`parquet/val_rl_s4clean` |
| `TEST_FREQ` | 1000000(关) | 每 N 个 param_version 评一次 |
| `VAL_BEFORE_TRAIN` | False | 训前 step0 基线评测 |
| `SAVE_FREQ` | -1 | checkpoint 保存频率 |
| `VAL_DUMP_DIR` | `logs/<RUN>_val_dumps` | 评测 I/O 落盘目录 |
| `LOG_VAL_GENERATIONS` | 20 | 抽样进 TB 的生成条数 |
| `EVENT_LOG_PATH` | `logs/<RUN>_events.jsonl` | 详细事件日志 |

## 4b. 正式跑踩坑:extra_info 字符串导致崩溃(已修)

- **现象**:正式跑在 `val_before_train` 阶段崩溃,`rl_dataset.py:384` `row_dict.get("extra_info",{}).get("index",0)` 抛 `AttributeError: 'str' object has no attribute 'get'`。
- **根因**:`parquet/{train,val}_rl_s4clean.parquet` 的 `extra_info` 列被存成 **string**(JSON 文本),而 verl 数据集要求 dict。`rl_smoke` 是 struct 所以 smoke 没暴露。`reward_model` 两边都是 struct,不受影响。train+val 都中招。
- **修复**:解析字符串→dict→重存 `struct<difficulty,source>`,生成 `*_rl_s4clean_fix.parquet`(0 解析失败,原文件保留)。正式跑改用 `_fix` 版,`val_before_train` 通过(全量5000评测 ~200s)。
- **根治建议**:数据构建脚本(jsonl→parquet)应直接把 extra_info 写成 struct,而非 JSON 字符串。

## 4c. 正式跑配置(2026-06-13 启动)
- RUN_GROUP=`grpo_formal_fullepoch_s3r1_sync4_stale05_n16_t12`
- group size `rollout.n=16`、采样温度 `rollout.temperature=1.2`(val 仍贪心 temp=0)。
- 数据:`*_rl_s4clean_fix.parquet`;TEST_FREQ=10、SAVE_FREQ=50、val_before_train、全 epoch。
- 全量5000评测实测 ~200s/次;评测指标**按 data_source 分题型**给 reward/correct/has_format@1。

## 5. 改动文件清单
- `verl/verl/experimental/fully_async_policy/fully_async_trainer.py`:`EventLogger` 类、各埋点、validate 包 pause/resume。
- `verl/verl/experimental/fully_async_policy/fully_async_rollouter.py`:`pause_for_eval` / `resume_after_eval`。
- `RL/run_grpo_fully_async_split_a800_ref3_a4000.sh`:eval/dump/event-log 可配置项。
- `RL/analyze_events.py`:事件日志分析器(新增)。
