# eval —— 过程跟踪 / 异步评测（hybrid_sft_rl_design.md §6/§7 · CLAUDE.md 过程跟踪原则）

不阻塞训练（A800 card1）的异步评测，在 A4000(card2,3) 上对训练外、无泄漏的 held-out 生成、判分、
**dump 完整输入输出** + per-source/per-difficulty 准确率，供追溯（v3 诊断正是靠这种 dump 挖根因）。

## held-out 评测集（全面谱，2018 条）
`eval/heldout.jsonl`，覆盖：算术分源(compute_cot 505) → 小学(gsm8k 200/cmath 200) →
高考(填空 118/选择 250) → 竞赛(competition-math 250) → 中文知识(cmmlu 279) → 推理(BBH 216)。
中英双语；5 种验证 style：compute_cot/gsm8k/math_verify/mcq/exact_match。
```bash
python -m eval.build_heldout --out eval/heldout.jsonl    # 重建
```

## 异步评测（监视 checkpoint）
```bash
# 在 card2,3 跑，监视 SFT 保存目录，新档一出就评
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  python -m eval.async_eval --ckpt-dir <SFT保存目录> --watch --device cuda:0
# 只评某一步: --step N
```
产出（`<ckpt>/eval_dumps/`）：
- `step_N/heldout.jsonl` —— 每题 {source,difficulty,prompt,generation,gold,correct,has_format}（**追溯命根**）
- `metrics.jsonl` —— 每步 {acc, format_rate, avg_gen_chars, acc_by_source, acc_by_difficulty}
- `tb/` —— tensorboard（eval/acc 等，与训练曲线并看）

## 终评（阶段末完整评测,2026-06-10 新增 `final_eval.py`）
```bash
# Pass@1+Pass@8,6 benchmark ~10.7k 题,A4000 双卡
.venv/bin/python -m eval.final_eval --ckpt <ckpt>/global_step_N --gpus 2,3
# 冒烟: --benchmarks svamp --limit 8 --k 2
```
benchmark：**cc-reserved**(Compute_Cot 保留集 id_test 每子源采20,数学基本功,per-source 切分) /
gsm8k(1319) / math500(level 分级) / gsmplus(7 类扰动) / cmath(grade·步数·位数) / svamp(300)。
产出 `<ckpt>/final_eval/`：summary.{md,json} + 每 benchmark 全量 IO dump jsonl。
判分同 async_eval(data_pipeline.reward);Pass@8=贪心∪8 个 T=0.8 采样任一对。
已冒烟验证(2026-06-10,base 模型):双卡分片/转换复用/判分/维度切分/产出全通。

## 已验证（2026-06-09）
对 3 步 SFT smoke checkpoint：权重加载 missing=0、5 种 style 判分正常、dump+双维指标+tb 全部产出。整条跟踪管线通。
