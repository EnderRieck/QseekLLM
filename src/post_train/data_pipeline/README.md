# data_pipeline —— 统一训练数据管线（D1）

实现 `docs/hybrid_sft_rl_design.md` §4/§5：把所有数据源统一成单一训练格式，做验证器路由、
全局题面去重、评测隔离，产出 SFT 池 / RL 池 / 通用 SFT 池。

## 产出（`/data/zilu/data_unified/`）

| 文件 | 内容 | 规模 |
|------|------|------|
| `train_sft.jsonl` | 数学 SFT（①worked 算术 + ②题目理解，带 `gold_response`） | **1,412,378** |
| `train_rl.jsonl` | 数学 RL（带 `reward_model.ground_truth/style`） | **1,411,265** |
| `train_general_sft.jsonl` | 通用 SFT（③防退化，中英，纯回复无 think） | **406,975** |
| `manifest.json` / `manifest_general.json` | 各源数量 / 去重 / 泄漏统计 | |

## 统一格式（每条 jsonl）

```json
{
  "prompt": [{"role":"system","content": <引导 <think> 逐步 + #### \boxed{}>},
             {"role":"user","content": <题面>}],
  "data_source": "compute_cot:arithmetic.long_multiplication",
  "ability": "math" | "general",
  "use": "sft" | "rl" | "both",
  "reward_model": {"ground_truth": "<可验证答案>", "style": "compute_cot|gsm8k|math_verify|math_dapo|none"},
  "gold_response": "<think>\n逐步(worked)推演\n</think>\n#### \\boxed{答案}",   // SFT 用；RL-only 为空
  "extra_info": {"difficulty": "...", "source": "..."}
}
```
- `use=both` 的样本同时进 SFT 与 RL 池（同题，SFT 喂 gold、RL 喂 prompt+答案）。
- 通用 SFT：`ability=general`、`gold_response`=纯回复（无 think）、`reward_style=none`。

## 模块

| 文件 | 职责 |
|------|------|
| `format.py` | 统一 Record + 系统提示(中英) + `<think>/boxed` 包装 + 答案抽取（boxed 用**平衡括号**匹配，处理嵌套）|
| `adapters.py` | 数学源适配 + `SOURCES` 注册（compute_cot/gsm8k/metamath/orca/openr1/numina/deepscaler/infinity-math/**openthoughts3-math/bespoke-stratos/calc-ape210k**/big-math/dapo）|
| `general_adapters.py` | 通用源适配（no_robots/dolly/coig-cqia/dynamics/tulutalk/infinity/**chinese-r1**）|
| `build.py` | 数学池构建：全局去重 + 评测隔离 + 分 SFT/RL 池 |
| `build_general.py` | 通用池构建：大源限量 + 去重 |
| `reweight_sft.py` | 基础 SFT 重采（易题为主、竞赛/长推理降权陪练、①算术骨干）|
| `to_verl_parquet.py` | jsonl → Verl parquet（SFT=messages / RL=prompt+reward_model）|

### SFT vs RL 分配原则（关键）
- **验证过答案的源 → 可进 RL**（compute_cot/gsm8k/metamath/orca/numina-有答案/openr1/deepscaler/infinity-math/**calc-ape210k**/big-math/dapo）。
- **蒸馏未验证答案的源 → 仅 SFT**（**openthoughts3-math / bespoke-stratos**：学强模型长推理风格，但答案可能错，不拿去当 RL 奖励）。
- openthoughts3/bespoke 偏难长 → 基础阶段低权重（reweight 限 5万），全量留给阶段 2-3。

## 用法

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 镜像直连
python -m data_pipeline.build --out /data/zilu/data_unified --cap 0          # 全量数学池
python -m data_pipeline.build_general --out /data/zilu/data_unified           # 通用池
# 冒烟：--cap 500
```

## 关键设计点（已落实）

- **验证器路由**：`reward_style` 按源映射到 Verl `reward_score/`（gsm8k 的 ####、math_verify 的 sympy 等价、math_dapo 的 MATH_v2、compute_cot 自有精确）。
- **奖励整形**：主信号仍是答案正确；另加 format bonus、有界 thinking 长度奖励和循环重复惩罚，并在 RL metrics 暴露分项。
- **全局题面去重**（归一化后 md5）：first-wins，`SOURCE_ORDER` 让 Compute_Cot 等优质源优先占题。本次去重命中 **78 万**（big-math/numina/metamath 题源高度重叠）。
- **评测隔离**：排除 gsm8k-test / cmath / competition-math / agieval-gaokao-math / math-beyond 的题面（**9.3 万**），防泄漏（呼应 Compute_Cot 审计教训）。
- **worked CoT 保真**：Compute_Cot 的逐步 worked 推演原样进 SFT（①算术执行力底座）；外部跳步解照用进 ②（解题理解力）。

## 待办（下一步，D2+）
- 训练侧 dataloader：按 `ability`/`source`/`difficulty` 切桶 + 动态采样（接 eval 驱动课程）。
- 转 parquet（Verl 友好）+ 难度分层索引。
- 通用大源(tulutalk/infinity)的 reward/质量过滤可再细化。
- 加 openthoughts3 的 math 子域（可选，stage3+）。
