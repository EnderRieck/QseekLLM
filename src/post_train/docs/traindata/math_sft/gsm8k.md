# 数据卡片 · openai/gsm8k

> 定位：**英文小学数学评测主 benchmark**（train 可少量用，**test 严禁进训练**）。
> 本地：`/data/zilu/math_sft_raw/gsm8k`（全量，`main` + `socratic` 两 config）

## 总体说明

- **来源**：OpenAI，人工编写的多步小学应用题（grade-school math）。
- **规模**：`main` train **7,473** + test **1,319**；`socratic` 同题但解答为"自问自答"式。
- **语言**：英文。
- **License**：MIT。
- **格式**：`question` / `answer`。answer 为多步自然语言推理，含 `<<计算式=结果>>` 标注，**结尾固定 `#### 最终数值`**。

## 抽样

- Q: `Janet's ducks lay 16 eggs per day. She eats three... bakes four... sells the remainder at $2 each. How much does she make daily?`
- A: `Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day.\n#### 18`

`socratic` 版同题，但 answer 拆成一连串"子问题→答案"。

## 对本项目的评估

- 🎯 **首要用途是评测**：GSM8K 是衡量"小学数学多步推理"的标准 benchmark，应纳入阶段2/3 的**过程评测**与最终汇报（Pass@k）。**test 集必须隔离、绝不进训练**。
- ✅ `#### 答案` 格式与我们的 `#### \boxed{}` 接近，抽取答案做自动判分很方便。
- ⚠️ train 仅 7.5k，量小；若用于训练只作少量高质量英文应用题补充，主力仍是 orca-math / metamathqa。
- 🔎 `<<...>>` 计算标注可在喂训练时剥离或转写；评测时只需解析 `####` 后的数值。

## 审计补充（2026-06-10 全量复核）

**实际使用现状**：train 7,473 全量入池(SFT+RL)，test 1,319 已隔离(0 泄漏，实测验证)。

- ✅ 答案抽取(`#### N`)100% 覆盖；在训切片 boxed 不一致仅 0.2%。
- 🔎 **`<<a+b=c>>` 计算器标注被洗掉了**(wrap_think_boxed 统一剥除)：98.7% 样本带平均 3.17 个标注，本是现成的"逐步计算监督"，可转写成 worked 步骤而非直接删除——对"多位数乘法弱"的短板是对症素材。
- GSM-Plus(2,400)是 gsm8k **test** 的扰动版,与训练池 0 重合(实测)，评测安全。
