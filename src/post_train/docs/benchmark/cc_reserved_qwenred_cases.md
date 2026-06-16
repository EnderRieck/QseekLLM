# CC-reserved · Qwen 最红家族 Case 研究（Qwen 错 / 我们对）

> 选材方法：以 `cc_reserved_family_heatmap.png` 为依据，挑出 **Qwen（Base/Instruct）准确率最低（最红）、而我们 S4 最高（最绿）** 的 6 个家族；
> 再从 798 题候选（两个 Qwen 都错、我们对）`qwen_base_instruct_wrong_ours_right_cc_reserved_full.jsonl` 中筛 case。
> **已剔除“格式等价却被判错”的假赢**（如 `Solve |x+7|=17` Qwen 答 `10,-24` 与 gold 等价、有理不等式区间答案与 gold 完全一致），只保留真实的数学/推理错误。

## 差距概览（Pass@1，来自热力图）

| 家族 | Qwen Base | Qwen Instruct | **我们 S4** | 候选题数 |
|---|---:|---:|---:|---:|
| absolute_value_schema | 0% | 26% | **90%** | 47 |
| rational_inequality_schema | 10% | 24% | **100%** | 61 |
| quadratic | 13% | 11% | **58%** | 90 |
| comparison | 22% | 46% | **89%** | 57 |
| matrices | 37% | 22% | **61%** | 55 |
| derivative_schema | 20% | 62% | **100%** | 25 |

---

## Selected Cases

### Case 1 — comparison（答案定向反了）`idx 1176`
- **题**：Compare 4.03 and 13.92 using `<` or `>`.
- **GOLD**：`4.03 < 13.92` ｜ **我们 S4**：`4.03 < 13.92` ✅
- **Qwen Base**（错，final=`<`）：只给裸符号 `<`，未带数对，判错。
- **Qwen Instruct**（错，final=`>`）：推理全对——“13.92 的整数部分更大… it is greater than 4.03”，却把结论 box 成 `\boxed{>}`（读作 4.03 > 13.92）。**会算不会定向**：正确比较后符号方向写反。

### Case 2 — quadratic 判别式（不答所问 / 不收口）`idx 3119`
- **题**：Classify the roots of −x²+x+110=0 using the discriminant.
- **GOLD**：`two distinct real roots` ｜ **我们 S4**：✅
- **Qwen Base**（错，final=`-10, 11`）：直接给出两个根，**答了“求根”而非“分类”**。
- **Qwen Instruct**（错，final=`ANSWER`）：正确算出 D>0、根 x=−10/11，结论也对（“two distinct real numbers”），但一直纠结输出格式、**始终没 box 出干净答案**，最终被截断成占位符 `ANSWER`。

### Case 3 — matrices Cramer 法（长链空转 → 截断）`idx 2698`
- **题**：Solve −3x−3y=−12; x+3y=8 using Cramer's rule.
- **GOLD**：`x=2, y=2` ｜ **我们 S4**：✅
- **Qwen Base**（错，final=`y = 2`）：只报了 `y=2`，**漏掉 x**。
- **Qwen Instruct**（错，final=`-24 +6y =`）：中途已正确得到 `x=2, y=2`（“So that's correct”），却继续用代入法反复自我复核，**token 耗尽停在半句** `-24 +6y =`。典型**长 CoT 不终止**。

### Case 4 — absolute_value（漏解）`idx 2`
- **题**：Solve |x−18|=15.
- **GOLD**：`x=3 or x=33` ｜ **我们 S4**：✅
- **Qwen Base**（错，final=`33, 3`）：格式问题被判错（内容实为等价，弱真赢，仅作对照）。
- **Qwen Instruct**（错，final=`33`）：只给一个根 33，**漏掉另一解 3**。

### Case 5 — derivative 极值（答不完整）`idx 1442`
- **题**：Find the local extremum of f(x)=(x+2)²+3.
- **GOLD**：`minimum 3 at x=-2` ｜ **我们 S4**：✅
- **Qwen Base & Instruct**（均错，final=`3`）：只报极值 `3`，**缺“是极小值、在 x=−2 取得”**，回答不完整。

---

## 总结

在这些题型上，Qwen 的“错”主要不是不会算，而是三类**收口 / 输出失败**：

1. **答案定向写反**：正确推理后符号/方向给反（Case 1）。
2. **长 CoT 空转、不终止、超长截断**：算对了却反复自我复核，耗尽 token 停在半句（Case 2/3）。
3. **答不完整 / 漏解**：只报部分结果，缺另一解或缺定性（Case 4/5）。

我们用合成 CoT 数据训练出的模型，在这类“机械演算 + 干净收口”的题上给出**简洁、定向正确、格式规整**的答案，反超 Qwen3-1.7B Base/Instruct。该结论与既有文档中反复诊断的 Qwen Instruct “长链空转”“收口脱节”问题一致。

---

*配图：`cc_reserved_family_heatmap.png`（家族热力图）、`cc_reserved_rl_delta.png`（RL 增量）。完整候选样本：`qwen_base_instruct_wrong_ours_right_cc_reserved_full.jsonl`（798 条）。*
