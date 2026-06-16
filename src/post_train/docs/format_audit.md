# 数据格式审计报告（评测集 + 训练数据）

> 2026-06-10。方法：脚本统计 + **逐源人工抽样核对**（避免启发式误判/漏判——本会话已有 orca「70%错」纯属粗筛误报的教训）。
> 范围：`eval/heldout.jsonl`（评测）、`train_sft_foundation_8k.parquet`（SFT 训练）、`train_rl.parquet`（RL 池）。
> 判分代码：`data_pipeline/reward.py`。关联 [data_audit_and_architecture.md](data_audit_and_architecture.md)、[eval_tracking.md](eval_tracking.md)。

## 结论速览

| 对象 | 格式健康度 | 要点 |
|---|---|---|
| **训练 SFT 数学 gold** | ✅ **优** | 10 源 100% `<think></think> #### \boxed{}` 完整；boxed 脏尾巴仅 0.08% |
| **训练 RL gold** | ✅ **良** | 无空 gt；脏 gt 0.25%（真复杂答案，非 bug） |
| **训练 system prompt** | 🟡 小瑕 | 数学统一英文 prompt；numina 198 条混了中文 prompt 且夹带非数学题 |
| **评测 gsm8k / cmath / compute_cot** | ✅ 可靠 | gold 干净、判分正确——**本阶段主看这三个** |
| **评测 competition-math** | 🔴 **gold 全坏** | 250/250 LaTeX 尾巴，判分不可信 |
| **评测 gaokao-mathcloze** | 🔴 判分缺陷 | 多空(;分隔)+中文 prose，math_verify 解析不了→必近 0 |
| **评测 bbh / gaokao-mathqa / cmmlu** | 🟡 旁观 | 判分逻辑对，但格式/语言错配，分数不可作能力信号 |

---

## 一、评测集（`eval/heldout.jsonl`，2018 题 / 8 源）

### 判分路由（`reward.py`，已核对逻辑正确）
- `mcq`：从输出抽字母 A-D，gold 用 `_to_letter` 把序号 0-3 / `(A)` 归一成字母再比 → **逻辑正确**（gold "3"→"D"，0-indexed）。
- `gsm8k`/`compute_cot`：`extract_pred`(平衡括号 boxed / `####`) 后精确比 + 数值容差(1e-6) + sympy 回退。
- `math_verify`：sympy 符号等价（处理 LaTeX/分数/区间/集合）。
- `exact_match`(bbh)：抽 pred 后字符串精确比。

### 逐源结论
| 源 | n | style | gold 健康 | 结论 |
|---|---|---|---|---|
| **gsm8k** | 200 | gsm8k | ✅ 干净(整数) | **可靠**，本阶段主指标 |
| **cmath** | 200 | math_verify | ✅ 干净(数值) | **可靠**(中文但数值判分) |
| **compute_cot** | 505 | compute_cot | ✅ 干净 | **可靠**；18 条「脏」经人工核对是**真复杂答案**(`vertex (7/6,71/12)`/区间/`quotient..remainder`)，非 bug；但这类 prose 答案数值验证器可能漏判(轻微低估) |
| **competition-math** | 250 | math_verify | 🔴 **250/250 坏** | gold 含 LaTeX 尾巴 `2}.\n\end{align*`(真答案 `2`)。根因：建评测集脚本用**贪婪正则**从 MATH solution 抽 boxed，未用平衡括号。判分不可信，**那条 ✅ 是假阳性** |
| **gaokao-mathcloze** | 118 | math_verify | 🔴 判分缺陷 | 大量**多空答案**(`$\frac{1}{221}$;$\frac{1}{17}$` 分号分隔)+ 中文 prose 答案 → math_verify 当单表达式解析→失败→**必近 0**(非能力问题) |
| **gaokao-mathqa** | 250 | mcq | ✅ gold 干净(序号) | 判分对，但**中文 MCQ 模型大量交白卷**(空 boxed)→近 0；旁观 |
| **cmmlu** | 279 | mcq | ✅ 干净(字母) | 判分对，中文 MCQ 未训→旁观；4 选 1 有蒙对成分 |
| **bbh** | 216 | exact_match | 🟡 gold 带括号 | gold `(B)`，模型常吐裸字母 `B`→`_exact` 对不上→**漏判低估**；且通用推理被强套 boxed；旁观 |

### 评测侧待修 → ✅ 已在 v2 重建中解决（2026-06-10，见 [eval_tracking.md](eval_tracking.md)）
1. ~~competition-math gold 抽取换平衡括号~~ → **已修**：`format.py:extract_boxed` 本就是平衡版，旧 heldout 是过期文件；重生成后 **250/250 坏 → 200/200 干净**（含 1 条合法 pmatrix 矩阵答案，人工确认非坏）。
2. ~~gaokao-mathcloze 多空判分~~ → **直接剔除**（语言+多空双错配，不值得修，v2 已移除）。
3. ~~bbh 归一~~ → **直接剔除**（通用推理被迫套 boxed，与本阶段无关，v2 已移除）。

> v2 held-out 转为"能力分解"设计：保留 compute_cot(1848)/gsm8k(500)/SVAMP(300)/GSM-Plus(798)/cmath(200)/competition-math(200,已修)，砍掉 cmmlu/gaokao×2/bbh。新增 SVAMP/GSM-Plus 作"模板过拟合"探针。详见 eval_tracking.md。

---

## 二、训练数据 SFT（`train_sft_foundation_8k.parquet`，118.9万）

### 数学 gold 格式完整性 ✅ 优
逐源核：**10 个数学源 100% 都有完整 `<think>…</think>\n#### \boxed{}`**（compute_cot/orca/metamath/numina/openr1/gsm8k/deepscaler/infinity-math/bespoke/openthoughts3 各 100%）。
> 注：eval 里看到的「45% 不闭合 think」是**模型输出**问题，**训练数据本身干净**，二者勿混。

### boxed gold 干净度 ✅
含 `\end{` LaTeX 尾巴的仅 **679/84.2万 = 0.08%**（numina 551 居多，是源 answer 字段自带 align）。可忽略，不影响训练。
> 早前「orca boxed 70% 错」是粗筛误报——拿 RL 真值对 16.5万题实测 **97.9% 正确**，已作废。

### system prompt 一致性 🟡 小瑕
- 数学源：基本统一英文 prompt「Solve the problem step by step...」。
- **numina 198 条混入中文 prompt**「请一步一步解题...」，且人工抽看发现**其中夹带非数学题**（如「下列哪项没有错别字」中文语言题被当 math）→ 轻微标签/语言污染（0.2%，量小）。
- 通用源：system prompt 多样（tulutalk/no_robots 含大量角色扮演 prompt）——**正常**，指令多样性，非问题。

### 格式分流（非 bug，设计选择）
- proof / 空答案 → 有意 SFT-only（`\boxed{proof}`，~1.4万），见架构文档诊断。
- chinese-r1：通用 prompt + 裸自然语言(剥 think)，0% think / 35% 内联 boxed——格式最杂的源。

## 三、训练数据 RL 池（`train_rl.parquet`，160.6万）✅ 良
- **无空 ground_truth**（RL 必须可验证，达标）。
- 脏 gt（含 `\end`/`\n`）0.25%：人工核对是 **`\begin{pmatrix}…\end{pmatrix}` 多解答案**（真复杂答案，非抽取 bug），但 math_verify 较难判这类——少量题 RL 难拿到奖励，影响小。
- style 路由正确（gsm8k→gsm8k / compute_cot→compute_cot / 余 math_verify）。

---

## 四、总体判断

- **训练数据格式总体健康**：SFT 数学 gold 100% 结构完整、boxed 99.92% 干净；RL gold 无空值。可放心继续训。
- **真正的格式 bug 集中在评测侧**：competition-math gold 100% 坏、gaokao-mathcloze 多空判不了——但这两者已是旁观指标，**对本阶段决策无损害**。
- **本阶段健康度只依赖 gsm8k + cmath + compute_cot**，三者 gold 与判分均已核实可靠。
- 小瑕（numina 198 中文 prompt 夹非数学、RL pmatrix 难判、compute_cot prose 答案轻微漏判）量级都 <1%，记录备查，不阻塞训练。

## 五、待办（按优先级，均不阻塞当前训练）
- [ ] 重建正经 benchmark 时修 competition-math / gaokao-mathcloze / bbh 的判分（见 §一）。
- [ ] 清 numina 那 198 条中文 prompt + 其中非数学题（下一版数据）。
- [ ] （可选）RL 池 pmatrix 类难判答案标注/剔除。
