# Compute_Cot 数据质量审计报告

> 审计日期: 2026-06-09 ｜ 数据位置: `/data/zilu/fastrl/Compute_Cot/data` (上一轮用仓库自带导出接口产出的 550k)
> 复跑工具: `scripts/qc/quality_report.py`、`scripts/qc/audit_reasoning.py`、`scripts/qc/inspect_samples.py`、`scripts/qc/build_audit_batches.py`

## 结论速览

| 维度 | 结论 |
|------|------|
| 格式 / 答案校验 | ✅ 满分。550k 全部 `verified=true`、格式合规、`boxed==answer`、无脏片段/空 trace/编号列表。 |
| 样本去重 | ❌ 严重。train 去重后仅 ~26.5 万唯一题(标称 55 万)；s3 重复率 **73.5%**。 |
| train/test 泄漏 | ❌ 超标。id_test **34.4%**、val **29.4%** 题面与 train 完全重合(违反 CLAUDE.md 约束)；仅 template_ood 0%。 |
| 推演正确性(验证器盲区) | ⚠️ 268 源中确认 **2 个**系统性推演 bug，均在基础算术域。其余抽检通过。 |

## 一、重复 & 泄漏 (quality_report.py)

各 split 内部 user 题面重复率: s1 42.9% / s2 31.0% / **s3 73.5%** / s4 48.8% / s5 21.4%。
跨集合泄漏(与 train 完全重合): val 29.4% / id_test 34.4% / extrap_ood 9.2% / template_ood 0%。
根因: 生成器无去重；各 split 仅靠"换 seed"区分，小空间题目必然碰撞。
附带: s3 列表里 `expression_rewrite.collect_like_terms` 写了两次 → 拿到 2× 配额 (11,111 vs 5,556)。

## 二、推演审计 (确定性 + 12×haiku LLM 扫 + 人工复核)

三层方法:
1. **确定性数值审计** (`audit_reasoning.py`): 抽 <think> 中纯数字等式实算比对。全量 268 源 → 仅 `decimal_division_by_decimal` 命中。盲区: 只查"局部数字等式", 看不到含变量/局部成立但整体不自洽的错误。
2. **LLM 定性扫** (12 个 haiku subagent, 每源 4 样本): 覆盖全部 268 源。能抓"全样本系统性"错, 但条件触发型(仅特定符号/边界出现)会漏。
3. **人工复核** 所有标红项 + 算术域逐字精读。

### 确认的推演 bug (2)

**① `arithmetic.decimal_division_by_decimal` — 严重, 100% 样本**
- 现象: 两边同乘 10^k 后给出**放大 10 倍的错误整数商**, 再用假的"补小数点"步骤圆回正确答案。
- 例: `168 ÷ 140 = 12`(实为 1.2)、`4424 ÷ 560 = 79`(实为 7.9)、`2775 ÷ 250 = 111`(实为 11.1)。
- 危害: 每条都在教模型一个**错误的算术事实** + 不存在的伪流程。三方一致确认。

**② `arithmetic.fraction_division` — 中(呈现断层, 非算式错误), 仅负除数样本(~半数)**
- 订正(读源码后): 之前误判为"操作数错/深度不自洽"。实际每个单步算式**都为真**——Python `Fraction` 把符号恒放分子(`n2<0, d2>0`), 故 "11×(-8)=-88" 的 `-8` 确是倒数分母, 操作数没错。
- 真正的毛病: 当负号落到分母乘积 `raw_den` 上时, 末行 `simplify_step_text`→`fmt_raw_fraction` 在 `den<0` 时悄悄把符号翻到分子, 于是上面写 `-99/-88`、下一行却跳成 `Simplify 99/88`——**"负负得正"那一步被吞掉**, 形成文本断层。对弱模型是"跳步", 非"教错数学"。
- 正除数样本干净。确定性审计(每个单步局部为真)+ 该批 haiku(抽到干净样本) **均漏报**, 仅人工逐字发现 → 印证"条件触发型 bug 需定向探测"。

### 复核后排除的误报 (2)
- `plane_geometry.similar_triangles`: agent 记串内容, 实际 `15/60 = 16/x` 正确。
- `analytic_geometry.midpoint`: agent 嫌 `(12 - 9)/2`, 但 `12 + (-9)` 渲染成 `- 9` 正是数据规范要求的格式, 正确。

## 三、coverage 警告 (重要)

"确认 2 个" 是**下界不是上界**。LLM 扫每源只看 4 样本, 像 `fraction_division` 这类**条件触发** bug, 若该源被抽到的 4 条恰是干净 case 就会漏 (它本身就被对应 agent 判了 clean)。要更彻底需: 加大每源样本量, 或对每个生成器做**边界/符号定向探测**(强制负数、0、跨零借位等)。两个已确认 bug 同根: 结构化 `trace`/`answer` 正确, 错在**自然语言渲染层**, 修复定位于 `mathgen/domains/arithmetic_core.py` 两个生成器的文本输出。

## 四、生成器修复记录 (2026-06-09, 已改并通过验收)

> 改的都是**自然语言渲染层**, `trace`/`answer`/verify 逻辑未动。
> 验收: `PYTHONPATH=. python scripts/check_sources.py` → 268 源, verification/validator/blemish **全 0, PASS**。
> 另各自抽 3000~4000 条做程序化一致性检查, 0 问题。

| # | 文件:函数 | 改动 | 验证 |
|---|----------|------|------|
| ① | `mathgen/domains/arithmetic_core.py: gen_decimal_division_by_decimal` | 丢弃错误的"乘 10^shift→整数商=quotient_scaled"叙事, 改为"去小数点→`dividend_scaled ÷ divisor_scaled = quotient_scaled`(真整数除)→按 `places_q=places_dividend-places_d` 补小数点"; 并拒绝任何 `%10==0` 的操作数(否则尾零被 `fmt_decimal_from_scaled` 抹掉, 显示位数与叙述不符) | 3000 条: 整数除法行全为真整数且相等, 显示位数==叙述位数 |
| ② | `mathgen/domains/arithmetic_core.py: gen_fraction_division` | 取倒数时把符号归一到分子(`rd>0`), 使分母乘积恒正, 三行(倒数/乘积/simplify)符号一致, 不再吞"负负得正" | 3000 条(含负 1310): 分母恒正, simplify 分数文本与乘积一致 |
| ③ | `mathgen/domains/case_split_core.py: gen_split_by_piecewise_condition` | 新发现(验收脏片段扫描命中, validate 与 haiku 均漏): 问题串直接 f-string 拼 `{a}x + {b}` → `1x + -6`(脏片段`+ -`、未处理系数1/0)。改用 `fmt_poly` 渲染线性式, 并补代入步 `f(7) = (-9)×7 - 8 = -71` | 4000 条: 0 脏片段, 渲染干净 |

## 四点五、教学 CoT 颗粒度重构 (2026-06-09, 据 v3 实测诊断报告)

`fastrl/outputs/sft/compute_cot_arithmetic_cot_diagnosis.md` 用 v3 step2400 实测发现：低分根因是教学把**多位算术压成单行断言**，模型在那步只能瞎猜（decimal_mult 0%、long_mult 0.8%、long_div 试商 77% 过大）。已据此重写 arithmetic_core 教学结构，**原则：多位运算一律逐步 worked**：
- long_mult 求和 → 逐对逐列进位 worked 加法 (`emit_worked_addition`/`emit_partial_sum`)
- decimal_mult → 内联完整 long-mult (`emit_long_multiplication`)
- decimal_div(by_decimal/by_integer) → 内联完整 worked long-division
- long_div 试商 → 显式约束 `q×d≤current<(q+1)×d` + 自检
- subtraction → 连续零借位显式级联(中间各 0 变 9)
验证：check_sources 全 0 PASS；4 源各 6000 样本旧单行断言残留=0/verified=0失败/validate=0失败。
**注意：此前的 `data/clean/` 220k 是旧生成器产物，必须用修好的生成器重生成。**

## 五、仍待办 (数据层, 未做)

1. **重新生成全部数据** — 现有 `/data/zilu/fastrl/Compute_Cot/data` 的 550k 仍是修复前产物, 三个 bug 都还在, 必须用修好的生成器重跑。
2. 生成时**全局去重**(当前 train 真唯一题不到标称一半)。
3. **先建无泄漏 test/val, 再从剩余空间采 train**(当前 id_test 泄漏 34%)。
4. 修 `generate_data.sh` 里 s3 的重复 source + 补 s5/val/test 缺失的 difficulty 标签。
5. 给生成器加**推演自检**(不止校验答案); 对每个生成器做**符号/边界定向探测**(强制负数/0/跨零借位等), 因为随机抽样会漏条件触发型 bug(如本次 ②)。
