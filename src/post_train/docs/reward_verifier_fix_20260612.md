# 判分器(reward.py)误判修复 · 2026-06-12

## 背景

审计 f2_step_9858 终评(`final_eval`)math500 的 pass@8 dump 时发现:92 道
"贪心错、采样救回"的题里 31 道的 boxed 答案与 gold 不一致却被判对,例如
`6-3i` vs `6+9i`、`4\sqrt{5}` vs `4`、`8` vs `8\pi`、`1+2i` vs `1+274i`、
`y=11x-10` vs `y=6x-10`。

## 根因

`data_pipeline/reward.py` 兜底调用 `math_verify` 库时把**裸字符串**直接传给
`parse()`。该库对不在 `$...$`/`\boxed{}` 数学环境内的字符串不走 LaTeX 解析,
退化为"抽取其中一个数字":`parse('6+9i') -> [6, '6']`。于是 sympy 等价比较
退化成"某个数字相等即判对"。

## 修复(均在 `data_pipeline/reward.py`)

1. `_mv_parse_wrapped()`:进 `math_verify.parse` 前包上 `$...$`;
2. `verify_answer` 的 math_verify 兜底路径改为统一走 `_verify_single`
   (精确 → 数值容差 → 元组逐分量 → sympy 等价),元组守卫对该路径生效;
3. `_verify_single` 先剥离 `\left`/`\right` 再做元组判定;
4. 新增进制下标守卫:`40_{10}` vs `40_9` 要求数字与底数都一致;
5. 回归用例补进 `__main__` 自测(23 例全过):`python -m data_pipeline.reward`。

## 修复后离线重打分(f2_step_9858 终评 dump,数字"原→新")

| bench | pass@1 | pass@8 |
|---|---|---|
| math500 | 0.0560 → 0.0540 | 0.2400 → **0.2100** |
| cc-reserved | 0.7182 → **0.6942** | 0.8440 → 0.8225 |
| gsm8k | 0.1600 → 0.1600 | 0.4299 → 0.4291 |
| cmath / gsmplus / svamp | 不变 | 不变 |

抽查翻转判例确认全部是真错答被纠正(斜率错、坐标分量错、根号下数字错)。

## 影响范围与遗留事项

- 该函数是 RL 奖励 / async_eval / final_eval 共用入口,**修复发生在
  sft_s3r1 训练期间**:此前所有用 math_verify/compute_cot style 报出的
  历史数字(含 heldout 的 per-source acc)偏高,跨日对比需以重打分为准;
- [ ] 对 sft_s3r1 的 `eval_dumps/` 历史 heldout 重打分;
- [ ] 抽查 R1 蒸馏数据按 `correctness_math_verify` 字段选 gold 轨迹是否
  受同类问题影响(字段若来自上游数据集则风险较低);
- [ ] 另一独立发现:math500 上采样救回的题大半是"小整数答案蒙对"
  (66/92 仅 1/8 条对,gold 多为 0–10),RL 训练集需按答案可猜性过滤/降权。

## 追加(同日晚):mcq 裸字母假阳修复

**现象**(用户在 S4 step200 dump 发现):gaokao-mathqa 截断输出(无 \boxed)被判对——
旧逻辑无格式时退化为"扫尾部 80 字符找 \b[A-D]\b",数学正文 `sinA cosB` 里的 'A'
撞上 gold=A 即假阳。

**修复**(`data_pipeline/reward.py`):无格式时不再扫裸字母,只认
① 明确作答声明 `_MCQ_DECL`(故选X/答案是X/正确答案X/The answer is X,取最后一处防中途改口);
② 末行单独字母 `_MCQ_LASTLINE`(`(C)`/`C.` 之类)。
有 \boxed/#### 时维持原逻辑。新增 8 条 mcq 回归用例,自测 31/31 过。

**影响面**:仅 mcq style(heldout v2 里 gaokao-mathqa 347 题 + kaoyan 子集少量)。
S4 step200 dump 已用新版重判:翻转 2 条,gaokao-mathqa 0.029→0.023,整体 0.3829→0.3825,
metrics.jsonl 同步改(带 note)。step400 起 worker 子进程自动 import 新代码,口径一致。
