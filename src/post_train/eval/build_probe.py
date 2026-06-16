"""构建"随手丢"定性探针集（**不打分、看输出**），测模型跳出 think+boxed 模板的行为。

三类（中英混，知识问答中英同题成对，note 带 ·zh/·en 后缀 → 报告里按 note 排序自然相邻对比）：
  A 通用对话/知识问答 → 普通 helpful prompt（看它会不会还硬套 <think>/boxed、语言是否退化）
  B 自编基础数学题   → 数学 prompt（看简单题会不会乱套方程；含经典陷阱/反模板题）
  C 高考风数学题     → 数学 prompt（数列/函数/三角/概率/解析几何，中英对照，测知识点迁移）

数学题带 gold（gold 含分数的走 math_verify，整数走 gsm8k）；问答无 gold（只读生成）。
用法: python -m eval.build_probe --out eval/probe.jsonl   （通常不单独跑，由 build_heldout 并入）
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.format import make_prompt, detect_lang

HELP_ZH = "你是一个乐于助人、友好的助手。"
HELP_EN = "You are a helpful, friendly assistant."

# A 通用对话 + 知识问答（无标准答案，看输出；note 的 ·zh/·en 成对项报告里相邻展示）
CHAT = [
    # 闲聊
    ("用三句话给一个小学生解释什么是彩虹。", "闲聊·彩虹·zh"),
    ("我最近老是熬夜，有什么改善建议吗？", "闲聊·熬夜·zh"),
    ("给我写一句关于猫的俏皮话。", "闲聊·猫·zh"),
    ("Recommend three fun things to do on a rainy weekend.", "闲聊·雨天周末·en"),
    ("What's a good icebreaker question to ask someone you just met?", "闲聊·破冰问题·en"),
    # 知识问答（中英同题成对，直接对比双语表达/知识一致性）
    ("用两句话解释为什么天空看起来是蓝色的。", "知识·天空为何蓝·zh"),
    ("Explain in two sentences why the sky looks blue.", "知识·天空为何蓝·en"),
    ("什么是抛物线？用两三句话解释，并举一个生活中的例子。", "知识·抛物线·zh"),
    ("What is a parabola? Explain in two or three sentences and give a real-life example.", "知识·抛物线·en"),
    ("勾股定理是什么？用一句话说明，并举一个具体例子。", "知识·勾股定理·zh"),
    ("What is the Pythagorean theorem? State it in one sentence and give a concrete example.", "知识·勾股定理·en"),
    ("什么是质数？请列出 10 以内的所有质数。", "知识·质数·zh"),
    ("What is a prime number? List all primes less than 10.", "知识·质数·en"),
    ("为什么会有闰年？", "知识·闰年·zh"),
    ("Why do we have leap years?", "知识·闰年·en"),
    # 续写（测开放生成/想象力，看会不会硬套数学模板）
    ("请续写这个故事的开头：「雷德王还有三小时降临地球……」", "续写·雷德王·zh"),
    ("请续写：「哈基米……」", "续写·哈基米·zh"),
]

# B 自编基础数学题（带 gold，重点测"跳模板/反套方程/陷阱"）
MATH = [
    ("小明有 7 个苹果，吃了 2 个，又买了 5 个，现在有几个？", "10", "反模板·直接算即可,别设x"),
    ("计算 384 × 27。", "10368", "多位数乘法·逐位展开"),
    ("一辆车 3 小时行驶 180 公里，照这个速度，5 小时能行驶多少公里？", "300", "速率应用·先求单位再乘"),
    ("3 个工人 3 小时砌 3 面墙，那么 9 个工人砌 9 面墙需要几小时？", "3", "经典陷阱·答案是3不是9"),
    ("A book costs $12. You buy 4 of them and pay with a $50 bill. How much change do you get?", "2", "英文·钱"),
    ("In a class of 40 students, there are 6 more boys than girls. How many boys are there?", "23", "英文·稍绕,(40+6)/2"),
    # C 高考风（中文 6）
    ("已知等差数列 {a_n} 的首项 a_1 = 3，公差 d = 2，求 a_10 的值。", "21", "高考风·等差数列·zh"),
    ("设函数 f(x) = x^2 - 4x + 3，求 f(x) 的最小值。", "-1", "高考风·二次函数最值·zh"),
    ("方程 x^2 - 5x + 6 = 0 的两个根之和是多少？", "5", "高考风·韦达定理·zh"),
    ("从 1 到 10 的整数中随机取一个，取到质数的概率是多少？", "2/5", "高考风·古典概率·zh"),
    ("计算 sin 30° + cos 60° 的值。", "1", "高考风·三角求值·zh"),
    ("计算 log_2(8) + 2^3 的值。", "11", "高考风·指对运算·zh"),
    # C 高考风（英文 6，知识点与中文组互补）
    ("The sum of the first n terms of an arithmetic sequence is S_n = n^2. Find a_5.", "9", "高考风·Sn求an·en"),
    ("If f(x) = 2x + 3 and g(x) = x^2, find g(f(1)).", "25", "高考风·复合函数·en"),
    ("A line passes through the points (1, 2) and (3, 8). What is its slope?", "3", "高考风·斜率·en"),
    ("Solve for x: 2^x = 32.", "5", "高考风·指数方程·en"),
    ("What is the distance between the points (0, 0) and (3, 4)?", "5", "高考风·两点距离·en"),
    ("A fair six-sided die is rolled once. What is the probability of rolling a number greater than 4?", "1/3", "高考风·概率·en"),
]


def records():
    """返回探针记录（与 heldout 同 schema）。被 build_heldout 复用以并进正式集。
    问答 ground_truth="" → async_eval 视为"自由题"：只展示输出、不进 acc 分母。"""
    out = []
    for q, note in CHAT:
        sysp = HELP_ZH if detect_lang(q) == "zh" else HELP_EN
        out.append({"prompt": [{"role": "system", "content": sysp}, {"role": "user", "content": q}],
                    "ground_truth": "", "style": "exact_match", "source": "probe-chat",
                    "difficulty": note, "ability": "general"})
    for q, gold, note in MATH:
        style = "math_verify" if "/" in gold else "gsm8k"  # 分数 gold 走 sympy 等价
        out.append({"prompt": make_prompt(q), "ground_truth": gold, "style": style,
                    "source": "probe-math", "difficulty": note, "ability": "math"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/probe.jsonl")
    args = ap.parse_args()
    out = records()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"probe: {len(out)} 条 ({len(CHAT)} 问答 + {len(MATH)} 数学) -> {args.out}")


if __name__ == "__main__":
    main()
