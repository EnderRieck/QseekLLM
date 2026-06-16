"""自洽性检查:gold_response 的 <think> 推理结论是否与最终 \\boxed 答案一致。
启发式:box 答案(归一化后)出现在 think 末段 → 一致;数值答案再用 float 匹配 think 末段所有数。
不一致的 dump 出来人工抽读(可能是表述差异而非真错)。
"""
import argparse, json, random, re
from collections import defaultdict
from data_pipeline.format import extract_boxed

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="/data/zilu/data_unified_v2/train_sft.jsonl")
ap.add_argument("--per-source", type=int, default=400)
ap.add_argument("--seed", type=int, default=20260610)
ap.add_argument("--out", default="outputs/think_box_suspects.jsonl")
args = ap.parse_args()

THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def norm(s):
    return re.sub(r"[\s,，$\\{}()（）]", "", str(s)).rstrip(".。").lower()

def consistent(think, boxed):
    tail = think[-600:]
    nb = norm(boxed)
    if nb and nb in norm(tail):
        return True
    # 数值:box 是数,则 think 末段任一数值相等即认一致
    try:
        bv = float(nb.replace("%", ""))
    except ValueError:
        return False
    for m in NUM.findall(tail):
        try:
            if abs(float(m.replace(",", "")) - bv) < 1e-6:
                return True
        except ValueError:
            pass
    return False

rng = random.Random(args.seed)
buckets, counts = defaultdict(list), defaultdict(int)
with open(args.pool) as f:
    for line in f:
        o = json.loads(line)
        src = o["data_source"].split(":")[0]
        counts[src] += 1
        b = buckets[src]
        if len(b) < args.per_source:
            b.append(o)
        else:
            j = rng.randrange(counts[src])
            if j < args.per_source:
                b[j] = o

suspects = []
print(f"{'source':<22}{'n':>6}{'一致':>8}{'无think':>8}{'可疑':>8}")
for src in sorted(buckets):
    n = ok = nothink = 0
    for o in buckets[src]:
        resp = o["gold_response"]
        m = THINK.search(resp)
        boxed = extract_boxed(resp)
        n += 1
        if not m or boxed is None:
            nothink += 1
            continue
        if consistent(m.group(1), boxed):
            ok += 1
        else:
            suspects.append({"source": o["data_source"], "boxed": boxed,
                             "gt": o["reward_model"]["ground_truth"],
                             "q": o["prompt"][-1]["content"][:200],
                             "think_tail": m.group(1)[-500:]})
    bad = n - ok - nothink
    flag = "  <-- 抽读" if bad / max(n, 1) > 0.03 else ""
    print(f"{src:<22}{n:>6}{ok/n:>8.1%}{nothink:>8}{bad:>8}{flag}")
print(f"\nsuspects dumped: {len(suspects)} -> {args.out}")
with open(args.out, "w") as f:
    for x in suspects:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
