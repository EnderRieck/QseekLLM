"""训练数据自检:用评测同一套 reward.verify_answer 解析 gold_response 的 box 答案,
与 ground_truth 比对。gold 自身判不对 = box 解析/答案字段有问题。
每个源随机抽 N 条(seed 固定),输出 per-source 通过率 + 失败样例。
"""
import argparse, json, random, sys
from collections import defaultdict
from data_pipeline.reward import verify_answer, extract_pred

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="/data/zilu/data_unified_v2/train_sft.jsonl")
ap.add_argument("--per-source", type=int, default=300)
ap.add_argument("--seed", type=int, default=20260610)
ap.add_argument("--fails-out", default="outputs/gold_selfcheck_fails.jsonl")
args = ap.parse_args()

rng = random.Random(args.seed)
# 蓄水池采样:每个源前缀留 per-source 条
buckets = defaultdict(list)
counts = defaultdict(int)
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

fails = []
print(f"{'source':<22}{'n':>6}{'pass':>8}{'no_pred':>9}{'mismatch':>10}")
total_n = total_ok = 0
for src in sorted(buckets):
    n = ok = nopred = 0
    for o in buckets[src]:
        gt = o["reward_model"]["ground_truth"]
        style = o["reward_model"]["style"]
        resp = o["gold_response"]
        good = verify_answer(resp, gt, style)
        n += 1
        if good:
            ok += 1
        else:
            pred = extract_pred(resp)
            if pred is None:
                nopred += 1
            fails.append({"source": o["data_source"], "style": style, "gt": gt,
                          "pred": pred, "tail": resp[-300:]})
    total_n += n; total_ok += ok
    flag = "" if ok / n >= 0.97 else "  <-- CHECK"
    print(f"{src:<22}{n:>6}{ok/n:>8.1%}{nopred:>9}{n-ok-nopred:>10}{flag}")
print(f"\nTOTAL {total_n} pass {total_ok/total_n:.2%}; fails dumped: {len(fails)}")
with open(args.fails_out, "w") as f:
    for x in fails:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
