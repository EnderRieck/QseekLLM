"""终评(阶段末完整评测,training_plan v2 §5)：Pass@1/Pass@8 + 维度切分 + 完整 IO dump。

与 async_eval(过程监控)的分工：async_eval 窄而快(heldout 3881,每 500 step)；
本脚本宽而全(6 benchmark ~10.7k 题,阶段末跑一次)。判分复用 data_pipeline.reward(5 种 style)。

Benchmark 与维度：
  cc-reserved : Compute_Cot 保留集 test/id_test.jsonl(训练隔离有保证),每子源采 N(默认20) —— 数学基本功
  gsm8k       : test 1319                                  —— 英文应用题
  math500     : 500,按 level/subject 切分                   —— 竞赛分级
  gsmplus     : 2400,按 7 类 perturbation 切分              —— 扰动鲁棒性
  cmath       : test 1098,按 grade/reasoning_step/num_digits —— 中文小学数学
  svamp       : 300,按 Type 切分                            —— 简单应用题

用法:
  .venv/bin/python -m eval.final_eval --ckpt /data/zilu/fastrl/checkpoints/sft_foundation_v2/global_step_N --gpus 2,3
  # 冒烟: --benchmarks svamp --limit 8 --k 2
产出(默认 <ckpt>/final_eval/): {bench}.jsonl(全量 dump) + summary.json + summary.md
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.format import make_prompt
from data_pipeline.reward import compute_reward

CC_TEST = "/data/zilu/QseekLLM/src/post_train/Compute_Cot/data/clean/test/id_test.jsonl"
BENCH_ROOT = "/data/zilu/fastrl/data/benchmark"


# ---------------- benchmark 加载(统一为 {question, gold, style, meta}) ----------------

def _load_cc_reserved(per_source: int, seed: int):
    by_src = defaultdict(list)
    with open(CC_TEST, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            by_src[o["source"]].append(o)
    rng = random.Random(seed)
    rows = []
    for src in sorted(by_src):
        pool = by_src[src]
        for o in (rng.sample(pool, per_source) if len(pool) > per_source else pool):
            rows.append({"question": o["messages"][0]["content"], "gold": str(o["answer"]),
                         "style": "compute_cot", "meta": {"source": src}})
    return rows


def _load_arrow(name):
    from datasets import load_from_disk
    return load_from_disk(os.path.join(BENCH_ROOT, name))


def _load_gsm8k():
    ds = _load_arrow("gsm8k")["test"]
    return [{"question": r["question"], "gold": r["answer"].split("####")[-1].strip().replace(",", ""),
             "style": "gsm8k", "meta": {}} for r in ds]


def _load_math500():
    ds = _load_arrow("math-500")["test"]
    return [{"question": r["problem"], "gold": str(r["answer"]), "style": "math_verify",
             "meta": {"level": str(r["level"]), "subject": r["subject"]}} for r in ds]


def _load_gsmplus():
    ds = _load_arrow("gsm-plus")
    return [{"question": r["question"], "gold": str(r["answer"]), "style": "math_verify",
             "meta": {"perturbation": r["perturbation_type"]}} for r in ds]


def _load_cmath():
    ds = _load_arrow("cmath")["test"]
    return [{"question": r["question"], "gold": str(r["golden"]), "style": "math_verify",
             "meta": {"grade": str(r["grade"]), "steps": str(r["reasoning_step"]),
                      "digits": str(r["num_digits"])}} for r in ds]


def _load_svamp():
    ds = _load_arrow("svamp")
    return [{"question": r["question_concat"], "gold": str(r["Answer"]), "style": "math_verify",
             "meta": {"type": r["Type"]}} for r in ds]


BENCH_LOADERS = {
    "cc-reserved": None,  # 特殊:带参,在 main 里构造
    "gsm8k": _load_gsm8k,
    "math500": _load_math500,
    "gsmplus": _load_gsmplus,
    "cmath": _load_cmath,
    "svamp": _load_svamp,
}


# ---------------- 生成(vLLM 子进程,单卡一进程;先贪心 Pass@1 再采样 Pass@k) ----------------

def _vllm_worker(card, hf_dir, shard, shard_path, k, temperature, top_p,
                 max_new_tokens, chunk_size, gpu_mem_util):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(card)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir)
    llm = LLM(model=hf_dir, dtype="bfloat16", gpu_memory_utilization=gpu_mem_util,
              max_model_len=2048 + max_new_tokens, disable_log_stats=True)
    sp1 = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    spk = SamplingParams(n=k, temperature=temperature, top_p=top_p,
                         max_tokens=max_new_tokens) if k > 1 else None
    idxs = [x[0] for x in shard]
    rows = [x[1] for x in shard]
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
               for r in rows]
    with open(shard_path, "w", encoding="utf-8") as f:
        for c0 in range(0, len(prompts), chunk_size):
            chunk = prompts[c0:c0 + chunk_size]
            outs1 = llm.generate(chunk, sp1, use_tqdm=False)
            outsk = llm.generate(chunk, spk, use_tqdm=False) if spk else None
            for j, o in enumerate(outs1):
                rec = {"idx": idxs[c0 + j], "gen_greedy": o.outputs[0].text,
                       "gens": [x.text for x in outsk[j].outputs] if outsk else []}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[card{card}] {min(c0 + chunk_size, len(prompts))}/{len(prompts)}", flush=True)
    try:  # vLLM 正常析构常 hang(同 async_eval),杀子进程后强退
        import psutil
        for c in psutil.Process().children(recursive=True):
            try:
                c.kill()
            except Exception:
                pass
    except Exception:
        pass
    os._exit(0)


def generate_all(rows, hf_dir, gpus, k, temperature, top_p, max_new_tokens,
                 chunk_size, gpu_mem_util, tmp_dir):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    shards = {g: [] for g in gpus}
    for i, r in enumerate(rows):
        shards[gpus[i % len(gpus)]].append((i, r))
    procs, paths = [], []
    os.makedirs(tmp_dir, exist_ok=True)
    for g in gpus:
        p_out = os.path.join(tmp_dir, f"shard_{g}.jsonl")
        paths.append(p_out)
        p = ctx.Process(target=_vllm_worker, args=(g, hf_dir, shards[g], p_out, k, temperature,
                                                   top_p, max_new_tokens, chunk_size, gpu_mem_util))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    gens = {}
    for p_out in paths:
        with open(p_out, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                gens[o["idx"]] = o
    missing = len(rows) - len(gens)
    if missing:
        print(f"[warn] {missing} 条生成缺失(worker 异常?)", flush=True)
    return gens


# ---------------- 判分与汇总 ----------------

def score_bench(name, rows, gens, k):
    records, n1 = [], 0
    nk = 0
    fmt = 0
    by_meta = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # meta_key -> val -> [correct, total]
    for i, r in enumerate(rows):
        g = gens.get(i)
        if g is None:
            continue
        res1 = compute_reward(g["gen_greedy"], r["gold"], r["style"])
        ck = [compute_reward(t, r["gold"], r["style"])["correct"] for t in g["gens"]]
        p8 = bool(res1["correct"] or any(ck)) if ck else None  # pass@k 含贪心样本
        n1 += res1["correct"]
        nk += bool(p8)
        fmt += res1["has_format"]
        for mk, mv in r["meta"].items():
            c = by_meta[mk][mv]
            c[0] += res1["correct"]
            c[1] += 1
        records.append({"idx": i, "bench": name, "meta": r["meta"], "question": r["question"],
                        "gold": r["gold"], "style": r["style"], "gen_greedy": g["gen_greedy"],
                        "correct@1": res1["correct"], "has_format": res1["has_format"],
                        "gens": g["gens"], "gens_correct": ck, f"pass@{k}": p8})
    n = len(records)
    metrics = {"n": n, "pass@1": round(n1 / n, 4) if n else 0.0,
               f"pass@{k}": round(nk / n, 4) if (n and k > 1) else None,
               "format_rate": round(fmt / n, 4) if n else 0.0,
               "breakdown": {mk: {mv: {"acc": round(c / t, 4), "n": t}
                                  for mv, (c, t) in sorted(vals.items())}
                             for mk, vals in by_meta.items()}}
    return records, metrics


def write_summary_md(path, ckpt, all_metrics, k):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 终评 summary · `{ckpt}`\n\n")
        f.write(f"| benchmark | n | Pass@1 | Pass@{k} | format |\n|---|---|---|---|---|\n")
        for b, m in all_metrics.items():
            pk = f"{m[f'pass@{k}']:.1%}" if m.get(f"pass@{k}") is not None else "-"
            f.write(f"| {b} | {m['n']} | {m['pass@1']:.1%} | {pk} | {m['format_rate']:.1%} |\n")
        for b, m in all_metrics.items():
            for mk, vals in m["breakdown"].items():
                if b == "cc-reserved" and mk == "source":
                    weak = sorted(vals.items(), key=lambda kv: kv[1]["acc"])[:25]
                    f.write(f"\n## {b} · 最弱 25 个子源(Pass@1)\n\n| source | acc | n |\n|---|---|---|\n")
                    for mv, st in weak:
                        f.write(f"| {mv} | {st['acc']:.0%} | {st['n']} |\n")
                else:
                    f.write(f"\n## {b} · by {mk}(Pass@1)\n\n| {mk} | acc | n |\n|---|---|---|\n")
                    for mv, st in vals.items():
                        f.write(f"| {mv} | {st['acc']:.1%} | {st['n']} |\n")
        f.write("\n> 完整 IO dump 见同目录 `{bench}.jsonl`;判分=data_pipeline.reward(boxed/#### 抽取+sympy 等价)。\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="verl step 目录(global_step_N)或 HF 目录")
    ap.add_argument("--base", default="/data/zilu/fastrl/checkpoints/qseek_digitsplit_base")
    ap.add_argument("--out-dir", default="", help="默认 <ckpt>/final_eval")
    ap.add_argument("--gpus", default="2,3", help="PCI 序卡号,逗号分隔")
    ap.add_argument("--benchmarks", default="cc-reserved,gsm8k,math500,gsmplus,cmath,svamp")
    ap.add_argument("--k", type=int, default=8, help="Pass@k 的 k;1=只跑贪心")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--cc-per-source", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--limit", type=int, default=0, help="每 benchmark 限量(冒烟用)")
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]

    # checkpoint -> HF 目录(verl step 目录则转换,幂等复用 async_eval 的转换器)
    ckpt = args.ckpt.rstrip("/")
    if os.path.exists(os.path.join(ckpt, "config.json")):
        hf_dir = ckpt
    else:
        from eval.async_eval import convert_to_hf
        hf_dir = convert_to_hf(ckpt, args.base, ckpt + "_hf")
    out_dir = args.out_dir or os.path.join(ckpt, "final_eval")
    os.makedirs(out_dir, exist_ok=True)

    all_metrics = {}
    for bench in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        rows_raw = (_load_cc_reserved(args.cc_per_source, args.seed) if bench == "cc-reserved"
                    else BENCH_LOADERS[bench]())
        if args.limit:
            rows_raw = rows_raw[:args.limit]
        rows = [{**r, "prompt": make_prompt(r["question"])} for r in rows_raw]
        print(f"=== [{bench}] {len(rows)} 题, Pass@1{f'+Pass@{args.k}' if args.k > 1 else ''} ===", flush=True)
        gens = generate_all(rows, hf_dir, gpus, args.k, args.temperature, args.top_p,
                            args.max_new_tokens, args.chunk_size, args.gpu_mem_util,
                            tmp_dir=os.path.join(out_dir, f".tmp_{bench}"))
        records, metrics = score_bench(bench, rows, gens, args.k)
        with open(os.path.join(out_dir, f"{bench}.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_metrics[bench] = metrics
        pk = f"  pass@{args.k}={metrics[f'pass@{args.k}']}" if args.k > 1 else ""
        print(f"  [{bench}] n={metrics['n']} pass@1={metrics['pass@1']}{pk} fmt={metrics['format_rate']}", flush=True)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"ckpt": ckpt, "k": args.k, "metrics": all_metrics}, f, ensure_ascii=False, indent=2)
    write_summary_md(os.path.join(out_dir, "summary.md"), ckpt, all_metrics, args.k)
    print(f"完成 -> {out_dir}/{{summary.md,summary.json,<bench>.jsonl}}", flush=True)


if __name__ == "__main__":
    main()
