"""测各 checkpoint 的策略熵(SFT 阶段补测,GRPO 前的探索性诊断)。

方法:复用 eval_dumps/step_*/heldout.jsonl 里各 ckpt 自己的贪心生成,
teacher-forcing 一遍前向,对"生成段"每个位置算 softmax 熵(nats)。
与生成时的逐步分布完全一致(贪心、同一权重),无需重新 rollout。

用法:
  python -m eval.entropy_probe --device 5 --stride 8 \
    --out /data/zilu/fastrl/checkpoints/sft_s3r1/entropy_probe.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

BASE_HF = "/data/zilu/fastrl/checkpoints/sft_foundation_v2/global_step_9858_hf"
S3R1 = "/data/zilu/fastrl/checkpoints/sft_s3r1"
F2_DUMP = "/data/zilu/fastrl/checkpoints/sft_foundation_v2/eval_dumps/step_9858/heldout.jsonl"
HELDOUT = os.path.join(os.path.dirname(__file__), "heldout.jsonl")


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def mean_entropy(hf_dir, dump_path, heldout, stride, device, max_len=2688):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.bfloat16).to(device).eval()
    rows = {r["idx"]: r for r in load_jsonl(dump_path)}
    ent_sum, ent_n, per_sample = 0.0, 0, []
    with torch.inference_mode():
        for idx in range(0, len(heldout), stride):
            r = rows.get(idx)
            if r is None or not r["generation"]:
                continue
            prefix = tok.apply_chat_template(heldout[idx]["prompt"], tokenize=False,
                                             add_generation_prompt=True)
            pre_ids = tok(prefix, add_special_tokens=False).input_ids
            gen_ids = tok(r["generation"], add_special_tokens=False).input_ids
            ids = (pre_ids + gen_ids)[:max_len]
            n_gen = len(ids) - len(pre_ids)
            if n_gen <= 0:
                continue
            x = torch.tensor([ids], device=device)
            logits = model(x).logits[0, len(pre_ids) - 1:-1].float()  # 预测生成段各 token 的分布
            logp = torch.log_softmax(logits, dim=-1)
            ent = -(logp.exp() * logp).sum(-1)  # nats
            ent_sum += ent.sum().item()
            ent_n += n_gen
            per_sample.append(ent.mean().item())
    del model
    torch.cuda.empty_cache()
    per_sample.sort()
    return {"mean_token_entropy": ent_sum / max(ent_n, 1),
            "median_sample_entropy": per_sample[len(per_sample) // 2] if per_sample else None,
            "n_samples": len(per_sample), "n_tokens": ent_n}


def ensure_hf(step_dir, hf_out):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eval.async_eval import convert_to_hf
    return convert_to_hf(step_dir, BASE_HF, hf_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default=f"{S3R1}/entropy_probe.json")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)
    device = "cuda:0"

    heldout = load_jsonl(HELDOUT)
    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))

    jobs = [("f2_step_9858", BASE_HF, F2_DUMP, None)]
    for d in sorted(glob.glob(f"{S3R1}/global_step_*"), key=lambda p: int(p.rsplit("_", 1)[-1])):
        step = int(d.rsplit("_", 1)[-1])
        dump = f"{S3R1}/eval_dumps/step_{step}/heldout.jsonl"
        if os.path.exists(dump):
            jobs.append((f"s3r1_step_{step}", f"{S3R1}/entropy_tmp/step_{step}", dump, d))

    for name, hf_dir, dump, step_dir in jobs:
        if name in results:
            print(f"[skip] {name}", flush=True)
            continue
        if step_dir is not None:
            print(f"[convert] {name}", flush=True)
            ensure_hf(step_dir, hf_dir)
        print(f"[probe] {name}", flush=True)
        results[name] = mean_entropy(hf_dir, dump, heldout, args.stride, device)
        print(f"  {name}: H={results[name]['mean_token_entropy']:.4f} nats "
              f"(n={results[name]['n_samples']})", flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
