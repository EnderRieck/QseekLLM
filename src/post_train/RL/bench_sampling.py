"""单卡采样吞吐基准:复刻 GRPO rollout 的工作量(64 prompt × n=8 = 512 条, max 1024 tok),
测某张卡的纯 vLLM decode 墙钟 + tok/s。用来标定 A800 vs A4000 单卡倍率,
为多卡/异步 GRPO 的提速估算提供实测地基。

用法: CUDA_VISIBLE_DEVICES=<card> python RL/bench_sampling.py [n_prompt=64] [n=8] [maxtok=1024] [memutil=0.5]
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyarrow.parquet as pq
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 64
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
MEM = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
CKPT = "/data/zilu/fastrl/checkpoints/sft_s3r1/global_step_3874_hf"

t = pq.read_table("/data/zilu/data_unified_v2/rl_smoke/train.parquet").to_pylist()[:NP]
tok = AutoTokenizer.from_pretrained(CKPT)
prompts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in t]

t0 = time.time()
llm = LLM(model=CKPT, dtype="bfloat16", gpu_memory_utilization=MEM, max_model_len=2048, enforce_eager=False)
t_load = time.time() - t0

sp = SamplingParams(n=N, temperature=1.0, top_p=1.0, max_tokens=MAXTOK)
t1 = time.time()
outs = llm.generate(prompts, sp)
t_gen = time.time() - t1

out_tok = sum(len(o.token_ids) for out in outs for o in out.outputs)
nseq = sum(len(out.outputs) for out in outs)
card = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(json.dumps({
    "card": card, "n_prompt": NP, "n": N, "n_seq": nseq, "max_tok": MAXTOK, "mem_util": MEM,
    "load_s": round(t_load, 1), "gen_s": round(t_gen, 1),
    "out_tokens": out_tok, "tok_per_s": round(out_tok / t_gen, 1),
    "avg_resp_len": round(out_tok / nseq, 1),
}, ensure_ascii=False))
