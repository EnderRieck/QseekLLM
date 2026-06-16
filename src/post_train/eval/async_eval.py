"""异步过程评测 / dump（hybrid_sft_rl_design.md §4.1/§6/§7；CLAUDE.md 过程跟踪原则）。

监视 SFT/RL 的 checkpoint 目录，每出现新 global_step_N：
  1) 把 Verl FSDP checkpoint（fp32 state_dict `.pt`）转成 HF 格式 → 用 vLLM 加载；
  2) 在 held-out（训练外、无泄漏）上贪心生成（Pass@1，temperature=0，确定性可比）；
  3) 用 data_pipeline.reward 判分；
  4) 落盘 **完整输入输出 dump** `eval_dumps/step_N/heldout.jsonl`（每条含 prompt/generation/gold/correct）
     + 指标 `metrics.jsonl`（overall / per-source / per-difficulty 准确率、格式合规率、生成长度）
     + **自动可视化 `eval_dumps/step_N/heldout.md`**（每 source 抽样 N 对+N 错对照；无 gold 自由题全留；--md-per-class 调，0=关）。

特性：
  - **vLLM 后端**（默认）：paged-attention + continuous batching，比 HF generate 快 ~10x；--backend hf 兜底。
  - **多卡数据并行**：held-out 按长度均衡切片，每张空闲 A4000 一个独立实例并行生成（--max-gpus）。
  - **实时进度**：主进程每 --progress-every 秒打印各卡完成度（读 shard 行数）。
  - **增量落盘**：每块生成完立即 flush 写 shard，崩溃也保住已生成结果；全部完成后合并。
在 card2,3（A4000）异步跑，不阻塞 A800 训练。

用法:
  CUDA_DEVICE_ORDER=PCI_BUS_ID python -m eval.async_eval \
      --ckpt-dir <SFT保存目录> --gpu-candidates 2,3 --max-gpus 2 --watch
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import shutil
import time
from collections import defaultdict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.reward import compute_reward


def load_heldout(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _strip_prefix(sd):
    out = {}
    for k, v in sd.items():
        k2 = k.replace("_fsdp_wrapped_module.", "").replace("module.", "")
        out[k2] = v
    return out


def find_ckpt_pt(step_dir):
    # verl 档可能把 .pt 放在 step_dir/ 或 step_dir/actor/ 下,两处都找
    for pat in ("model_world_size_*_rank_0.pt", "actor/model_world_size_*_rank_0.pt"):
        pts = sorted(glob.glob(os.path.join(step_dir, pat)))
        if pts:
            return pts[0]
    return None


def convert_to_hf(step_dir, base, hf_out):
    """把 Verl FSDP 的 fp32 state_dict(.pt) 灌进 base 结构，存成 bf16 HF safetensors 目录（vLLM 可加载）。
    已存在则直接复用（幂等）。在 CPU 上做，不占 GPU。"""
    if os.path.exists(os.path.join(hf_out, "config.json")) and \
       glob.glob(os.path.join(hf_out, "*.safetensors")):
        return hf_out
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt_pt = find_ckpt_pt(step_dir)
    if not ckpt_pt:
        # 绝不静默退回纯 base(那会评出 base 级乱码而难以察觉);宁可显式失败
        raise FileNotFoundError(
            f"convert_to_hf: 在 {step_dir}(及其 actor/ 子目录)未找到 model_world_size_*_rank_0.pt;"
            f"若是 verl FSDP 档,请改用 `python -m verl.model_merger merge --backend fsdp "
            f"--local_dir {step_dir}/actor --target_dir <hf>` 预合并后再喂 final_eval。")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    if ckpt_pt:
        sd = torch.load(ckpt_pt, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
        sd = _strip_prefix(sd)
        sd = {k: (v.to(torch.bfloat16) if hasattr(v, "to") and v.is_floating_point() else v)
              for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  转换 {os.path.basename(ckpt_pt)}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    os.makedirs(hf_out, exist_ok=True)
    model.save_pretrained(hf_out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(hf_out)
    del model
    return hf_out


def _build_prompts(tok, rows):
    return [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in rows]


def _write_record(f, idx, r, g):
    res = compute_reward(g, r["ground_truth"], r["style"])
    f.write(json.dumps({"idx": idx, "source": r["source"], "difficulty": r["difficulty"],
                        "prompt": r["prompt"][-1]["content"], "generation": g,
                        "gold": r["ground_truth"], "correct": res["correct"],
                        "has_format": res["has_format"]}, ensure_ascii=False) + "\n")


def _md_block(r):
    """一条记录渲染成 md（✅对/❌错/🆓无gold）。"""
    mark = "🆓" if str(r.get("gold", "")) == "" else ("✅" if r["correct"] else "❌")
    fmt = "有格式" if r["has_format"] else "无格式"
    note = r.get("difficulty", "")
    note = f"  ·  {note}" if note and note != "-" else ""
    return (f"### {mark} `{r['source']}`  ·  gold=`{r.get('gold','')}`  ·  {fmt}  ·  len={len(r['generation'])}{note}\n\n"
            f"**问题**\n\n```text\n{r['prompt'].strip()}\n```\n\n"
            f"**模型输出**\n\n```text\n{r['generation'].strip()}\n```\n\n---\n")


def write_eval_md(records, out_path, metrics, per_class=3, freeform_cap=50):
    """评完自动出可读 md：各 source 抽样【N 对 + N 错】对照，无 gold 的自由题全留。"""
    by_src = defaultdict(list); freeform = []
    for r in records:
        (freeform if str(r.get("gold", "")) == "" else by_src[r["source"].split(":")[0]]).append(r)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 评测可视化 · step {metrics['step']}\n\n")
        f.write(f"- 总数 **{metrics['n']}**  ·  acc **{metrics['acc']:.1%}**  ·  "
                f"format_rate {metrics['format_rate']:.1%}  ·  avg_len {metrics['avg_gen_chars']}\n")
        f.write(f"- 展示规则：每 source 抽 ≤{per_class} 对 + ≤{per_class} 错；无 gold 自由题全留（≤{freeform_cap}）\n")
        f.write(f"- 完整原始 dump 见同目录 `heldout.jsonl`（{metrics['n']} 条全量）\n\n")
        f.write("## 各 source 准确率\n\n| source | acc |\n|---|---|\n")
        for s, a in sorted(metrics.get("acc_by_source", {}).items(), key=lambda kv: -kv[1]):
            f.write(f"| {s} | {a:.1%} |\n")
        f.write("\n---\n")
        if freeform:
            f.write(f"\n## 🆓 自由题（无 gold，全留，共 {len(freeform)} 条；按 note 排序，中英同题相邻）\n\n")
            for r in sorted(freeform, key=lambda r: str(r.get("difficulty", "")))[:freeform_cap]:
                f.write(_md_block(r) + "\n")
        for s in sorted(by_src, key=lambda k: -len(by_src[k])):
            sub = by_src[s]
            if s.startswith("probe"):  # 探针题不抽样，全量展示，按 note 排序（中英同类相邻）
                cap = len(sub)
                sub = sorted(sub, key=lambda r: str(r.get("difficulty", "")))
            else:
                cap = per_class
            correct = [r for r in sub if r["correct"]][:cap]
            wrong = [r for r in sub if not r["correct"]][:cap]
            c = sum(r["correct"] for r in sub)
            f.write(f"\n## {s}  ({c}/{len(sub)} 对，展示 {len(correct)}✅ + {len(wrong)}❌)\n\n")
            for r in correct + wrong:
                f.write(_md_block(r) + "\n")


def _vllm_worker(card, hf_dir, base, shard, shard_path, max_new_tokens, chunk_size, gpu_mem_util):
    """单卡 vLLM 子进程：cuda:{card} 上加载 hf_dir，分块 generate，每块完 flush 写盘。
    shard = [(global_idx, heldout_row), ...]。CUDA_VISIBLE_DEVICES 必须在 import vllm 前设。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(card)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir)
    llm = LLM(model=hf_dir, dtype="bfloat16", gpu_memory_utilization=gpu_mem_util,
              max_model_len=2048 + max_new_tokens, disable_log_stats=True, enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)  # 贪心 Pass@1
    idxs = [x[0] for x in shard]
    rows = [x[1] for x in shard]
    prompts = _build_prompts(tok, rows)
    with open(shard_path, "w", encoding="utf-8") as f:
        for c0 in range(0, len(prompts), chunk_size):
            chunk = prompts[c0:c0 + chunk_size]
            outs = llm.generate(chunk, sp, use_tqdm=False)
            for k, o in enumerate(outs):
                _write_record(f, idxs[c0 + k], rows[c0 + k], o.outputs[0].text)
            f.flush()  # 增量落盘
    # 结果已全部 flush 落盘。vLLM 引擎正常析构常 hang，故 os._exit 强退。
    # 但 vLLM V1 会 spawn EngineCore 子进程，os._exit 不清理它 → 会变孤儿常驻占 A4000，
    # 导致下个 step 的 pick_free_gpus 永远找不到空卡（死锁）。所以强退前先杀掉所有子进程。
    try:
        import psutil
        for c in psutil.Process().children(recursive=True):
            try:
                c.kill()
            except Exception:
                pass
    except Exception:
        pass
    os._exit(0)


def _hf_worker(card, hf_dir, base, shard, shard_path, max_new_tokens, batch_size, token_budget):
    """兜底：单卡 HF transformers.generate 子进程（动态 token 预算分批 + 增量落盘）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = f"cuda:{card}"
    model = None
    try:
        tok = AutoTokenizer.from_pretrained(hf_dir)
        model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.bfloat16).to(device).eval()
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token_id = 3
        idxs = [x[0] for x in shard]
        rows = [x[1] for x in shard]
        texts = _build_prompts(tok, rows)
        plen = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: plen[i])
        with open(shard_path, "w", encoding="utf-8") as f:
            i = 0
            while i < len(order):
                b = [order[i]]; i += 1
                while i < len(order) and len(b) < batch_size:
                    cand = b + [order[i]]
                    if sum(plen[j] for j in cand) + len(cand) * max_new_tokens > token_budget:
                        break
                    b.append(order[i]); i += 1
                enc = tok([texts[j] for j in b], return_tensors="pt", padding=True,
                          truncation=True, max_length=2048).to(device)
                enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                for k, j in enumerate(b):
                    g = tok.decode(gen[k][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    _write_record(f, idxs[j], rows[j], g)
                f.flush()
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


def eval_step(step, step_dir, base, heldout, out_dir, devices, args):
    """多卡并行评测一个 step：(vLLM)转换→各卡子进程并行生成(增量落盘)→实时进度→合并→指标。"""
    import torch.multiprocessing as mp
    step_out = os.path.join(out_dir, f"step_{step}")
    os.makedirs(step_out, exist_ok=True)
    n = len(heldout)

    # vLLM 需要 HF 格式权重：先转换一次（评完按需清理）
    hf_dir = base
    if args.backend == "vllm":
        hf_dir = os.path.join(step_out, "_vllm_hf")
        print(f"[step {step}] 转换 checkpoint → {hf_dir} …", flush=True)
        convert_to_hf(step_dir, base, hf_dir)

    # 按 prompt 文本长度排序后轮询分配 → 各卡总工作量均衡
    order = sorted(range(n), key=lambda i: len(str(heldout[i]["prompt"])))
    shards = [[] for _ in devices]
    for rank, i in enumerate(order):
        shards[rank % len(devices)].append((i, heldout[i]))

    ctx = mp.get_context("spawn")
    procs = []
    shard_paths = []
    for d, sh in zip(devices, shards):
        sp = os.path.join(step_out, f"shard_card{d}.jsonl")
        shard_paths.append((d, sp, len(sh)))
        if args.backend == "vllm":
            wargs = (d, hf_dir, base, sh, sp, args.max_new_tokens, args.chunk_size, args.gpu_mem_util)
            target = _vllm_worker
        else:
            wargs = (d, hf_dir, base, sh, sp, args.max_new_tokens, args.batch_size, args.token_budget)
            target = _hf_worker
        p = ctx.Process(target=target, args=wargs)
        p.start(); procs.append(p)
    print(f"[step {step}] 启动 {len(devices)} 卡并行({args.backend})：" +
          " ".join(f"card{d}={ln}题" for d, _, ln in shard_paths), flush=True)

    # 实时进度：按 shard 行数判完成（不依赖进程退出——vLLM 引擎析构可能 hang）
    t0 = time.time()
    while True:
        time.sleep(args.progress_every)
        counts = []
        for d, sp, tot in shard_paths:
            c = sum(1 for _ in open(sp, encoding="utf-8")) if os.path.exists(sp) else 0
            counts.append((d, c, tot))
        done_n = sum(c for _, c, _ in counts)
        detail = " ".join(f"c{d}:{c}/{tot}" for d, c, tot in counts)
        el = time.time() - t0
        rate = done_n / el if el > 0 else 0
        eta = (n - done_n) / rate if rate > 0 else float("inf")
        print(f"[step {step}] 进度 {done_n}/{n} ({100*done_n/n:.0f}%) | {detail} | "
              f"{el:.0f}s 已用, ETA {eta:.0f}s", flush=True)
        if done_n >= n:  # 全部 shard 写满 → 生成完成，不等可能 hang 的引擎析构
            break
        if not any(p.is_alive() for p in procs):  # 进程都退了但没写满 → 有 worker 崩了
            print(f"[step {step}] 子进程已全部退出但仅 {done_n}/{n}，进入合并/重试判定", flush=True)
            break
    # 收尾：给宽限期自然退出，否则强制终止残留(已落盘，安全)。
    # 同时杀掉 worker 的 vLLM EngineCore 子进程，避免孤儿常驻占卡。
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            try:
                import psutil
                for c in psutil.Process(p.pid).children(recursive=True):
                    try:
                        c.kill()
                    except Exception:
                        pass
            except Exception:
                pass
            p.terminate()

    # 合并各卡 shard → 规范 dump（含 prompt/generation/gold/correct），按原始 idx 排序
    records = []
    for _, sp, _ in shard_paths:
        if os.path.exists(sp):
            for line in open(sp, encoding="utf-8"):
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    records.sort(key=lambda r: r["idx"])
    dump_path = os.path.join(step_out, "heldout.jsonl")
    with open(dump_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for _, sp, _ in shard_paths:
        try:
            os.remove(sp)
        except Exception:
            pass
    if args.backend == "vllm" and not args.keep_hf:
        shutil.rmtree(hf_dir, ignore_errors=True)  # 清理转换出的临时权重(~3.4G)

    got = len(records)
    if got < n:
        raise RuntimeError(f"step {step} 仅完成 {got}/{n}（dump 已保存到 {dump_path}），将重试")

    # 仅对【有 gold】的题算 acc/format（无 gold 的自由对话题只看输出、不进分母）
    scored = [r for r in records if str(r.get("gold", "")) != ""]
    ns = max(len(scored), 1)
    by_src = defaultdict(lambda: [0, 0]); by_diff = defaultdict(lambda: [0, 0])
    n_correct = sum(r["correct"] for r in scored)
    n_fmt = sum(r["has_format"] for r in scored)
    tot_len = sum(len(r["generation"]) for r in records)  # 生成长度统计含全部
    for r in scored:
        by_src[r["source"].split(":")[0]][0] += r["correct"]
        by_src[r["source"].split(":")[0]][1] += 1
        by_diff[r["difficulty"]][0] += r["correct"]; by_diff[r["difficulty"]][1] += 1

    metrics = {"step": step, "n": got, "n_scored": len(scored), "acc": round(n_correct / ns, 4),
               "format_rate": round(n_fmt / ns, 4), "avg_gen_chars": round(tot_len / got, 1),
               "acc_by_source": {k: round(v[0] / v[1], 3) for k, v in sorted(by_src.items())},
               "acc_by_difficulty": {k: round(v[0] / v[1], 3) for k, v in sorted(by_diff.items())}}
    with open(os.path.join(out_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    md_path = os.path.join(step_out, "heldout.md") if args.md_per_class > 0 else None
    if md_path:
        try:
            write_eval_md(records, md_path, metrics, per_class=args.md_per_class)
        except Exception as e:
            print(f"  md warn: {str(e)[:100]}", flush=True)
            md_path = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        sw = SummaryWriter(os.path.join(out_dir, "tb"))
        sw.add_scalar("eval/acc", metrics["acc"], step)
        sw.add_scalar("eval/format_rate", metrics["format_rate"], step)
        sw.add_scalar("eval/avg_gen_chars", metrics["avg_gen_chars"], step)
        for k, v in metrics["acc_by_difficulty"].items():
            sw.add_scalar(f"eval_diff/{k}", v, step)
        sw.close()
    except Exception:
        pass
    print(f"[step {step}] 完成 acc={metrics['acc']} fmt={metrics['format_rate']} "
          f"len={metrics['avg_gen_chars']} -> {dump_path}"
          + (f"\n           可视化 md -> {md_path}" if md_path else ""), flush=True)
    return metrics


def pick_free_gpus(candidates, min_free_gb=8.0, max_n=2, wait_poll=30):
    """候选卡里挑最空闲(free≥阈值)的至多 max_n 张，返回 index 列表（需 CUDA_DEVICE_ORDER=PCI_BUS_ID）。"""
    import subprocess
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                encoding="utf-8")
            free = {int(l.split(",")[0]): int(l.split(",")[1]) / 1024 for l in out.strip().splitlines()}
        except Exception:
            free = {}
        avail = sorted([(c, free.get(c, 0.0)) for c in candidates if free.get(c, 0.0) >= min_free_gb],
                       key=lambda x: -x[1])
        if avail:
            chosen = [c for c, _ in avail[:max_n]]
            print("  选中 " + ", ".join(f"cuda:{c}({free[c]:.1f}G)" for c in chosen), flush=True)
            return chosen
        print(f"  候选卡 {candidates} 都不空闲(需≥{min_free_gb}G)，{wait_poll}s 后重试…", flush=True)
        time.sleep(wait_poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="训练保存目录（含 global_step_N）")
    ap.add_argument("--base", default="/data/zilu/fastrl/checkpoints/qseek_digitsplit_base")
    ap.add_argument("--heldout", default="eval/heldout.jsonl")
    ap.add_argument("--out-dir", default="", help="dump 输出，默认 <ckpt-dir>/eval_dumps")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm", help="生成后端，默认 vllm(快~10x)")
    ap.add_argument("--gpu-candidates", default="2,3", help="自动选卡候选(逗号分隔, PCI序)")
    ap.add_argument("--max-gpus", type=int, default=2, help="最多并行用几张卡")
    ap.add_argument("--min-free-gb", type=float, default=8.0, help="候选卡至少空闲多少G才用")
    ap.add_argument("--devices", default="", help="强制指定卡(如 2,3)；留空=自动选卡")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85, help="[vllm] 单卡显存占用比例")
    ap.add_argument("--chunk-size", type=int, default=256, help="[vllm] 每块生成多少条(增量落盘粒度)")
    ap.add_argument("--keep-hf", action="store_true", help="[vllm] 保留转换出的临时 HF 权重(默认评完删)")
    ap.add_argument("--batch-size", type=int, default=32, help="[hf] 单 batch 条数硬上限")
    ap.add_argument("--token-budget", type=int, default=12000, help="[hf] 每 batch token 预算(KV 显存∝此值)")
    ap.add_argument("--md-per-class", type=int, default=3, help="可视化 md 每 source 抽几条对/错(0=不出 md)")
    ap.add_argument("--progress-every", type=int, default=15, help="进度打印间隔秒")
    ap.add_argument("--watch", action="store_true", help="持续监视新 checkpoint")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--step", type=int, default=-1, help="只评指定 step（不监视）")
    args = ap.parse_args()
    cands = [int(x) for x in args.gpu_candidates.split(",") if x.strip()]

    def choose_devices():
        if args.devices:
            return [int(x) for x in args.devices.split(",") if x.strip()]
        return pick_free_gpus(cands, args.min_free_gb, args.max_gpus)

    out_dir = args.out_dir or os.path.join(args.ckpt_dir, "eval_dumps")
    os.makedirs(out_dir, exist_ok=True)
    heldout = load_heldout(args.heldout)
    print(f"held-out {len(heldout)} 条；监视 {args.ckpt_dir}（{args.backend}, 最多 {args.max_gpus} 卡并行）", flush=True)

    def steps_available():
        s = {}
        for d in glob.glob(os.path.join(args.ckpt_dir, "global_step_*")):
            m = re.search(r"global_step_(\d+)", d)
            if m and find_ckpt_pt(d):
                s[int(m.group(1))] = d
        return s

    done = set()
    if os.path.exists(os.path.join(out_dir, "metrics.jsonl")):
        for line in open(os.path.join(out_dir, "metrics.jsonl"), encoding="utf-8"):
            try:
                done.add(json.loads(line)["step"])
            except Exception:
                pass

    if args.step >= 0:
        s = steps_available()
        if args.step in s:
            eval_step(args.step, s[args.step], args.base, heldout, out_dir, choose_devices(), args)
        else:
            print(f"step {args.step} 不存在/无权重")
        return

    while True:
        for step, d in sorted(steps_available().items()):
            if step in done:
                continue
            try:
                eval_step(step, d, args.base, heldout, out_dir, choose_devices(), args)
                done.add(step)  # 仅成功才标记，失败下轮重试
            except Exception:
                import traceback
                traceback.print_exc()
                print(f"[eval step {step}] 失败，下一轮 poll 重试", flush=True)
        if not args.watch:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
