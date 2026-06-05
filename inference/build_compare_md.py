#!/usr/bin/env python
"""Merge per-stage infer outputs into one side-by-side comparison markdown.

  python tools/build_compare_md.py <out_dir> <prompts.jsonl> > compare.md

Reads <out_dir>/out_<stage>.jsonl (one completion per prompt, aligned by line order)
for each stage and emits, per prompt, every stage's continuation so the style/quality
shift across 50B -> CPT -> 8k -> 16k is easy to eyeball.
"""
import json, sys, glob, os, datetime

out_dir = sys.argv[1]
prompts_path = sys.argv[2]

prompts = [json.loads(l)["prompt"] for l in open(prompts_path) if l.strip()]

# stage files: out_<stage>.jsonl, sorted by the numeric prefix in <stage>
stage_files = sorted(glob.glob(os.path.join(out_dir, "out_*.jsonl")))
stages = []  # (label, [completions])
for f in stage_files:
    label = os.path.basename(f)[len("out_"):-len(".jsonl")]
    comps = []
    for l in open(f):
        if not l.strip():
            continue
        d = json.loads(l)
        comps.append(d.get("completion", d.get("text", "")))
    stages.append((label, comps))

print(f"# 阶段输出对比 (50B base → CPT → 8k → 16k)")
print(f"\n_生成于 {datetime.datetime.now():%Y-%m-%d %H:%M} · greedy 解码 · max_new_tokens 见脚本_\n")
print("模型: qwen_like 1.7B。对比四个 checkpoint 在相同 prompt 上的续写,看 CPT(风格优化)"
      "和长文本阶段(8k/16k)对输出的影响。\n")
for i, prompt in enumerate(prompts):
    print(f"\n---\n\n## Case {i+1}\n")
    print(f"**Prompt:** `{prompt}`\n")
    for label, comps in stages:
        cont = comps[i] if i < len(comps) else "(无输出)"
        # 续写部分(completion 通常不含 prompt;若含则原样展示)
        print(f"### {label}\n")
        print(f"```\n{cont.strip()}\n```\n")
