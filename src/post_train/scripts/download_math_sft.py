#!/usr/bin/env python3
"""下载"数学 SFT/评测"数据集到本地（阶段2/3 数学课程 + 评测）。

与 download_general_sft.py 同一套健壮机制（强制真 hub + 代理 + .download_complete 标记
+ --loop 骑过代理坏窗口）。巨型集按 allow_patterns 采样分片。

注意：orca-math-200k / openr1-math-220k / calc-ape210k **已在 /data/zilu/fastrl/data/train/**，
本脚本不重复下载，分析时直接用 fastrl 副本（见对应数据卡片）。

落地: /data/zilu/math_sft_raw/<name>/
用法: python scripts/download_math_sft.py all --loop
"""
from __future__ import annotations
import os
# 默认 hf-mirror 镜像直连(无代理, 最稳)；Xet 文件兜底用真 hub+代理(调用方 export)。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/data/zilu/.hf-cache")

import argparse
import time
from pathlib import Path
from huggingface_hub import snapshot_download

RAW_ROOT = Path("/data/zilu/math_sft_raw")

# name -> (repo, allow_patterns)  None=全量, [...]=采样
REGISTRY = {
    "metamathqa":        ("meta-math/MetaMathQA", None),                                   # 单 json 395k
    "numinamath-1.5":    ("AI-MO/NuminaMath-1.5", None),                                     # 全量(~0.5G, 896k)
    "openthoughts3-1.2m":("open-thoughts/OpenThoughts3-1.2M", None),                         # 全量(~28G, 1.2M)
    "bespoke-stratos-17k":("bespokelabs/Bespoke-Stratos-17k", None),                        # 17k 全量
    "calc-gsm8k":        ("MU-NLPC/Calc-gsm8k", None),                                       # 全量(小)
    "gsm8k":             ("openai/gsm8k", None),                                             # main+socratic 全量(小, 评测)
    "cmath":             ("weitianwen/cmath", None),                                         # 全量(小, 仅评测)
    # —— RL 阶段(阶段4 GRPO)数据池：强调可验证答案 ——
    "big-math-rl-verified": ("SynthLabsAI/Big-Math-RL-Verified", None),                      # 25万+, RL 主池
    "dapo-math-17k-dedup":  ("YouJiacheng/DAPO-Math-17k-dedup", None),                       # 17k 去重, GRPO/DAPO 起步
    "deepscaler-preview":   ("agentica-org/DeepScaleR-Preview-Dataset", None),               # 40k, hard math RL
    "math-beyond":          ("brendel-group/MATH-Beyond", None),                             # 小, hard eval(不训练)
}


def _marker(name: str) -> Path:
    return RAW_ROOT / name / ".download_complete"


def download_one(name: str, force: bool = False, n_try: int = 3) -> bool:
    if name not in REGISTRY:
        raise ValueError(f"Unknown: {name}. Available: {list(REGISTRY)}")
    repo, patterns = REGISTRY[name]
    local_dir = RAW_ROOT / name
    if _marker(name).exists() and not force:
        print(f"[SKIP] {name} 已完成", flush=True)
        return True
    print(f"[GET ] {name} <- {repo}  ({'采样:'+','.join(patterns) if patterns else '全量'})", flush=True)
    for attempt in range(1, n_try + 1):
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(local_dir),
                              allow_patterns=patterns, max_workers=4)
            n = sum(1 for _ in local_dir.rglob("*") if _.is_file())
            _marker(name).write_text("ok\n")
            print(f"[DONE] {name} -> {local_dir}  ({n} files{'，采样' if patterns else ''})", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            if attempt < n_try:
                wait = 8 * attempt
                print(f"  [retry {attempt}/{n_try}] {type(e).__name__}: {str(e)[:70]} — {wait}s", flush=True)
                time.sleep(wait)
            else:
                print(f"  [FAIL] {name}: {type(e).__name__}: {str(e)[:70]} (留待下轮)", flush=True)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="all", choices=list(REGISTRY) + ["all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--loop-sleep", type=int, default=60)
    ap.add_argument("--max-minutes", type=int, default=600)
    args = ap.parse_args()
    if args.list:
        for k, (repo, pat) in REGISTRY.items():
            print(f"  {k:22s} {repo:38s} {'采样' if pat else '全量'}")
        return
    names = list(REGISTRY) if args.dataset == "all" else [args.dataset]
    if not args.loop:
        raise SystemExit(0 if all(download_one(n, force=args.force) for n in names) else 1)
    elapsed, rnd = 0, 0
    while True:
        rnd += 1
        pending = [n for n in names if not _marker(n).exists()]
        if not pending:
            print(f"[ALL DONE] 全部 {len(names)} 个完成 (round {rnd})", flush=True)
            return
        print(f"=== round {rnd}: 待下 {len(pending)}: {pending} ===", flush=True)
        for n in pending:
            download_one(n, force=False)
        if elapsed >= args.max_minutes * 60:
            print(f"[TIMEOUT] 仍缺: {[n for n in names if not _marker(n).exists()]}", flush=True)
            raise SystemExit(1)
        time.sleep(args.loop_sleep)
        elapsed += args.loop_sleep


if __name__ == "__main__":
    main()
