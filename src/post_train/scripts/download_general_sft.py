#!/usr/bin/env python3
"""下载"通用 SFT"数据集到本地（用于阶段1初步 SFT + 阶段2 防语言退化的通用语料）。

为什么用 snapshot_download 而非 load_dataset：本机 ~/.bashrc 把 HF_ENDPOINT 指向了
hf-mirror.com 镜像，该镜像的 API 可用但**文件元数据残缺**，导致 datasets/hub 的文件下载
报 LocalEntryNotFoundError。本脚本强制走真 hub（huggingface.co）+ 本机代理，把原始文件
直接拉到本地；巨型数据集(Infinity-Instruct / FLAN)按 allow_patterns 采样下载若干分片
（够覆盖所有子类 + 抽样即可，全量是 TB 级）。

代理(127.0.0.1:7890)间歇返回 502，故：单数据集失败不中断整批；用 .download_complete 标记
跳过已完成的；--loop 模式反复重试未完成的，骑过代理坏窗口直到全部拿到。

环境（脚本已内置默认，可用同名环境变量覆盖）：
  HF_ENDPOINT=https://huggingface.co / http(s)_proxy=http://127.0.0.1:7890 / HF_HOME=/data/zilu/.hf-cache
原始文件落到: /data/zilu/general_sft_raw/<name>/

用法:
  python scripts/download_general_sft.py all --loop          # 夜间无人值守, 反复重试到全部完成
  python scripts/download_general_sft.py no_robots
  python scripts/download_general_sft.py --list
"""
from __future__ import annotations
import os

# —— 必须在导入 huggingface_hub 之前设好 ——
# 默认走 hf-mirror 镜像直连(无代理)：实测最稳最快，覆盖绝大多数文件。
# 少数 Xet 存储文件镜像取不到 → 兜底用真 hub+代理：调用方 export
#   HF_ENDPOINT=https://huggingface.co http(s)_proxy=http://127.0.0.1:7890
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/data/zilu/.hf-cache")

import argparse
import time
from pathlib import Path
from huggingface_hub import snapshot_download

RAW_ROOT = Path("/data/zilu/general_sft_raw")

# name -> (hf_repo, allow_patterns)；allow_patterns=None=全量，[...]=采样(TB 级巨型)
REGISTRY = {
    "no_robots":                   ("HuggingFaceH4/no_robots", None),                 # EN 10k, category 列 10 类
    "dolly-15k":                   ("databricks/databricks-dolly-15k", None),         # EN 15k, category 列 8 类
    "tulutalk-annotated":          ("aladinDJ/tulutalk-annotated", None),             # 多轮对话, 带标注列
    "coig-cqia":                   ("m-a-p/COIG-CQIA", None),                          # ZH, 13 子来源(目录)
    "dynamics-instruction-tuning": ("ChiyuSONG/dynamics-of-instruction-tuning", None),# ZH, curated/synthetic × 能力
    "infinity-instruct":           ("BAAI/Infinity-Instruct", None),                   # 全量(~21G, 7M+; 镜像403→需代理+真hub)
    # flan 全量 412G，只取 4 个小而有用的子类(CoT 推理 + 对话 + 指令多样性, zero-shot)
    "flan":                        ("Open-Orca/FLAN", ["cot_zsopt_data/*", "cot_fsopt_data/*",
                                                       "dialog_zsopt_data/*", "niv2_zsopt_data/*"]),  # ~3.7G
}


def _marker(name: str) -> Path:
    return RAW_ROOT / name / ".download_complete"


def download_one(name: str, force: bool = False, n_try: int = 3) -> bool:
    """返回 True=完成/已完成, False=本轮失败(可下轮重试)。"""
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(REGISTRY)}")
    repo, patterns = REGISTRY[name]
    local_dir = RAW_ROOT / name
    sampled = patterns is not None
    if _marker(name).exists() and not force:
        print(f"[SKIP] {name} 已完成", flush=True)
        return True
    print(f"[GET ] {name} <- {repo}  ({'采样:'+','.join(patterns) if sampled else '全量'})", flush=True)
    for attempt in range(1, n_try + 1):
        try:
            snapshot_download(  # 自带断点续传：已下文件跳过
                repo_id=repo, repo_type="dataset", local_dir=str(local_dir),
                allow_patterns=patterns, max_workers=4,
            )
            n = sum(1 for _ in local_dir.rglob("*") if _.is_file())
            _marker(name).write_text("ok\n")
            print(f"[DONE] {name} -> {local_dir}  ({n} files{'，采样' if sampled else ''})", flush=True)
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
    ap.add_argument("--loop", action="store_true", help="循环重试未完成的，直到全部完成")
    ap.add_argument("--loop-sleep", type=int, default=60)
    ap.add_argument("--max-minutes", type=int, default=600)
    args = ap.parse_args()
    if args.list:
        for k, (repo, pat) in REGISTRY.items():
            print(f"  {k:30s} {repo:42s} {'采样' if pat else '全量'}")
        return
    names = list(REGISTRY) if args.dataset == "all" else [args.dataset]

    if not args.loop:
        ok = all(download_one(n, force=args.force) for n in names)
        raise SystemExit(0 if ok else 1)

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
            print(f"[TIMEOUT] {args.max_minutes}min 到，仍缺: {[n for n in names if not _marker(n).exists()]}", flush=True)
            raise SystemExit(1)
        time.sleep(args.loop_sleep)
        elapsed += args.loop_sleep


if __name__ == "__main__":
    main()
