"""错题池 hook(组件①,对照实验 v3-wrongpool)。

在 trainer 的 _fit_compute_advantage 之后调用:按 prompt 组(uid)统计 16-rollout
正确数,把全错/蒙对组 (a) 梯度屏蔽(advantages 置零,不再驱动策略梯度);
(b) 题面+GT 追加到 WRONGPOOL_DIR/pending.jsonl 供教师worker消费。

设计见 docs/rl_wrongpool_sft_experiment_20260614.md。通过环境变量配置,trainer 仅
加 import + 一行调用,verl 改动最小。

ENV:
  WRONGPOOL_ENABLE=1            打开本 hook(默认关,保证不影响其它run)
  WRONGPOOL_DIR=...            池目录(默认 /data/zilu/fastrl/wrongpool_v3exp)
  WRONGPOOL_LUCKY_MAX=2        c<=该值且>0 视为"蒙对"
  WRONGPOOL_MASK_ALLWRONG=1    全错组是否也屏蔽梯度(默认1;norm_adv=False下其adv本就≈0)
  WRONGPOOL_CORRECT_THRESH=0.5 无 correct 字段时,用 token_level_scores 求和>=该值判对
"""
from __future__ import annotations
import json
import os
import time
from collections import defaultdict

import numpy as np


def _enabled() -> bool:
    return os.environ.get("WRONGPOOL_ENABLE", "0") in ("1", "true", "True")


def _cfg():
    return dict(
        pool_dir=os.environ.get("WRONGPOOL_DIR", "/data/zilu/fastrl/wrongpool_v3exp"),
        lucky_max=int(os.environ.get("WRONGPOOL_LUCKY_MAX", "2")),
        mask_allwrong=os.environ.get("WRONGPOOL_MASK_ALLWRONG", "1") in ("1", "true", "True"),
        correct_thresh=float(os.environ.get("WRONGPOOL_CORRECT_THRESH", "0.5")),
    )


def _per_sample_correct(batch, thresh: float) -> np.ndarray:
    """每样本 0/1 正确。优先用 reward 暴露的 'correct',否则用 token_level_scores 求和阈值。"""
    ntb = batch.non_tensor_batch
    if "correct" in ntb:
        return np.array([1 if float(x) >= 0.5 else 0 for x in ntb["correct"]], dtype=np.int64)
    # fallback: token_level_scores / rewards 求和
    for key in ("token_level_scores", "token_level_rewards"):
        if key in batch.batch.keys():
            s = batch.batch[key].sum(dim=-1).detach().float().cpu().numpy()
            return (s >= thresh).astype(np.int64)
    raise KeyError("wrongpool_hook: no 'correct' nor token_level_scores in batch")


def _messages_of(raw_prompt):
    """raw_prompt 可能是 list 或 np 数组,元素为 {role, content}。规整为 list[dict]。"""
    out = []
    for m in list(raw_prompt):
        if isinstance(m, dict):
            out.append({"role": m.get("role"), "content": m.get("content")})
    return out


def _ground_truth_of(rm):
    if isinstance(rm, dict):
        return rm.get("ground_truth")
    return None


def process(batch, pver: int = -1, metrics: dict | None = None):
    """主入口。返回简短统计 dict;同时就地修改 batch.batch['advantages'] 并写 pending.jsonl。"""
    if not _enabled():
        return {}
    cfg = _cfg()
    ntb = batch.non_tensor_batch
    if "uid" not in ntb:
        return {}

    correct = _per_sample_correct(batch, cfg["correct_thresh"])
    uid = ntb["uid"]
    groups: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(uid):
        groups[str(u)].append(i)

    masked_rows: list[int] = []
    pool_entries: list[dict] = []
    n_allwrong = n_lucky = 0
    for u, idxs in groups.items():
        c = int(sum(int(correct[i]) for i in idxs))
        n = len(idxs)
        if c == 0:
            label = "all_wrong"
            n_allwrong += 1
        elif c <= cfg["lucky_max"]:
            label = "lucky"
            n_lucky += 1
        else:
            continue
        # 梯度屏蔽:蒙对总是屏蔽;全错按开关(其 adv 在 norm_adv=False 下本就≈0)
        if label == "lucky" or (label == "all_wrong" and cfg["mask_allwrong"]):
            masked_rows.extend(idxs)
        # 入池(组内 prompt 相同,取代表样本)
        i0 = idxs[0]
        try:
            msgs = _messages_of(ntb["raw_prompt"][i0]) if "raw_prompt" in ntb else []
            gt = _ground_truth_of(ntb["reward_model"][i0]) if "reward_model" in ntb else None
            ds = str(ntb["data_source"][i0]) if "data_source" in ntb else None
        except Exception:
            continue
        pool_entries.append({
            "data_source": ds,
            "ground_truth": gt,
            "prompt": msgs,
            "label": label,
            "correct_count": c,
            "n": n,
            "pver": int(pver),
            "ts": time.time(),
        })

    # 应用梯度屏蔽
    if masked_rows and "advantages" in batch.batch.keys():
        adv = batch.batch["advantages"]
        adv[masked_rows] = 0.0
        batch.batch["advantages"] = adv

    # 写 pending.jsonl(driver 单写者,追加)
    if pool_entries:
        os.makedirs(cfg["pool_dir"], exist_ok=True)
        path = os.path.join(cfg["pool_dir"], "pending.jsonl")
        with open(path, "a") as f:
            for e in pool_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            f.flush()

    stats = {
        "wrongpool/n_groups": len(groups),
        "wrongpool/n_all_wrong": n_allwrong,
        "wrongpool/n_lucky": n_lucky,
        "wrongpool/n_masked_rows": len(masked_rows),
        "wrongpool/n_pooled": len(pool_entries),
    }
    if metrics is not None:
        metrics.update(stats)
    return stats
