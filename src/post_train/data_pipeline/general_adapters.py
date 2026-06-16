"""通用 SFT 数据源适配（③：防语言退化 + 指令跟随，中英）。

与数学源不同：ability="general"、use="sft"、无 `<think>/boxed`、gold_response=回复。
多轮对话保留完整 prompt 历史，末条 assistant 作 gold。
"""
from __future__ import annotations
import glob
import json
import os
from typing import Iterator, Optional

from .format import Record, detect_lang

GEN_RAW = "/data/zilu/general_sft_raw"
FASTRL = "/data/zilu/fastrl/data/train"

SYS_GEN_EN = "You are a helpful, harmless and honest assistant."
SYS_GEN_ZH = "你是一个有帮助、诚实、无害的助手。"


def _sys_for(text: str) -> str:
    return SYS_GEN_ZH if detect_lang(text) == "zh" else SYS_GEN_EN


def _chat_record(messages: list, source: str, data_source: str) -> Optional[Record]:
    """messages=[{role,content}...]，末条须 assistant。前文作 prompt，末条作 gold。"""
    msgs = [m for m in messages if m.get("content")]
    if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
        return None
    gold = msgs[-1]["content"]
    hist = msgs[:-1]
    first_user = next((m["content"] for m in hist if m["role"] == "user"), hist[0]["content"])
    # 无 system 则补一个
    if hist[0].get("role") != "system":
        hist = [{"role": "system", "content": _sys_for(first_user)}] + hist
    return Record(prompt=hist, data_source=data_source, ability="general", use="sft",
                  reward_style="none", ground_truth="", gold_response=gold,
                  difficulty="", source=source, question=first_user)


def _io_record(instruction: str, ctx: str, output: str, source: str, data_source: str) -> Optional[Record]:
    if not instruction or not output:
        return None
    user = (instruction + ("\n\n" + ctx if ctx else "")).strip()
    return Record(prompt=[{"role": "system", "content": _sys_for(user)},
                          {"role": "user", "content": user}],
                  data_source=data_source, ability="general", use="sft",
                  reward_style="none", ground_truth="", gold_response=output.strip(),
                  difficulty="", source=source, question=user)


# ---- loaders ----
def _iter_arrow(path):
    from datasets import load_from_disk
    # 若 path 本身不是 dataset，而是含多个子数据集的目录（如 coig-cqia 的 13 子集），逐个加载
    if not (os.path.exists(os.path.join(path, "dataset_dict.json"))
            or os.path.exists(os.path.join(path, "dataset_info.json"))):
        for sub in sorted(os.listdir(path)):
            subp = os.path.join(path, sub)
            if os.path.isdir(subp):
                yield from _iter_arrow(subp)
        return
    ds = load_from_disk(path)
    for sp in (ds.values() if hasattr(ds, "values") else [ds]):
        yield from sp


def _iter_parquet(globpat):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(globpat, recursive=True)):
        for b in pq.read_table(f).to_batches(max_chunksize=2048):
            yield from b.to_pylist()


def _iter_jsonl(globpat):
    for f in sorted(glob.glob(globpat, recursive=True)):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


# ---- adapters ----
def adapt_no_robots(r):
    return _chat_record(r.get("messages", []), "no_robots", "no_robots")


def adapt_dolly(r):
    return _io_record(r.get("instruction", ""), r.get("context", ""), str(r.get("response", "")),
                      "dolly", "dolly-15k")


def adapt_coig(r):
    return _io_record(r.get("instruction", ""), str(r.get("input", "") or ""), str(r.get("output", "")),
                      f"coig:{(r.get('task_type') or {}).get('major', [''])[0] if isinstance(r.get('task_type'), dict) else ''}",
                      "coig-cqia")


def adapt_dynamics(r):
    # 2026-06-10 审计:type=reasoning(1.27万)是"选择题→裸字母答案"无推理过程,
    # 教模型不思考作答(MCQ 空区病征来源),剔除
    if r.get("type") == "reasoning":
        return None
    return _chat_record(r.get("messages", []), f"dynamics:{r.get('type', '')}", "dynamics-instruction-tuning")


def adapt_tulutalk(r):
    # 2026-06-10 审计:启用自带标注过滤(此前纯随机,浪费 80 万行标注)
    # ① llama_guard unsafe(1.9%)剔除;② 单轮 st_instruct_reward<0.75(最低 ~10%)剔除(NaN=多轮,放行)
    lg = r.get("llama_guard_2")
    if lg and str(lg).strip().lower() != "safe":
        return None
    sr = r.get("st_instruct_reward")
    try:
        if sr is not None and sr == sr and float(sr) < 0.75:   # sr==sr 排除 NaN
            return None
    except (ValueError, TypeError):
        pass
    msgs = r.get("messages")
    if isinstance(msgs, list):
        return _chat_record([{"role": m.get("role"), "content": m.get("content")} for m in msgs],
                            f"tulu:{r.get('task_category', '')}", "tulutalk-annotated")
    return None


def adapt_infinity(r):
    conv = r.get("conversations") or []
    # 含 \boxed 的数学对话归数学池(adapters.adapt_infinity_math)，这里只留非数学的当通用
    if conv and "\\boxed" in (conv[-1].get("value") or ""):
        return None
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    msgs = [{"role": role_map.get(m.get("from"), m.get("from")), "content": m.get("value")} for m in conv]
    return _chat_record(msgs, "infinity", "infinity-instruct")


def adapt_chinese_r1(r):
    """chinese-deepseek-r1-distill：通用**中文** R1（补中文 + 推理曝光，防退化）。score 过滤。"""
    q, content = r.get("input", ""), str(r.get("content", "") or "")
    if not q or not content:
        return None
    try:
        if r.get("score") is not None and float(r["score"]) < 7:   # 仅留较高分
            return None
    except (ValueError, TypeError):
        pass
    return _io_record(q, "", content, "chinese-r1", "chinese-deepseek-r1-distill")


GENERAL_SOURCES = {
    "no_robots":   (lambda: _iter_parquet(f"{GEN_RAW}/no_robots/**/*.parquet"), adapt_no_robots),
    "dolly":       (lambda: _iter_jsonl(f"{GEN_RAW}/dolly-15k/**/*.jsonl"), adapt_dolly),
    "coig-cqia":   (lambda: _iter_arrow(f"{FASTRL}/coig-cqia"), adapt_coig),  # 复用 fastrl(13 子集已合)
    "dynamics":    (lambda: _iter_jsonl(f"{GEN_RAW}/dynamics-instruction-tuning/**/full/*.json"), adapt_dynamics),
    "tulutalk":    (lambda: _iter_parquet(f"{GEN_RAW}/tulutalk-annotated/**/*.parquet"), adapt_tulutalk),
    "infinity":    (lambda: _iter_parquet(f"{GEN_RAW}/infinity-instruct/**/*.parquet"), adapt_infinity),
    "chinese-r1":  (lambda: _iter_arrow(f"{FASTRL}/chinese-deepseek-r1-distill"), adapt_chinese_r1),
}


def iter_general(name: str) -> Iterator[Record]:
    loader, adapt = GENERAL_SOURCES[name]
    for raw in loader():
        rec = adapt(raw)
        if rec is not None:
            yield rec
