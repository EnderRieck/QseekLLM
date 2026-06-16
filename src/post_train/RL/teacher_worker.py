#!/usr/bin/env python3
"""错题池教师worker(组件②,对照实验 v3-wrongpool)。

消费 WRONGPOOL_DIR/pending.jsonl(RL trainer 追加的全错/蒙对题),调用
GPT-5.3-Codex-Spark(`codex exec`)生成 <think>+\\boxed 解法,直接写入
WRONGPOOL_DIR/sft_ready.jsonl(SFT messages 格式)。**不过滤**(按用户决定:
教师输出即便错也直接用)。独立进程、不占 GPU、与 RL 解耦。

设计见 docs/rl_wrongpool_sft_experiment_20260614.md。

用法:
  python3 RL/teacher_worker.py --pool-dir /data/zilu/fastrl/wrongpool_v3exp --workers 8

pending.jsonl 行: {data_source, ground_truth, prompt:[{role,content}...], label, correct_count, n, pver, ts}
sft_ready.jsonl 行: {messages:[sys,user,assistant], data_source, ability, extra_info}
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class RateLimiter:
    """全局调用限速:保证相邻 codex 调用间隔 >= 3600/per_hour 秒(线程安全)。"""

    def __init__(self, per_hour: float):
        self.interval = 3600.0 / per_hour if per_hour and per_hour > 0 else 0.0
        self.lock = threading.Lock()
        self.next_t = 0.0

    def acquire(self):
        if self.interval <= 0:
            return
        with self.lock:
            now = time.time()
            wait = max(0.0, self.next_t - now)
            self.next_t = max(now, self.next_t) + self.interval
        if wait > 0:
            time.sleep(wait)


# codex 调用用的中性空目录(避免加载仓库上下文抬高 token)
_CODEX_CWD = "/tmp/codex_teacher_cwd"
os.makedirs(_CODEX_CWD, exist_ok=True)

DEFAULT_SYSTEM = (
    "Solve the problem step by step. Put your full step-by-step reasoning between "
    "<think> and </think>, expanding every multi-digit calculation; then give the "
    "final answer on a new line as #### \\boxed{ANSWER}."
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_offset(path: str) -> int:
    try:
        return int(open(path).read().strip())
    except Exception:
        return 0


def write_offset(path: str, n: int):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(n))
    os.replace(tmp, path)


def build_teacher_prompt(system: str, user: str) -> str:
    # 给教师我们的答案规范,让补全直接落到训练格式(这是格式约束,非内容过滤)。
    return (
        f"{system}\n\n"
        "Answer-format rules: intervals as (a, b); fractions as a/b; use sqrt(n) or \\sqrt{n}; "
        "do NOT append decimal approximations after the boxed answer.\n\n"
        f"Problem: {user}"
    )


_MIN_SYSTEM = "You are a precise math solver."


def _call_codex(prompt: str, model: str, effort: str, timeout: int) -> str | None:
    of = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False).name
    try:
        # 空目录调用,避免 codex 加载仓库 AGENTS.md/CLAUDE.md/扫文件树抬高 token。
        r = subprocess.run(
            ["codex", "exec", "-m", model, "-s", "read-only",
             "-c", f"model_reasoning_effort={effort}", "-o", of, prompt],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=timeout, cwd=_CODEX_CWD,
        )
        if r.returncode != 0:
            return None
        out = open(of).read().strip()
        return out or None
    except Exception:
        return None
    finally:
        try:
            os.unlink(of)
        except Exception:
            pass


def _call_claude(prompt: str, model: str, timeout: int) -> str | None:
    # --system-prompt 覆盖默认 agent 系统提示、--setting-sources "" 不加载 CLAUDE.md
    # → 近零固定开销;空目录调用,< /dev/null 跳过 stdin 等待。
    try:
        r = subprocess.run(
            ["claude", "-p", prompt, "--model", model,
             "--system-prompt", _MIN_SYSTEM, "--setting-sources", ""],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=timeout, cwd=_CODEX_CWD, text=True,
        )
        if r.returncode != 0:
            return None
        out = (r.stdout or "").strip()
        return out or None
    except Exception:
        return None


class Backend:
    """一个教师后端(claude 或 codex):自带限速 + 失败冷却。"""

    def __init__(self, name, model, per_hour, effort, cooldown, timeout):
        self.name = name
        self.model = model
        self.effort = effort
        self.cooldown = cooldown
        self.timeout = timeout
        self.limiter = RateLimiter(per_hour)
        self.cool_until = 0.0
        self.lock = threading.Lock()
        self.ok = 0
        self.fail = 0

    def available(self) -> bool:
        return time.time() >= self.cool_until

    def _mark_cool(self):
        with self.lock:
            self.cool_until = time.time() + self.cooldown

    def call(self, prompt: str) -> str | None:
        self.limiter.acquire()
        if self.name == "claude":
            out = _call_claude(prompt, self.model, self.timeout)
        else:
            out = _call_codex(prompt, self.model, self.effort, self.timeout)
        with self.lock:
            if out is None:
                self.fail += 1
            else:
                self.ok += 1
        if out is None:
            self._mark_cool()  # 失败(疑似限流)→ 冷却,路由切到另一后端
        return out


# 由 main 按 --backends 构建,顺序=偏好(前者优先,限流则切后者)
_BACKENDS: list = []


def call_teacher(prompt: str):
    """按额度路由:挑第一个未冷却的后端调用;失败则就近切到下一个未冷却后端。

    返回 (文本, 后端名);两个后端都冷却/失败 → (None, None),交给批次 ALL-FAILED 退避。
    """
    tried = 0
    for b in _BACKENDS:
        if not b.available():
            continue
        tried += 1
        out = b.call(prompt)
        if out is not None:
            return out, b.name
    # 若全部都在冷却,至少试一次偏好后端(可能刚好恢复)
    if tried == 0 and _BACKENDS:
        out = _BACKENDS[0].call(prompt)
        if out is not None:
            return out, _BACKENDS[0].name
    return None, None


def _find_last_boxed(text: str):
    """返回最后一个 \\boxed{...} 的 (start_idx, full_substring),处理嵌套大括号。"""
    key = "\\boxed{"
    i = text.rfind(key)
    if i < 0:
        return None, None
    j = i + len(key)
    depth = 1
    while j < len(text) and depth > 0:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return i, text[i:j]


def normalize_assistant(text: str) -> str:
    """规整为我们的 SFT 格式: <think>{reasoning}</think>\\n#### \\boxed{ANS}。

    教师有时不写 <think> 包裹或不写 #### 标记。这是格式规整(非内容过滤):
    不改推理内容,只保证标签/答案行合规,否则 SFT 教不出我们的输出约定。
    """
    text = text.strip()
    has_think = "<think>" in text and "</think>" in text
    has_marker = "####" in text
    if has_think and has_marker:
        return text  # 已合规

    # 拆出答案行 vs 推理
    if has_marker:
        k = text.rfind("####")
        reasoning = text[:k]
        ans_line = "#### " + text[k + 4:].strip()
    else:
        bi, boxed = _find_last_boxed(text)
        if boxed is None:
            # 没有 boxed:无法构造答案行,原样包 think(下游判错也按用户决定保留)
            return f"<think>\n{text}\n</think>\n#### \\boxed{{}}"
        reasoning = text[:bi]
        ans_line = "#### " + boxed
    # 去掉残留的 think 标签后重新包裹
    reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()
    return f"<think>\n{reasoning}\n</think>\n{ans_line.strip()}"


def extract_messages(entry: dict):
    """从 pending 行取出 system / user。"""
    sys_msg, user_msg = DEFAULT_SYSTEM, None
    for m in entry.get("prompt", []):
        if m.get("role") == "system":
            sys_msg = m.get("content", sys_msg)
        elif m.get("role") == "user":
            user_msg = m.get("content")
    return sys_msg, user_msg


def process_one(entry: dict, args) -> dict | None:
    sys_msg, user_msg = extract_messages(entry)
    if not user_msg:
        return None
    tp = build_teacher_prompt(sys_msg, user_msg)
    resp, used_backend = call_teacher(tp)
    if resp is None:
        return None
    resp = normalize_assistant(resp)
    extra = {
        "wrongpool": True,
        "label": entry.get("label"),
        "correct_count": entry.get("correct_count"),
        "n": entry.get("n"),
        "pver": entry.get("pver"),
        "teacher_backend": used_backend,
        "ground_truth": entry.get("ground_truth"),
    }
    return {
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": resp},
        ],
        "data_source": entry.get("data_source"),
        "ability": "math",
        "extra_info": json.dumps(extra, ensure_ascii=False),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", default=os.environ.get("WRONGPOOL_DIR", "/data/zilu/fastrl/wrongpool_v3exp"))
    ap.add_argument("--backends", default="claude,codex",
                    help="后端偏好顺序(逗号分隔):前者优先,限流冷却则切后者。如 'claude,codex' 或 'claude'")
    ap.add_argument("--claude-model", default="claude-haiku-4-5")
    ap.add_argument("--codex-model", default="gpt-5.3-codex-spark")
    ap.add_argument("--codex-effort", default="low", choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--claude-per-hour", type=float, default=0, help="claude 速率上限(次/h),0=不限")
    ap.add_argument("--codex-per-hour", type=float, default=60, help="codex 速率上限(次/h),0=不限")
    ap.add_argument("--cooldown", type=float, default=900, help="后端失败(疑似限流)后冷却秒数,期间路由切到另一后端")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=240, help="per teacher call (s)")
    ap.add_argument("--poll-interval", type=float, default=10.0, help="sleep when caught up (s)")
    ap.add_argument("--batch-size", type=int, default=40,
                    help="每批最多处理多少条(让 offset 定期推进)")
    ap.add_argument("--once", action="store_true", help="drain pending once and exit (for smoke)")
    args = ap.parse_args()

    global _BACKENDS
    _BACKENDS = []
    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        if name == "claude":
            _BACKENDS.append(Backend("claude", args.claude_model, args.claude_per_hour,
                                     None, args.cooldown, args.timeout))
        elif name == "codex":
            _BACKENDS.append(Backend("codex", args.codex_model, args.codex_per_hour,
                                     args.codex_effort, args.cooldown, args.timeout))
        else:
            log(f"unknown backend '{name}', skip")
    if not _BACKENDS:
        log("ERROR: no valid backend; exit")
        return

    os.makedirs(args.pool_dir, exist_ok=True)
    pending = os.path.join(args.pool_dir, "pending.jsonl")
    ready = os.path.join(args.pool_dir, "sft_ready.jsonl")
    offset_f = os.path.join(args.pool_dir, ".teacher_offset")

    desc = ", ".join(f"{b.name}({b.model},{b.limiter.interval:.0f}s)" for b in _BACKENDS)
    log(f"teacher_worker start | pool={args.pool_dir} workers={args.workers} | backends: {desc}")

    total_done = read_offset(offset_f)
    log(f"resume from offset={total_done}")

    backoff = 0  # 连续整批失败次数(疑似 codex 限流/异常)→ 指数退避,不推进 offset

    while True:
        if not os.path.exists(pending):
            if args.once:
                log("no pending file; exit (--once)")
                return
            time.sleep(args.poll_interval)
            continue
        # read all lines; process from offset
        with open(pending) as f:
            lines = f.readlines()
        new = lines[total_done:total_done + args.batch_size]
        if not new:
            if args.once:
                log(f"drained; total_done={total_done}; exit (--once)")
                return
            time.sleep(args.poll_interval)
            continue

        entries = []
        for ln in new:
            ln = ln.strip()
            if not ln:
                entries.append(None)
                continue
            try:
                entries.append(json.loads(ln))
            except Exception:
                entries.append(None)

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda e: process_one(e, args) if e else None, entries))

        n_valid = sum(1 for e in entries if e is not None)
        ok = 0
        with open(ready, "a") as out:
            for r in results:
                if r is not None:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    ok += 1
            out.flush()
        dt = time.time() - t0

        # 鲁棒性:有有效条目却整批全失败 → 疑似 codex 限流/异常。不推进 offset,
        # 指数退避后重试同一批(限流恢复后接着做,不丢错题)。--once 例外(冒烟不卡死)。
        if ok == 0 and n_valid > 0 and not args.once:
            backoff = min(backoff + 1, 6)
            wait = min(30 * (2 ** (backoff - 1)), 1800)  # 30s,60s,...,封顶30min
            log(f"batch ALL-FAILED (in={len(new)} valid={n_valid}); likely codex "
                f"rate-limit/outage. backoff #{backoff} -> sleep {wait}s, NOT advancing offset.")
            time.sleep(wait)
            continue

        backoff = 0
        total_done += len(new)
        write_offset(offset_f, total_done)
        usage = " ".join(f"{b.name}={b.ok}ok/{b.fail}f{'(cool)' if not b.available() else ''}"
                         for b in _BACKENDS)
        log(f"batch: in={len(new)} ok={ok} fail={len(new)-ok} | {dt:.1f}s "
            f"({dt/max(1,len(new)):.1f}s/item) | total_done={total_done} | {usage}")

        if args.once and not lines[total_done:]:
            log(f"drained all; total_done={total_done}; exit (--once)")
            return


if __name__ == "__main__":
    main()
