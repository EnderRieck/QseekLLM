#!/usr/bin/env python3
"""Analyze the fully-async GRPO event log (``*_events.jsonl``).

The trainer's EventLogger writes one JSON record per key event with an absolute
``ts`` (epoch seconds) and a ``t`` offset from logger start. This script replays
that stream to recover information the aggregated TensorBoard step metrics hide:

  * per-actor-update wall-clock (sync>1 aggregates these into one TB point);
  * param-sync durations over time;
  * ref-service micro-batch throughput and compute-time distribution;
  * validation cost and how long sampling was paused for held-out eval;
  * concurrency: how much ref-service compute overlapped actor-update windows.

Usage:
    python3 RL/analyze_events.py logs/<run>_events.jsonl
"""

import json
import sys
from collections import defaultdict


def load(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def summarize(values):
    if not values:
        return "n=0"
    return (
        f"n={len(values)} mean={sum(values) / len(values):.2f}s "
        f"min={min(values):.2f} p50={pct(values, 50):.2f} "
        f"p95={pct(values, 95):.2f} max={max(values):.2f}"
    )


def interval_overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    events = load(path)
    if not events:
        print(f"no events parsed from {path}")
        sys.exit(1)

    by_ev = defaultdict(list)
    for e in events:
        by_ev[e["ev"]].append(e)

    print(f"=== event log: {path} ===")
    span = events[-1]["t"] - events[0]["t"]
    print(f"records={len(events)} span={span:.1f}s event_types={dict((k, len(v)) for k, v in sorted(by_ev.items()))}")

    # --- per-actor-update wall-clock (the granularity sync>1 hides) ---
    update_durs = [e["update_s"] for e in by_ev.get("update_actor_end", []) if "update_s" in e]
    print("\n[per-actor-update wall-clock]  ", summarize(update_durs))
    # group by param version to show within-version spread
    by_pver = defaultdict(list)
    for e in by_ev.get("update_actor_end", []):
        if "update_s" in e:
            by_pver[e.get("pver")].append(e["update_s"])
    for pver in sorted(k for k in by_pver if k is not None):
        durs = by_pver[pver]
        print(f"    pver={pver}: {len(durs)} updates -> {[round(d, 1) for d in durs]}")

    # --- param sync ---
    ps = [e["param_sync_s"] for e in by_ev.get("param_sync_end", []) if "param_sync_s" in e]
    print("\n[param_sync]                   ", summarize(ps))

    # --- ref-service micro-batches ---
    rmb = by_ev.get("ref_micro_batch", [])
    comp = [e["compute_s"] for e in rmb if "compute_s" in e]
    seqs = sum(e.get("seqs", 0) for e in rmb)
    print("\n[ref micro-batch compute]      ", summarize(comp))
    if rmb and span > 0:
        print(f"    ref batches={len(rmb)} total_seqs={seqs} "
              f"throughput={seqs / span:.1f} seq/s (over whole run)")
    readyq = [e.get("ready_q", 0) for e in rmb]
    if readyq:
        print(f"    ready_q depth: mean={sum(readyq) / len(readyq):.1f} max={max(readyq)} "
              f"min={min(readyq)}  (0 == trainer starved)")

    # --- validation: cost + sampling pause ---
    vstart = by_ev.get("validate_start", [])
    vend = by_ev.get("validate_end", [])
    vdurs = [e["validate_s"] for e in vend if "validate_s" in e]
    print("\n[validation]                   ", summarize(vdurs))
    print(f"    eval runs={len(vend)} (incl. val_before_train={sum(1 for e in vstart if e.get('val_before_train'))})")

    # --- concurrency: ref compute overlapped with actor-update windows ---
    starts = {(e.get("gstep"), e.get("local_trigger_step")): e["ts"] for e in by_ev.get("update_actor_start", [])}
    windows = []
    for e in by_ev.get("update_actor_end", []):
        key = (e.get("gstep"), e.get("local_trigger_step"))
        if key in starts:
            windows.append((starts[key], e["ts"]))
    total_update = sum(b - a for a, b in windows)
    overlap = 0.0
    for e in rmb:
        if "compute_s" not in e:
            continue
        rb1 = e["ts"]
        rb0 = rb1 - e["compute_s"]
        for a, b in windows:
            overlap += interval_overlap(rb0, rb1, a, b)
    if total_update > 0:
        print("\n[concurrency] ref-compute hidden under actor-update:")
        print(f"    sum(update windows)={total_update:.1f}s  ref-compute overlapped={overlap:.1f}s  "
              f"ref-hidden-ratio={overlap / max(sum(comp), 1e-9):.2%}")

    print("\n(tip: each record has absolute ts/t — load with pandas for custom timelines.)")


if __name__ == "__main__":
    main()
