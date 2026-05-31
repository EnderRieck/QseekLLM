#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import sys

import _bootstrap  # noqa: F401
from llmtrain.preprocessing.pipeline import run_stream_preprocess
from llmtrain.utils.config import dump_resolved, load_config


class ShutdownRequested(Exception):
    pass


def _install_signal_handlers() -> None:
    def _handle_signal(signum: int, _frame: object) -> None:
        raise ShutdownRequested(f"received {signal.Signals(signum).name}")

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, _handle_signal)


def main() -> None:
    _install_signal_handlers()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", action="store_true", help="Resume from preprocess.state.json and persisted dedup state.")
    parser.add_argument(
        "--cleanup-dedup-work",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override preprocess.cleanup_dedup_work from the config.",
    )
    parser.add_argument("--progress", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars.")
    args = parser.parse_args()
    if args.progress and args.no_progress:
        raise SystemExit("--progress and --no-progress are mutually exclusive")
    if args.no_progress:
        os.environ["LLMTRAIN_PROGRESS"] = "0"
    else:
        os.environ.setdefault("LLMTRAIN_PROGRESS", "1")
    cfg, chain = load_config(args.config, args.override)
    if cfg.preprocess is None:
        raise SystemExit("preprocess config section is required")
    if args.cleanup_dedup_work is not None:
        cfg = cfg.model_copy(
            update={
                "preprocess": cfg.preprocess.model_copy(update={"cleanup_dedup_work": args.cleanup_dedup_work})
            }
        )
    dump_resolved(cfg, cfg.run.output_dir, chain)
    try:
        summary = run_stream_preprocess(cfg.preprocess, resume=args.resume)
    except ShutdownRequested as exc:
        print(f"stream_preprocess interrupted: {exc}", file=sys.stderr)
        raise SystemExit(130) from exc
    except KeyboardInterrupt as exc:
        print("stream_preprocess interrupted", file=sys.stderr)
        raise SystemExit(130) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
