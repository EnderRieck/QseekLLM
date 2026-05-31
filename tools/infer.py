#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from llmtrain.inference import InferenceEngine, load_inference_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text generation from an llmtrain checkpoint.")
    parser.add_argument("--config", required=True, help="Training config used to build the model/tokenizer.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory. Defaults to run.output_dir/checkpoints/latest.")
    parser.add_argument("--inference-config", default="configs/inference/default.yaml")
    parser.add_argument("--override", action="append", default=[], help="Override inference config, e.g. generation.temperature=0.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--input-jsonl", default=None, help="Batch mode input JSONL.")
    parser.add_argument("--output-jsonl", default=None, help="Batch mode output JSONL. Defaults to stdout.")
    parser.add_argument("--stream", action="store_true", help="Stream a single prompt token-by-token.")
    args = parser.parse_args()

    infer_cfg = load_inference_config(args.inference_config, args.override)
    engine = InferenceEngine.from_config_path(args.config, checkpoint_path=args.checkpoint, runtime=infer_cfg.runtime)

    if args.input_jsonl:
        _run_batch(engine, args.input_jsonl, args.output_jsonl, infer_cfg)
        return

    prompt = _read_prompt(args.prompt, args.prompt_file)
    if args.stream:
        for step in engine.iter_generate(prompt, infer_cfg.generation):
            print(step.text, end="", flush=True)
        print()
        return

    result = engine.generate(prompt, infer_cfg.generation)
    print(result.text)


def _read_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if prompt is not None and prompt_file is not None:
        raise ValueError("Use only one of --prompt or --prompt-file")
    if prompt_file is not None:
        return Path(prompt_file).read_text(encoding="utf-8")
    if prompt is not None:
        return prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("Provide --prompt, --prompt-file, --input-jsonl, or stdin")


def _run_batch(engine: InferenceEngine, input_jsonl: str, output_jsonl: str | None, infer_cfg) -> None:
    out_f = Path(output_jsonl).open("w", encoding="utf-8") if output_jsonl else sys.stdout
    close = output_jsonl is not None
    try:
        with Path(input_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = str(row[infer_cfg.batch.prompt_field])
                result = engine.generate(prompt, infer_cfg.generation)
                if infer_cfg.batch.include_metadata:
                    row[infer_cfg.batch.output_field] = result.text
                    row["_generation"] = {
                        "stop_reason": result.stop_reason,
                        "input_tokens": result.input_tokens,
                        "generated_tokens": result.generated_tokens,
                    }
                else:
                    row = {infer_cfg.batch.output_field: result.text}
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if close:
            out_f.close()


if __name__ == "__main__":
    main()
