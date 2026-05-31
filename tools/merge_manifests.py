#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from llmtrain.data.manifest import merge_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--skip-input-validation", action="store_true")
    args = parser.parse_args()
    paths = merge_manifests(
        args.manifest,
        Path(args.output_dir),
        validate_inputs=not args.skip_input_validation,
    )
    print(json.dumps({"manifest": str(paths.manifest), "meta": str(paths.meta)}, indent=2))


if __name__ == "__main__":
    main()
