from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable


def expand_paths(paths: Iterable[Path], *, recursive: bool = True) -> list[Path]:
    out: list[Path] = []
    for item in paths:
        text = str(item)
        if any(ch in text for ch in "*?[]"):
            out.extend(Path(p) for p in sorted(glob.glob(text, recursive=recursive)))
        elif item.is_dir():
            pattern = "**/*" if recursive else "*"
            out.extend(sorted(p for p in item.glob(pattern) if p.is_file()))
        else:
            out.append(item)
    return out


def is_probably_binary(path: Path, sample_size: int = 4096) -> bool:
    try:
        data = path.open("rb").read(sample_size)
    except OSError:
        return True
    return b"\x00" in data
