from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WPLCSample:
    id: str
    masked_text: str
    context: str
    target: str


def normalize_wplc_record(row: dict, *, index: int) -> WPLCSample:
    try:
        masked_text = str(row["masked_text"])
        target = str(row["correct_word"])
    except KeyError as exc:
        raise ValueError(f"WPLC row {index} is missing required field: {exc.args[0]}") from exc
    marker = "<mask>"
    if marker not in masked_text:
        raise ValueError(f"WPLC row {index} does not contain {marker}")
    context = masked_text.split(marker, 1)[0]
    if target == "":
        raise ValueError(f"WPLC row {index} has an empty correct_word")
    return WPLCSample(id=str(index), masked_text=masked_text, context=context, target=target)


def prepare_wplc_dataset(source_path: str | Path, output_path: str | Path) -> dict[str, int | str]:
    source = Path(source_path).expanduser()
    output = Path(output_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"WPLC source file does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for index, line in enumerate(src):
            if not line.strip():
                continue
            sample = normalize_wplc_record(json.loads(line), index=index)
            dst.write(
                json.dumps(
                    {
                        "id": sample.id,
                        "masked_text": sample.masked_text,
                        "context": sample.context,
                        "target": sample.target,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rows += 1
    return {"source_path": str(source), "output_path": str(output), "samples": rows}
