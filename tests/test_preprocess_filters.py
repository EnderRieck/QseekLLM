import json

from llmtrain.preprocessing.config import PreprocessConfig
from llmtrain.preprocessing.pipeline import run_stream_preprocess


def test_preprocess_excludes_configured_metadata_values(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps({"text": "这是来自 CCI3 的文本。" * 20, "source": "CCI3"}, ensure_ascii=False),
                json.dumps({"text": "这是来自 ChineseWebText 的文本。" * 20, "source": "ChineseWebText"}, ensure_ascii=False),
                json.dumps({"text": "这是来自其他来源的高质量中文教育文本。" * 20, "source": "OTHER"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    cfg = PreprocessConfig.model_validate(
        {
            "sources": [
                {
                    "name": "fineweb_edu_chinese_v21",
                    "type": "jsonl",
                    "domain": "zh",
                    "language": "zh",
                    "paths": [str(raw)],
                    "text_field": "text",
                    "id_field": None,
                    "metadata_fields": ["source"],
                    "exclude_metadata_values": {"source": ["CCI3", "ChineseWebText"]},
                    "weight": 1.0,
                }
            ],
            "cleaning": {"min_chars": 1, "min_alpha_or_cjk_ratio": 0.0},
            "dedup": {"exact": False, "simhash": False, "persist_state": False},
            "quality": {"min_score": 0.0},
            "writer": {
                "output_dir": str(out),
                "output_format": "jsonl",
                "shard_prefix": "clean",
                "shard_max_bytes": 1_000_000,
                "rejected_path": str(out / "rejected.jsonl"),
            },
        }
    )

    result = run_stream_preprocess(cfg)

    assert result["sources"]["fineweb_edu_chinese_v21"]["seen"] == 3
    assert result["sources"]["fineweb_edu_chinese_v21"]["rejected_excluded_metadata_source"] == 2
    assert result["sources"]["fineweb_edu_chinese_v21"]["written"] == 1
    rejected = (out / "rejected.jsonl").read_text(encoding="utf-8")
    assert "excluded_metadata_source" in rejected
