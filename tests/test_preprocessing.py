import gzip
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

from tools.shard_wiki_xml import shard_wiki_xml
from llmtrain.data.manifest import load_manifest, validate_manifest
from llmtrain.data.readers import ShardReader
from llmtrain.preprocessing.config import (
    CleaningConfig,
    DedupConfig,
    PreprocessConfig,
    PreprocessSourceConfig,
    PreprocessWriterConfig,
    QualityConfig,
)
from llmtrain.preprocessing.dedup import StreamingDeduper, hamming_distance, simhash
from llmtrain.preprocessing.documents import RawDocument
from llmtrain.preprocessing.parallel import _SimhashShardJob, _reduce_simhash_shard
from llmtrain.preprocessing.parsers import _remote_jsonl_urls, _remote_parquet_urls, iter_documents
from llmtrain.preprocessing.pipeline import run_stream_preprocess


def test_simhash_and_exact_dedup():
    a = simhash("the quick brown fox jumps over the lazy dog")
    b = simhash("the quick brown fox jumps over the lazy dog")
    c = simhash("completely unrelated technical document")
    assert hamming_distance(a, b) == 0
    assert hamming_distance(a, c) > 0

    deduper = StreamingDeduper(DedupConfig(exact=True, simhash=False))
    doc = RawDocument("1", "same text same text", "s", "en", "en", {})
    assert not deduper.check(doc).duplicate
    assert deduper.check(doc).duplicate


def test_stream_preprocess_jsonl_html_wiki_git(tmp_path):
    raw_jsonl = tmp_path / "raw.jsonl"
    raw_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": "Useful open dataset text with enough words for the cleaner."}),
                json.dumps({"id": "b", "text": "Useful open dataset text with enough words for the cleaner."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    (html_dir / "page.html").write_text(
        "<html><title>T</title><body><nav>skip</nav><article>Useful HTML article body with enough language content.</article></body></html>",
        encoding="utf-8",
    )
    wiki = tmp_path / "wiki.xml"
    wiki.write_text(
        "<mediawiki><page><title>测试</title><ns>0</ns><id>1</id><revision><text>'''测试''' 条目包含足够的中文正文，用于验证 Wiki XML 流式解析。</text></revision></page></mediawiki>",
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def useful_function(value):\n    return value + 1\n", encoding="utf-8")

    cfg = PreprocessConfig(
        writer=PreprocessWriterConfig(output_dir=tmp_path / "out", shard_max_bytes=512),
        cleaning=CleaningConfig(min_chars=10, min_alpha_or_cjk_ratio=0.05),
        dedup=DedupConfig(exact=True, simhash=False, state_dir=tmp_path / "dedup"),
        quality=QualityConfig(min_score=0.1),
        sources=[
            PreprocessSourceConfig(name="json", type="jsonl", domain="en", language="en", paths=[raw_jsonl]),
            PreprocessSourceConfig(name="html", type="html", domain="en", language="en", paths=[html_dir]),
            PreprocessSourceConfig(name="wiki", type="wiki_xml", domain="wiki", language="zh", paths=[wiki]),
            PreprocessSourceConfig(name="git", type="git", domain="code", language="multi", paths=[repo], include_extensions=[".py"]),
        ],
    )
    summary = run_stream_preprocess(cfg)
    assert summary["totals"]["written"] >= 4
    assert summary["totals"]["rejected_exact_duplicate"] == 1
    validate_manifest(summary["manifest"])
    shards = load_manifest(summary["manifest"])
    assert shards
    records = list(ShardReader(summary["manifest"]))
    assert {r.domain for r in records} >= {"en", "wiki", "code"}
    assert all("quality_score" in r.metadata for r in records)


def test_stream_preprocess_progress_bar_can_be_forced(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LLMTRAIN_PROGRESS", "1")
    raw_jsonl = tmp_path / "raw.jsonl"
    raw_jsonl.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"r{i}",
                    "text": f"Useful document {i} with enough natural language for progress reporting tests.",
                }
            )
            for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = PreprocessConfig(
        writer=PreprocessWriterConfig(output_dir=tmp_path / "out", shard_max_bytes=10_000),
        cleaning=CleaningConfig(min_chars=10, min_alpha_or_cjk_ratio=0.05),
        dedup=DedupConfig(exact=True, simhash=False, state_dir=tmp_path / "dedup"),
        quality=QualityConfig(min_score=0.1),
        sources=[
            PreprocessSourceConfig(
                name="json_progress",
                type="jsonl",
                domain="en",
                language="en",
                paths=[raw_jsonl],
                limit=2,
            )
        ],
    )

    run_stream_preprocess(cfg)

    err = capsys.readouterr().err
    assert "json_progress" in err
    assert "written=2" in err


def test_parallel_preprocess_global_dedup_across_files(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMTRAIN_PROGRESS", "0")
    p1 = tmp_path / "part1.jsonl"
    p2 = tmp_path / "part2.jsonl"
    duplicate = "Shared useful document with enough natural language for global dedup testing."
    p1.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": duplicate}),
                json.dumps({"id": "b", "text": "Unique first useful document with enough natural language."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    p2.write_text(
        "\n".join(
            [
                json.dumps({"id": "c", "text": duplicate}),
                json.dumps({"id": "d", "text": "Unique second useful document with enough natural language."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    cfg = PreprocessConfig(
        writer=PreprocessWriterConfig(output_dir=output_dir, shard_max_bytes=10_000),
        cleaning=CleaningConfig(min_chars=10, min_alpha_or_cjk_ratio=0.05),
        dedup=DedupConfig(exact=True, simhash=False, state_dir=output_dir / "dedup_state"),
        quality=QualityConfig(min_score=0.1),
        num_workers=2,
        keep_candidates=True,
        sources=[
            PreprocessSourceConfig(
                name="json",
                type="jsonl",
                domain="en",
                language="en",
                paths=[p1, p2],
            )
        ],
    )

    summary = run_stream_preprocess(cfg)

    assert summary["parallel"] is True
    assert summary["totals"]["candidate"] == 4
    assert summary["totals"]["written"] == 3
    assert summary["totals"]["rejected_exact_duplicate"] == 1
    records = list(ShardReader(summary["manifest"]))
    assert len(records) == 3
    assert len(list((output_dir / "candidates").glob("*.done.json"))) == 2


def test_parallel_preprocess_resume_reuses_completed_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMTRAIN_PROGRESS", "0")
    p1 = tmp_path / "part1.jsonl"
    p2 = tmp_path / "part2.jsonl"
    p1.write_text(json.dumps({"id": "a", "text": "Useful first document with enough natural language."}) + "\n", encoding="utf-8")
    p2.write_text(json.dumps({"id": "b", "text": "Useful second document with enough natural language."}) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    cfg = PreprocessConfig(
        writer=PreprocessWriterConfig(output_dir=output_dir, shard_max_bytes=10_000),
        cleaning=CleaningConfig(min_chars=10, min_alpha_or_cjk_ratio=0.05),
        dedup=DedupConfig(exact=True, simhash=False, state_dir=output_dir / "dedup_state"),
        quality=QualityConfig(min_score=0.1),
        num_workers=2,
        keep_candidates=True,
        cleanup_dedup_work=False,
        sources=[
            PreprocessSourceConfig(name="json", type="jsonl", domain="en", language="en", paths=[p1, p2])
        ],
    )
    first = run_stream_preprocess(cfg)
    candidate = output_dir / "candidates" / "json_000000.jsonl.zst"
    before_mtime = candidate.stat().st_mtime_ns

    second = run_stream_preprocess(cfg, resume=True)

    assert first["totals"]["written"] == 2
    assert second["totals"]["written"] == 2
    assert candidate.stat().st_mtime_ns == before_mtime
    assert len(list(ShardReader(second["manifest"]))) == 2


def test_parallel_preprocess_resume_reuses_dedup_work(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMTRAIN_PROGRESS", "0")
    p1 = tmp_path / "part1.jsonl"
    p2 = tmp_path / "part2.jsonl"
    p1.write_text(json.dumps({"id": "a", "text": "Useful first document with enough natural language."}) + "\n", encoding="utf-8")
    p2.write_text(json.dumps({"id": "b", "text": "Useful second document with enough natural language."}) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    cfg = PreprocessConfig(
        writer=PreprocessWriterConfig(output_dir=output_dir, shard_max_bytes=10_000),
        cleaning=CleaningConfig(min_chars=10, min_alpha_or_cjk_ratio=0.05),
        dedup=DedupConfig(exact=True, simhash=False, state_dir=output_dir / "dedup_state"),
        quality=QualityConfig(min_score=0.1),
        num_workers=2,
        keep_candidates=True,
        cleanup_dedup_work=False,
        sources=[
            PreprocessSourceConfig(name="json", type="jsonl", domain="en", language="en", paths=[p1, p2])
        ],
    )

    first = run_stream_preprocess(cfg)
    done_files = sorted((output_dir / "dedup_work" / "materialize").glob("*.done.json"))
    assert done_files
    before = {path: path.stat().st_mtime_ns for path in done_files}

    second = run_stream_preprocess(cfg, resume=True)

    assert first["totals"]["written"] == second["totals"]["written"] == 2
    assert {path: path.stat().st_mtime_ns for path in done_files} == before


def test_parallel_preprocess_simhash_near_dedup_across_files(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMTRAIN_PROGRESS", "0")
    p1 = tmp_path / "part1.jsonl"
    p2 = tmp_path / "part2.jsonl"
    base = "Alpha beta gamma useful near duplicate document with enough natural language."
    p1.write_text(json.dumps({"id": "a", "text": base}) + "\n", encoding="utf-8")
    p2.write_text(json.dumps({"id": "b", "text": base + " extra"}) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    cfg = PreprocessConfig(
        writer=PreprocessWriterConfig(output_dir=output_dir, shard_max_bytes=10_000),
        cleaning=CleaningConfig(min_chars=10, min_alpha_or_cjk_ratio=0.05),
        dedup=DedupConfig(exact=False, simhash=True, simhash_threshold=8, state_dir=output_dir / "dedup_state"),
        quality=QualityConfig(min_score=0.1),
        num_workers=2,
        keep_candidates=True,
        sources=[
            PreprocessSourceConfig(name="json", type="jsonl", domain="en", language="en", paths=[p1, p2])
        ],
    )

    summary = run_stream_preprocess(cfg)

    assert summary["parallel_dedup"] is True
    assert summary["totals"]["candidate"] == 2
    assert summary["totals"]["written"] == 1
    assert summary["totals"]["rejected_near_duplicate"] == 1
    assert len(list(ShardReader(summary["manifest"]))) == 1


def test_reduce_simhash_shard_caps_group_size(tmp_path, monkeypatch):
    index_path = tmp_path / "index.jsonl"
    rows = [
        {"file_index": 0, "line_index": 0, "id": "a", "source": "json", "band": 0, "band_value": 7, "simhash": "0"},
        {"file_index": 0, "line_index": 1, "id": "b", "source": "json", "band": 0, "band_value": 7, "simhash": "0"},
        {"file_index": 0, "line_index": 2, "id": "c", "source": "json", "band": 0, "band_value": 7, "simhash": "8"},
    ]
    index_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = _reduce_simhash_shard(
        _SimhashShardJob(
            paths=[index_path],
            drop_path=tmp_path / "drops.jsonl",
            bits=64,
            threshold=0,
            max_group_size=1,
        )
    )

    assert result["dropped"] == 1
    assert result["capped"] == 1
    assert result["max_group_size"] == 1


def test_shard_wiki_xml_writes_jsonl_shards(tmp_path):
    wiki = tmp_path / "wiki.xml"
    wiki.write_text(
        "<mediawiki>"
        "<page><title>A</title><ns>0</ns><id>1</id><revision><text>'''Alpha''' article body.</text></revision></page>"
        "<page><title>Talk</title><ns>1</ns><id>2</id><revision><text>skip talk</text></revision></page>"
        "<page><title>B</title><ns>0</ns><id>3</id><revision><text>[[Beta|Beta page]] article body.</text></revision></page>"
        "<page><title>C</title><ns>0</ns><id>4</id><revision><text>Gamma article body.</text></revision></page>"
        "</mediawiki>",
        encoding="utf-8",
    )
    out = tmp_path / "shards"

    summary = shard_wiki_xml(
        input_path=wiki,
        output_dir=out,
        source_name="testwiki",
        language="en",
        license_name="CC-BY-SA",
        shard_docs=2,
    )

    assert summary["total_docs"] == 3
    shards = sorted(out.glob("*.jsonl"))
    assert [p.name for p in shards] == ["testwiki_000000.jsonl", "testwiki_000001.jsonl"]
    rows = [json.loads(line) for shard in shards for line in shard.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["testwiki/1", "testwiki/3", "testwiki/4"]
    assert rows[1]["text"] == "Beta page article body."


def test_remote_parquet_url_list(tmp_path):
    urls = tmp_path / "urls.txt"
    urls.write_text(
        "https://example.test/datasets/repo/resolve/main/data/a.parquet\n"
        "https://example.test/datasets/repo/resolve/main/data/b.parquet\n",
        encoding="utf-8",
    )
    source = PreprocessSourceConfig(
        name="remote",
        type="remote_parquet",
        domain="en",
        language="en",
        url_list_path=urls,
    )
    assert _remote_parquet_urls(source) == [
        "https://example.test/datasets/repo/resolve/main/data/a.parquet",
        "https://example.test/datasets/repo/resolve/main/data/b.parquet",
    ]


def test_remote_jsonl_gz_stream(monkeypatch):
    payload = gzip.compress(
        (
            json.dumps({"id": "a", "text": "远端 gzip jsonl 文档包含足够正文用于测试流式读取。", "url": "https://example.test/a"})
            + "\n"
            + json.dumps({"id": "b", "text": "第二条远端 gzip jsonl 文档也包含足够正文。"})
            + "\n"
        ).encode("utf-8")
    )

    class Response:
        raw = io.BytesIO(payload)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        return Response()

    monkeypatch.setattr("llmtrain.preprocessing.parsers.requests.get", fake_get)
    source = PreprocessSourceConfig(
        name="remote_jsonl",
        type="remote_jsonl",
        domain="zh",
        language="zh",
        urls=["https://example.test/data/part-0001.jsonl.gz"],
        metadata_fields=["url"],
    )

    docs = list(iter_documents(source))
    assert [doc.id for doc in docs] == ["a", "b"]
    assert docs[0].metadata["remote_line"] == 1
    assert docs[0].metadata["parser"] == "remote_jsonl"


def test_local_jsonl_gz_stream(tmp_path):
    path = tmp_path / "local.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a", "text": "本地 gzip jsonl 文档包含足够正文用于测试流式读取。"}) + "\n")

    source = PreprocessSourceConfig(
        name="local_jsonl",
        type="jsonl",
        domain="zh",
        language="zh",
        paths=[path],
    )
    docs = list(iter_documents(source))
    assert [doc.id for doc in docs] == ["a"]
    assert docs[0].metadata["line"] == 1


def test_local_json_gz_zstd_stream(tmp_path):
    path = tmp_path / "local.json.gz"
    path.write_bytes(
        zstd.ZstdCompressor().compress(
            (json.dumps({"id": "a", "text": "Local zstd jsonl document has enough useful language for tests."}) + "\n").encode("utf-8")
        )
    )

    source = PreprocessSourceConfig(
        name="local_jsonl",
        type="jsonl",
        domain="en",
        language="en",
        paths=[path],
    )
    docs = list(iter_documents(source))
    assert [doc.id for doc in docs] == ["a"]


def test_remote_jsonl_zstd_stream(monkeypatch):
    payload = zstd.ZstdCompressor().compress(
        (
            json.dumps({"id": "a", "text": "Remote zstd jsonl document has enough useful language for tests."})
            + "\n"
        ).encode("utf-8")
    )

    class Response:
        raw = io.BytesIO(payload)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        return Response()

    monkeypatch.setattr("llmtrain.preprocessing.parsers.requests.get", fake_get)
    source = PreprocessSourceConfig(
        name="remote_jsonl",
        type="remote_jsonl",
        domain="en",
        language="en",
        urls=["https://example.test/data/part-0001.json.gz"],
    )

    docs = list(iter_documents(source))
    assert [doc.id for doc in docs] == ["a"]
    assert docs[0].metadata["remote_line"] == 1


def test_remote_jsonl_nested_zstd_gzip_stream(monkeypatch):
    jsonl = (
        json.dumps({"id": "a", "text": "Nested compressed jsonl document has enough useful language for tests."})
        + "\n"
    ).encode("utf-8")
    payload = zstd.ZstdCompressor().compress(gzip.compress(jsonl))

    class Response:
        raw = io.BytesIO(payload)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        return Response()

    monkeypatch.setattr("llmtrain.preprocessing.parsers.requests.get", fake_get)
    source = PreprocessSourceConfig(
        name="remote_jsonl",
        type="remote_jsonl",
        domain="en",
        language="en",
        urls=["https://example.test/data/part-0001.json.gz"],
    )

    docs = list(iter_documents(source))
    assert [doc.id for doc in docs] == ["a"]


def test_remote_jsonl_url_list(tmp_path):
    urls = tmp_path / "urls.txt"
    urls.write_text("https://example.test/data/part-0001.jsonl.gz\n", encoding="utf-8")
    source = PreprocessSourceConfig(
        name="remote_jsonl",
        type="remote_jsonl",
        domain="zh",
        language="zh",
        url_list_path=urls,
    )
    assert _remote_jsonl_urls(source) == ["https://example.test/data/part-0001.jsonl.gz"]


def test_remote_jsonl_remote_url_list(monkeypatch):
    class Response:
        text = "https://example.test/data/part-0001.json.gz\n"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        return Response()

    monkeypatch.setattr("llmtrain.preprocessing.parsers.requests.get", fake_get)
    source = PreprocessSourceConfig(
        name="remote_jsonl",
        type="remote_jsonl",
        domain="en",
        language="en",
        url_list_url="https://example.test/urls.txt",
    )
    assert _remote_jsonl_urls(source) == ["https://example.test/data/part-0001.json.gz"]


def test_stream_preprocess_resume_skips_seen_records(tmp_path):
    raw_jsonl = tmp_path / "raw.jsonl"
    rows = [
        {"id": f"r{i}", "text": f"Useful document number {i} with enough natural language for preprocessing resume tests."}
        for i in range(5)
    ]
    raw_jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    output_dir = tmp_path / "out"
    cfg = PreprocessConfig(
        writer=PreprocessWriterConfig(output_dir=output_dir, shard_max_bytes=10_000),
        cleaning=CleaningConfig(min_chars=10, min_alpha_or_cjk_ratio=0.05),
        dedup=DedupConfig(exact=True, simhash=False, state_dir=output_dir / "dedup_state"),
        quality=QualityConfig(min_score=0.1),
        checkpoint_interval_records=1,
        sources=[
            PreprocessSourceConfig(
                name="json",
                type="jsonl",
                domain="en",
                language="en",
                paths=[raw_jsonl],
                limit=2,
            )
        ],
    )
    first = run_stream_preprocess(cfg)
    assert first["totals"]["seen"] == 2
    assert first["totals"]["written"] == 2
    state_path = output_dir / "preprocess.state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["sources"]["json"]["completed"] = False
    state_path.write_text(json.dumps(state), encoding="utf-8")

    resumed_cfg = cfg.model_copy(
        update={
            "sources": [
                cfg.sources[0].model_copy(update={"limit": 5})
            ]
        }
    )
    second = run_stream_preprocess(resumed_cfg, resume=True)
    assert second["totals"]["seen"] == 5
    assert second["totals"]["written"] == 5
    records = list(ShardReader(second["manifest"]))
    assert len(records) == 5
    assert {record.id for record in records} == {f"r{i}" for i in range(5)}


def test_remote_parquet_resume_tracks_url_offsets(tmp_path):
    p1 = tmp_path / "part1.parquet"
    p2 = tmp_path / "part2.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "p1_0", "text": "Remote parquet document zero has enough useful language for preprocessing."},
                {"id": "p1_1", "text": "Remote parquet document one has enough useful language for preprocessing."},
            ]
        ),
        p1,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "p2_0", "text": "Remote parquet document two has enough useful language for preprocessing."},
                {"id": "p2_1", "text": "Remote parquet document three has enough useful language for preprocessing."},
            ]
        ),
        p2,
    )

    output_dir = tmp_path / "remote_out"
    cfg = PreprocessConfig(
        writer=PreprocessWriterConfig(output_dir=output_dir, shard_max_bytes=10_000),
        cleaning=CleaningConfig(min_chars=10, min_alpha_or_cjk_ratio=0.05),
        dedup=DedupConfig(exact=True, simhash=False, state_dir=output_dir / "dedup_state"),
        quality=QualityConfig(min_score=0.1),
        checkpoint_interval_records=1,
        sources=[
            PreprocessSourceConfig(
                name="remote",
                type="remote_parquet",
                domain="en",
                language="en",
                urls=[str(p1), str(p2)],
                limit=3,
            )
        ],
    )
    first = run_stream_preprocess(cfg)
    assert first["totals"]["seen"] == 3
    state_path = output_dir / "preprocess.state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    remote_state = state["sources"]["remote"]["remote_parquet"]
    assert str(p1) in remote_state["completed_urls"]
    assert remote_state["current_url"] == str(p2)
    assert remote_state["current_url_seen"] == 1

    state["sources"]["remote"]["completed"] = False
    state_path.write_text(json.dumps(state), encoding="utf-8")
    resumed_cfg = cfg.model_copy(
        update={"sources": [cfg.sources[0].model_copy(update={"limit": 4})]}
    )
    second = run_stream_preprocess(resumed_cfg, resume=True)
    records = list(ShardReader(second["manifest"]))
    assert len(records) == 4
    assert {record.id for record in records} == {"p1_0", "p1_1", "p2_0", "p2_1"}
