from __future__ import annotations

import gzip
import io
import json
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pyarrow.parquet as pq
import requests
import zstandard as zstd
from lxml import html

from llmtrain.preprocessing.config import PreprocessSourceConfig
from llmtrain.preprocessing.documents import RawDocument
from llmtrain.preprocessing.sources import expand_paths, is_probably_binary


def iter_documents(
    source: PreprocessSourceConfig,
    *,
    skip: int = 0,
    resume_state: dict[str, Any] | None = None,
) -> Iterator[RawDocument]:
    source_without_limit = source.model_copy(update={"limit": None})
    if source.type == "remote_parquet" and resume_state:
        max_yield = None if source.limit is None else max(0, source.limit - skip)
        for yielded, doc in enumerate(_iter_remote_parquet(source_without_limit, resume_state=resume_state)):
            if max_yield is not None and yielded >= max_yield:
                return
            yield doc
        return
    for index, doc in enumerate(_iter_documents_unlimited(source_without_limit)):
        if index < skip:
            continue
        if source.limit is not None and index >= source.limit:
            return
        yield doc


def _iter_documents_unlimited(source: PreprocessSourceConfig) -> Iterator[RawDocument]:
    if source.type == "jsonl":
        yield from _iter_jsonl(source)
    elif source.type == "parquet":
        yield from _iter_parquet(source)
    elif source.type == "remote_jsonl":
        yield from _iter_remote_jsonl(source)
    elif source.type == "remote_parquet":
        yield from _iter_remote_parquet(source)
    elif source.type == "hf_dataset":
        yield from _iter_hf_dataset(source)
    elif source.type == "html":
        yield from _iter_html(source)
    elif source.type == "text":
        yield from _iter_text(source)
    elif source.type == "wiki_xml":
        yield from _iter_wiki_xml(source)
    elif source.type == "git":
        yield from _iter_git(source)
    elif source.type == "pdf":
        yield from _iter_pdf(source)
    else:
        raise ValueError(f"Unsupported source type: {source.type}")


def _doc_from_mapping(source: PreprocessSourceConfig, row: dict[str, Any], index: int) -> RawDocument | None:
    text = row.get(source.text_field)
    if text is None:
        return None
    doc_id = row.get(source.id_field) if source.id_field else None
    metadata = {k: row.get(k) for k in source.metadata_fields if k in row}
    metadata.update({"license": source.license, "parser": source.type})
    return RawDocument(
        id=str(doc_id or f"{source.name}/{index}"),
        text=str(text),
        source=source.name,
        domain=source.domain,
        language=source.language,
        metadata=metadata,
    )


def _iter_jsonl(source: PreprocessSourceConfig) -> Iterator[RawDocument]:
    count = 0
    for path in expand_paths(source.paths):
        line_no = 0
        try:
            with path.open("rb") as raw:
                for line_no, line in enumerate(_iter_text_lines(raw), start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    doc = _doc_from_mapping(source, row, count)
                    if doc is None:
                        continue
                    doc.metadata.update({"path": str(path), "line": line_no})
                    yield doc
                    count += 1
        except Exception as exc:
            raise RuntimeError(f"failed reading jsonl source={source.name} path={path} line={line_no}: {exc}") from None


def _iter_parquet(source: PreprocessSourceConfig) -> Iterator[RawDocument]:
    count = 0
    columns = list({source.text_field, *(source.metadata_fields or []), *(([source.id_field] if source.id_field else []))})
    for path in expand_paths(source.paths):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=columns, batch_size=2048):
            for row in batch.to_pylist():
                doc = _doc_from_mapping(source, row, count)
                if doc is None:
                    continue
                doc.metadata.update({"path": str(path)})
                yield doc
                count += 1


def _iter_remote_jsonl(source: PreprocessSourceConfig) -> Iterator[RawDocument]:
    _configure_hf_environment()
    count = 0
    for url in _remote_jsonl_urls(source):
        with requests.get(url, stream=True, timeout=60, proxies=_proxies_from_env()) as response:
            response.raise_for_status()
            lines = _iter_remote_text_lines(response)
            for line_index, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                doc = _doc_from_mapping(source, row, count)
                if doc is None:
                    continue
                doc.metadata.update({"remote_url": url, "remote_line": line_index, "parser": "remote_jsonl"})
                yield doc
                count += 1


def _iter_remote_text_lines(response: requests.Response) -> Iterator[str]:
    yield from _iter_text_lines(response.raw)


def _iter_text_lines(raw: Any) -> Iterator[str]:
    stream: Any = raw
    closers: list[Any] = []
    for _ in range(4):
        magic = stream.read(4)
        stream = _PrefixedReader(magic, stream)
        if magic.startswith(b"\x1f\x8b"):
            stream = gzip.GzipFile(fileobj=stream)
            closers.append(stream)
            continue
        if magic == b"\x28\xb5\x2f\xfd":
            stream = zstd.ZstdDecompressor().stream_reader(stream)
            closers.append(stream)
            continue
        break
    try:
        with io.TextIOWrapper(stream, encoding="utf-8") as reader:
            for line in reader:
                yield line
    finally:
        for closer in reversed(closers):
            closer.close()


class _PrefixedReader(io.RawIOBase):
    def __init__(self, prefix: bytes, raw: Any) -> None:
        self._prefix = io.BytesIO(prefix)
        self._raw = raw

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self._prefix.read() + self._raw.read()
        chunks = [self._prefix.read(size)]
        remaining = size - len(chunks[0])
        if remaining > 0:
            chunks.append(self._raw.read(remaining))
        return b"".join(chunks)

    def readinto(self, b: bytearray) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n


def _iter_hf_dataset(source: PreprocessSourceConfig) -> Iterator[RawDocument]:
    _configure_hf_environment()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for hf_dataset preprocessing") from exc
    if not source.hf_name:
        raise ValueError("hf_dataset source requires hf_name")
    kwargs: dict[str, Any] = {"split": source.hf_split, "streaming": source.hf_streaming}
    if source.hf_config:
        dataset = load_dataset(source.hf_name, source.hf_config, **kwargs)
    else:
        dataset = load_dataset(source.hf_name, **kwargs)
    for i, row in enumerate(dataset):
        doc = _doc_from_mapping(source, dict(row), i)
        if doc is not None:
            yield doc


def _configure_hf_environment() -> None:
    project_root = Path(__file__).resolve().parents[3]
    hf_cache = project_root / "hf_cache"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))


def _iter_remote_parquet(
    source: PreprocessSourceConfig,
    *,
    resume_state: dict[str, Any] | None = None,
) -> Iterator[RawDocument]:
    _configure_hf_environment()
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for remote_parquet preprocessing") from exc
    count = 0
    proxies = _proxies_from_env()
    download_config = DownloadConfig(proxies=proxies, max_retries=3) if proxies else DownloadConfig(max_retries=3)
    remote_state = resume_state or {}
    completed_urls = set(remote_state.get("completed_urls", []))
    current_url = remote_state.get("current_url")
    current_url_seen = int(remote_state.get("current_url_seen", 0))
    for url in _remote_parquet_urls(source):
        if url in completed_urls:
            continue
        row_skip = current_url_seen if current_url == url else 0
        dataset = load_dataset(
            "parquet",
            data_files=url,
            split="train",
            streaming=True,
            download_config=download_config,
        )
        for row_index, row in enumerate(dataset):
            if row_index < row_skip:
                continue
            doc = _doc_from_mapping(source, dict(row), count)
            if doc is None:
                continue
            doc.metadata.update({"remote_url": url, "remote_row_index": row_index, "parser": "remote_parquet"})
            yield doc
            count += 1


def _remote_parquet_urls(source: PreprocessSourceConfig) -> list[str]:
    return _remote_file_urls(source, source_type="remote_parquet", default_patterns=["**/*.parquet", "*.parquet"])


def _remote_jsonl_urls(source: PreprocessSourceConfig) -> list[str]:
    return _remote_file_urls(
        source,
        source_type="remote_jsonl",
        default_patterns=["**/*.jsonl", "*.jsonl", "**/*.jsonl.gz", "*.jsonl.gz", "**/*.json.gz", "*.json.gz"],
    )


def _remote_file_urls(source: PreprocessSourceConfig, *, source_type: str, default_patterns: list[str]) -> list[str]:
    urls = list(source.urls)
    if source.url_list_path:
        urls.extend(line.strip() for line in source.url_list_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if source.url_list_url:
        with requests.get(source.url_list_url, timeout=60, proxies=_proxies_from_env()) as response:
            response.raise_for_status()
            urls.extend(line.strip() for line in response.text.splitlines() if line.strip())
    hf_urls: list[str] = []
    if source.hf_repo_id:
        hf_urls = _hf_repo_file_urls(source, default_patterns=default_patterns)
        urls.extend(hf_urls)
    if not urls:
        if not (source.urls or source.url_list_path or source.url_list_url or source.hf_repo_id):
            raise ValueError(f"{source_type} source {source.name} requires urls, url_list_path, url_list_url, or hf_repo_id")
        if source.hf_repo_id and not hf_urls:
            patterns = source.hf_include_patterns or default_patterns
            raise ValueError(
                f"{source_type} source {source.name} found no files in HF repo "
                f"{source.hf_repo_id} matching patterns {patterns}; use hf_dataset, "
                "explicit urls/url_list_path, or adjust hf_include_patterns"
            )
        raise ValueError(f"{source_type} source {source.name} resolved no URLs from urls/url_list_path")
    return urls


def _hf_repo_parquet_urls(source: PreprocessSourceConfig) -> list[str]:
    return _hf_repo_file_urls(source, default_patterns=["**/*.parquet", "*.parquet"])


def _hf_repo_file_urls(source: PreprocessSourceConfig, *, default_patterns: list[str]) -> list[str]:
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    api_url = f"{endpoint}/api/datasets/{source.hf_repo_id}"
    response = requests.get(api_url, timeout=60, proxies=_proxies_from_env())
    response.raise_for_status()
    siblings = response.json().get("siblings", [])
    patterns = source.hf_include_patterns or default_patterns
    files = [
        item["rfilename"]
        for item in siblings
        if item.get("rfilename") and any(fnmatch(item["rfilename"], pattern) for pattern in patterns)
    ]
    return [
        f"{endpoint}/datasets/{source.hf_repo_id}/resolve/{source.hf_revision}/{quote(filename, safe='/')}"
        for filename in sorted(files)
    ]


def _proxies_from_env() -> dict[str, str] | None:
    http = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    proxies = {}
    if http:
        proxies["http"] = http
    if https:
        proxies["https"] = https
    return proxies or None


def _iter_html(source: PreprocessSourceConfig) -> Iterator[RawDocument]:
    count = 0
    for path in expand_paths(source.paths):
        if path.suffix.lower() not in {".html", ".htm", ".xhtml"}:
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        tree = html.fromstring(raw)
        for bad in tree.xpath("//script|//style|//noscript|//nav|//footer|//header|//aside|//form"):
            parent = bad.getparent()
            if parent is not None:
                parent.remove(bad)
        title = " ".join(tree.xpath("//title/text()")).strip()
        text = tree.text_content()
        yield RawDocument(
            id=f"{source.name}/{path.stem}",
            text=text,
            source=source.name,
            domain=source.domain,
            language=source.language,
            metadata={"path": str(path), "title": title, "license": source.license, "parser": "html"},
        )
        count += 1


def _iter_text(source: PreprocessSourceConfig) -> Iterator[RawDocument]:
    count = 0
    for path in expand_paths(source.paths):
        if path.stat().st_size > source.max_file_bytes or is_probably_binary(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        yield RawDocument(
            id=f"{source.name}/{path.stem}",
            text=text,
            source=source.name,
            domain=source.domain,
            language=source.language,
            metadata={"path": str(path), "license": source.license, "parser": "text"},
        )
        count += 1


def _iter_git(source: PreprocessSourceConfig) -> Iterator[RawDocument]:
    include = {ext.lower() for ext in source.include_extensions}
    exclude = set(source.exclude_dirs)
    count = 0
    for root in source.paths:
        root = root.resolve()
        files = sorted(p for p in root.rglob("*") if p.is_file())
        for path in files:
            rel_parts = path.relative_to(root).parts
            if any(part in exclude for part in rel_parts):
                continue
            if path.suffix.lower() not in include:
                continue
            if path.stat().st_size > source.max_file_bytes or is_probably_binary(path):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            rel = str(path.relative_to(root))
            yield RawDocument(
                id=f"{source.name}/{rel}",
                text=text,
                source=source.name,
                domain=source.domain,
                language=source.language,
                metadata={"repo": str(root), "path": rel, "ext": path.suffix.lower(), "license": source.license, "parser": "git"},
            )
            count += 1


def _iter_wiki_xml(source: PreprocessSourceConfig) -> Iterator[RawDocument]:
    count = 0
    for path in expand_paths(source.paths):
        context = ET.iterparse(path, events=("end",))
        for _event, elem in context:
            if _strip_ns(elem.tag) != "page":
                continue
            title = _find_text(elem, "title") or ""
            ns = _find_text(elem, "ns") or "0"
            page_id = _find_text(elem, "id") or str(count)
            text = _find_text(elem, "text") or ""
            elem.clear()
            if ns != "0" or not text.strip():
                continue
            yield RawDocument(
                id=f"{source.name}/{page_id}",
                text=_clean_wiki_markup(text),
                source=source.name,
                domain=source.domain,
                language=source.language,
                metadata={"title": title, "path": str(path), "license": source.license, "parser": "wiki_xml"},
            )
            count += 1


def _iter_pdf(source: PreprocessSourceConfig) -> Iterator[RawDocument]:
    try:
        import pypdf  # type: ignore
    except ImportError as exc:
        try:
            import PyPDF2 as pypdf  # type: ignore
        except ImportError:
            raise RuntimeError("PDF preprocessing needs pypdf/PyPDF2 or a future MinerU parser backend") from exc
    count = 0
    for path in expand_paths(source.paths):
        if path.suffix.lower() != ".pdf":
            continue
        reader = pypdf.PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        yield RawDocument(
            id=f"{source.name}/{path.stem}",
            text=text,
            source=source.name,
            domain=source.domain,
            language=source.language,
            metadata={"path": str(path), "license": source.license, "parser": "pdf_fallback"},
        )
        count += 1


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_text(elem: ET.Element, name: str) -> str | None:
    for child in elem.iter():
        if _strip_ns(child.tag) == name:
            return child.text
    return None


def _clean_wiki_markup(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\[\[File:[^\]]+\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    return text
