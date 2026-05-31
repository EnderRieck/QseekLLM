# llmTrain

Phase 1 implements the reproducible data and tokenizer foundation described in `DESIGN.md`.

## Environment

Use the existing `llmtrain` conda environment. Do not reinstall or upgrade Torch. Phase 1 uses already-present lightweight dependencies: Pydantic v2, PyYAML, pyarrow, sentencepiece, torch, and pytest.

## Minimal Phase 1 Run

From this directory:

```bash
python tools/stream_preprocess.py --config configs/preprocess/stage0_stream_preprocess.yaml
python tools/build_manifest.py --config configs/train/stage0_data_tokenizer.yaml
python tools/validate_manifest.py --config configs/train/stage0_data_tokenizer.yaml
python tools/sample_tokenizer_corpus.py --config configs/train/stage0_data_tokenizer.yaml
python tools/train_tokenizer.py --config configs/train/stage0_data_tokenizer.yaml
python tools/inspect_tokenizer.py --config configs/train/stage0_data_tokenizer.yaml
python tools/prepare_pretrain_stream.py --config configs/train/stage0_data_tokenizer.yaml
```

`prepare_pretrain_stream.py` 当前用于验证训练时的数据流闭环：读 manifest、跑 pipeline、tokenize、pack、checkpoint metadata。正式 trainer 接上后，这条链路会作为异步 producer/consumer 的数据入口。

Each command accepts repeatable overrides:

```bash
python tools/prepare_pretrain_stream.py \
  --config configs/train/stage0_data_tokenizer.yaml \
  --override data.packing.seq_len=64 \
  --override trainer.max_tokens=512
```

Run tests:

```bash
pytest
```

## Phase 1 Outputs

The sample config writes to `runs/stage0_data_tokenizer/`:

- `config.resolved.yaml`
- `tokenizer_corpus.txt`
- `tokenizer/tokenizer.metadata.json`
- `tokenizer_inspection.json`
- `metrics.jsonl`
- `events.jsonl`
- `heartbeat.json`
- `checkpoints/phase1_meta/meta.pt`
- `checkpoints/phase1_meta/_SUCCESS`

The example manifest is written under `examples/data/` as:

- `manifest.jsonl`
- `manifest.meta.json`

## Streaming Raw Preprocessing

`tools/stream_preprocess.py` is the raw-data layer before manifest/tokenizer/packing. It streams inputs, parses documents, cleans text, removes duplicates, scores quality, writes rolling unified shards, and emits a manifest.

Supported source types:

- `jsonl`: existing open datasets with configurable `text_field`, `id_field`, and `metadata_fields`
- `parquet`: existing open datasets stored as Parquet
- `hf_dataset`: HuggingFace datasets with `streaming=True`
- `html`: local HTML files, extracted with `lxml`
- `text`: plain text files
- `wiki_xml`: MediaWiki XML dump streaming parser
- `git`: local repository/file tree code extraction
- `pdf`: interface plus optional `pypdf`/`PyPDF2` fallback; MinerU should plug into this parser boundary later

Example:

```bash
python tools/stream_preprocess.py --config configs/preprocess/stage0_stream_preprocess.yaml
```

The default preprocess config follows the candidate sources listed in `DESIGN.md`:

```bash
python tools/stream_preprocess.py --config configs/preprocess/stage0_stream_preprocess.yaml
```

Most public dataset entries use `type: hf_dataset` with `hf_streaming: true`. HuggingFace streaming defaults to `HF_ENDPOINT=https://hf-mirror.com` and stores cache under `hf_cache/` in this project unless those environment variables are already set.

For large parquet-backed datasets, prefer `type: remote_parquet`. It uses explicit shard URLs or discovers parquet files from a HuggingFace dataset repo, then processes one remote parquet shard stream at a time. This is the default for FineWeb-Edu in `stage0_stream_preprocess.yaml` because it gives better control over long-running jobs and recovery than expanding an entire dataset through `load_dataset`.

`hf_dataset` is still supported for official dataset-script access and smoke tests. Resume for this mode is conservative: it reopens the stream and skips the previously seen records.

Resume a preprocessing run:

```bash
python tools/stream_preprocess.py --config configs/preprocess/stage0_stream_preprocess.yaml --resume
```

By default it writes under `runs/stream_preprocess/`. Resume state is stored in:

- `preprocess.state.json`
- `dedup_state/`
- existing finalized shards under `shards/`

On resume, completed sources are skipped. `hf_dataset` resumes conservatively from its recorded `seen` offset by replaying and skipping records. `remote_parquet` stores finer state:

- `completed_urls`
- `current_url`
- `current_url_seen`

This lets recovery skip already completed parquet URLs and replay only within the current parquet file up to the saved row offset.

This writes:

- `runs/stage0_stream_preprocess/shards/*.jsonl`
- `runs/stage0_stream_preprocess/manifest.jsonl`
- `runs/stage0_stream_preprocess/manifest.meta.json`
- `runs/stage0_stream_preprocess/stats.json`
- `runs/stage0_stream_preprocess/rejected.jsonl`
- optional dedup state under `runs/stage0_stream_preprocess/dedup_state/`

The cleaned shard records use the same unified schema as the rest of the framework, so the output manifest can be fed into tokenizer sampling and packing by setting `data.manifest_path`.

Dedup is layered:

- exact dedup uses normalized-text sha256
- near dedup uses an in-repo 64-bit SimHash implementation
- dedup state can be persisted; set `load_existing_state: true` only when intentionally resuming or extending a previous dedup run

Quality scoring is heuristic for now. It records `quality_score` and `quality_signals` in metadata, and can later be replaced by a model scorer behind the same interface.

## Scope

Implemented in Phase 1:

- streaming raw preprocessing for open datasets, HTML, Wiki XML, local git/code, and parser interfaces
- Pydantic v2 schemas with `extra: forbid`
- YAML `extends` merge and `--override`
- JSONL and Parquet manifest build/validation with sha256 checks
- deterministic rank/worker shard assignment
- record validation, normalization, filtering
- token-weighted source mixer
- concatenated packing with `document_ids`
- train-time async producer/consumer data path
- SentencePiece BPE sampling, training, inspection, and adapter
- JSONL metrics/events, heartbeat, and metadata checkpoint

Non-goals remain model training, FSDP/ZeRO, DCP model/optimizer checkpointing, and final 150K tokenizer quality validation.
