# Agent Handoff Notes

## Project Context

This repo is being used to build and run a large-scale preprocessing pipeline for LLM training data. The current focus is `tools/stream_preprocess.py` and the preprocessing implementation under `src/llmtrain/preprocessing/`.

Primary user goals:
- Parse and clean local/raw web, PDF, git/code, wiki, paper/math datasets.
- Stream wherever reasonable, but local downloaded datasets are now used for scale.
- Run global exact dedup and simhash near-dedup.
- Keep the pipeline resumable across interrupts.
- Support high worker counts for the heavy preprocessing and dedup stages.

## Important Paths

- Repo root: `/mnt/paper2any/ziyi/llmTrain`
- Main config: `configs/preprocess/stage0_stream_preprocess.yaml`
- Main CLI: `tools/stream_preprocess.py`
- Parallel preprocessing/dedup code: `src/llmtrain/preprocessing/parallel.py`
- Preprocess config model: `src/llmtrain/preprocessing/config.py`
- Output run dir: `/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/stream_preprocess`
- Local datasets symlink: `/mnt/paper2any/ziyi/llmTrain/datasets -> /mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/datasets`

## Current Run Status

As of this handoff, the active preprocessing command is running:

```bash
LLMTRAIN_DEDUP_INDEX_WORKERS=64 \
python tools/stream_preprocess.py \
  --config configs/preprocess/stage0_stream_preprocess.yaml \
  --resume
```

Observed current process state:
- Parent PID: `2484389`
- 64 child worker processes are active.
- `LLMTRAIN_DEDUP_INDEX_WORKERS=64` is present in the parent process environment.
- `preprocess.num_workers=64` in the resolved config.
- CPU total across children was around `5300%`, so 64-way parallelism is active.

Current stage:
- Candidate generation is complete.
- The run is in `dedup index`.
- `exact_drops`, `simhash_drops`, and `materialize` had not started at the time of this note.

Current artifacts:
- Candidate dir: `/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/stream_preprocess/candidates`
- Candidate size: about `7.6T`
- Candidate done markers: `3903`
- Candidate zstd files: `3166`
- Dedup work dir: `/mnt/DataFlow/lz/proj/agentgroup/ziyi/llmTrain/runs/stream_preprocess/dedup_work`
- Dedup work size had grown to about `384G`, showing index writing is progressing.

Recent dedup index speed sample:
- Total index write growth was about `21.9G` over 30 seconds.
- Approx total write throughput: about `748 MB/s`.
- Breakdown from that sample:
  - exact index: about `148 MB/s`
  - simhash index: about `600 MB/s`

## Resume and Cleanup Behavior

The pipeline was refactored to make parallel preprocessing and dedup resumable:
- Candidate tasks use `.done.json` / `.stats.json` markers.
- Dedup index jobs use result/done markers under `dedup_work/index_done`.
- Exact dedup, simhash dedup, and materialize/write stages also use resumable markers.
- `write deduped` is parallel and resumable.

`cleanup_dedup_work` exists in `PreprocessConfig` and defaults to `true`.
- If a full run finishes successfully, `dedup_work` is deleted by default.
- For debugging or preserving dedup intermediates, run with:

```bash
python tools/stream_preprocess.py \
  --config configs/preprocess/stage0_stream_preprocess.yaml \
  --resume \
  --no-cleanup-dedup-work
```

The CLI also supports:

```bash
--cleanup-dedup-work
--no-cleanup-dedup-work
```

These override the YAML config for the run.

## Worker Controls

There are two worker controls:

```bash
--override preprocess.num_workers=64
```

Controls:
- candidate worker count
- exact dedup reduce workers
- simhash dedup reduce workers
- materialize/write workers

```bash
LLMTRAIN_DEDUP_INDEX_WORKERS=64
```

Controls only:
- `dedup index` worker count

If changing from 64 back to 32, keep the same config/output dir and use `--resume`:

```bash
LLMTRAIN_DEDUP_INDEX_WORKERS=32 \
python tools/stream_preprocess.py \
  --config configs/preprocess/stage0_stream_preprocess.yaml \
  --override preprocess.num_workers=32 \
  --resume
```

Do not delete the output dir if resuming.

## Recent Code/Test State

Relevant changes already made:
- Added resumable parallel dedup stages in `parallel.py`.
- Added parallel materialize/write stage.
- Added `cleanup_dedup_work` to `PreprocessConfig`.
- Added CLI override for `--cleanup-dedup-work` / `--no-cleanup-dedup-work`.
- Improved process-pool shutdown behavior earlier in the session.

Test commands run successfully:

```bash
PYTHONPATH=.:src pytest -q tests/test_preprocessing.py tests/test_config.py
```

Result:
- `22 passed, 1 warning`

Resume-focused tests also passed:

```bash
PYTHONPATH=.:src pytest -q tests/test_preprocessing.py -k "resume"
```

Result:
- `4 passed`

## Notes for Next Agent

- The user may ask to monitor current dedup speed. Avoid repeatedly scanning the entire `dedup_work` tree too frequently because it is already hundreds of GB with many files.
- Prefer checking process count/CPU and small targeted marker directories first.
- If measuring throughput, take spaced snapshots of `exact_index_parts` and `simhash_index_parts`, but do not do it too often.
- The current bottleneck during `dedup index` appears to be heavy index writing/IO, not lack of worker processes.
- `/mnt/paper2any` is nearly full; output should remain under `/mnt/DataFlow`.
- `/mnt/DataFlow` had about `31T` free during the last check.
