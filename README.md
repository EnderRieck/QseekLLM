# LLMTrain

A scalable, production-ready framework for pretraining large language models with FSDP/DDP, streaming data pipeline, and comprehensive observability.

## Features

- **Distributed Training**: FSDP/DDP support with gradient accumulation and mixed precision
- **Streaming Data Pipeline**: Memory-efficient parquet-based data loading with async tokenization
- **Flexible Scheduling**: WSD (Warmup-Stable-Decay), cosine, and constant learning rate schedules
- **In-Training Validation**: Automatic per-source perplexity evaluation during training
- **Checkpoint Management**: DCP-based distributed checkpointing with milestone and interval saves
- **Observability**: JSONL metrics, heartbeat monitoring, and optional W&B/TensorBoard integration
- **Quality Filtering**: Optional text quality classification for data curation
- **Evaluation Harness**: Integration with lm-evaluation-harness for downstream task evaluation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/llmTrain.git
cd llmTrain

# Install dependencies
pip install -e .

# For evaluation support
pip install -e '.[eval]'

# Configure environment variables
cp .env.example .env
# Edit .env and set DATA_DIR to your data directory
```

**Important**: Before running any training, copy `.env.example` to `.env` and update `DATA_DIR` to point to your data directory:

```bash
# .env
DATA_DIR=/path/to/your/data
```

The framework will automatically load environment variables from `.env` and expand them in config files (e.g., `${DATA_DIR}/train_340b/manifest.jsonl`).

### Training

```bash
# Single-node training (8 GPUs)
python run.py train --config configs/train/stage1_general_300m_v2_wsd_30b.yaml

# Multi-node training
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=... \
    run.py train --config configs/train/your_config.yaml
```

### Evaluation

```bash
# Evaluate a checkpoint
python run.py eval --config configs/eval/default_300m_v2.yaml \
    --checkpoint runs/your_run/checkpoints/milestone_030000000000

# Batch evaluation of multiple checkpoints
bash scripts/eval_checkpoints.sh
```

## Project Structure

```
llmTrain/
├── src/llmtrain/
│   ├── models/          # Model architectures (Qwen-like, etc.)
│   ├── data/            # Data pipeline (reader, mixer, packer)
│   ├── training/        # Trainer, optimizer, scheduler
│   ├── evaluation/      # Validation callback, eval harness integration
│   ├── checkpointing/   # DCP checkpoint save/load
│   ├── distributed/     # FSDP/DDP setup
│   ├── tokenizer/       # Tokenizer wrappers
│   └── observability/   # Metrics logging
├── configs/
│   ├── train/           # Training configurations
│   ├── eval/            # Evaluation configurations
│   ├── model/           # Model architecture configs
│   ├── data/            # Data mixture configs
│   └── tokenizer/       # Tokenizer configs
├── scripts/             # Utility scripts
├── tools/               # Standalone tools (validation, preprocessing)
└── run.py               # Main entry point
```

## Configuration

Training is configured via YAML files with inheritance support:

```yaml
extends:
  - ../data/mixture_50b.yaml
  - ../tokenizer/hf_byte_bpe_150k.yaml
  - ../model/qwen_like_1_7b.yaml

run:
  name: my_training_run
  output_dir: runs/my_run
  seed: 42

trainer:
  micro_batch_size: 4
  global_batch_size: 512
  max_tokens: 50_000_000_000
  optimizer:
    type: adamw
    lr: 3.0e-4
    betas: [0.9, 0.95]
    weight_decay: 0.1
  scheduler:
    type: wsd
    warmup_tokens: 200_000_000
    decay_tokens: 10_000_000_000
    min_lr_ratio: 0.01
  precision: bf16

validation:
  enabled: true
  val_manifest: path/to/val_manifest.jsonl
  interval_tokens: 1_000_000_000
```

## Data Format

Training data is stored as parquet shards with a JSONL manifest:

**Manifest entry:**
```json
{
  "id": "shard_001",
  "uri": "/path/to/shard.parquet",
  "source": "wikipedia",
  "domain": "en",
  "language": "en",
  "num_records": 10000,
  "estimated_tokens": 5000000,
  "bytes": 50000000,
  "sha256": "...",
  "weight": 1.0
}
```

**Parquet schema:**
```
id: string
text: string
source: string
domain: string
language: string
metadata: string (JSON)
```

## Monitoring

Training metrics are logged to JSONL files:

- `metrics.jsonl`: Training loss, learning rate, throughput
- `val_metrics.jsonl`: Validation perplexity per source
- `data_metrics.jsonl`: Data loading statistics
- `events.jsonl`: Training events (start, checkpoint, validation)
- `heartbeat.json`: Real-time training status

## Advanced Features

### In-Training Validation

Automatically evaluate on held-out data during training:

```yaml
validation:
  enabled: true
  val_manifest: path/to/val_manifest.jsonl
  interval_tokens: 1_000_000_000  # Eval every 1B tokens
  max_tokens_per_source: 2_000_000
  run_at_start: true
```

### Quality Filtering

Filter low-quality documents during data loading:

```yaml
data:
  quality_filter:
    enabled: true
    model_path: models/quality_classifier.bin
    threshold: 0.5
    apply_to_sources: [common_crawl]
```

### Checkpoint Management

```yaml
checkpoint:
  format: auto  # DCP or torch
  save_interval_minutes: 60
  milestone_interval_tokens: 2_000_000_000  # Save every 2B tokens
  keep_latest: 2  # Keep last 2 interval checkpoints
```

## Model Architectures

Currently supported:
- **Qwen-like**: Transformer with GQA, RoPE, SwiGLU (300M, 700M, 1.7B configs provided)

Adding new architectures:
1. Implement model in `src/llmtrain/models/your_model.py`
2. Register in `src/llmtrain/models/__init__.py`
3. Add config in `configs/model/your_model.yaml`

## Performance Tips

- **Batch size**: Use 2-4M tokens/step for models >1B params
- **Activation checkpointing**: Enable for large models to save memory
- **Async tokenization**: Set `async_tokenization: true` for faster data loading
- **FSDP sharding**: Use `shard_grad_op` for best memory/speed tradeoff
- **Mixed precision**: bf16 recommended for training stability

## Development

See `DESIGN.md` and `DEVELOPMENT.md` for architecture details and Phase 1 implementation notes.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with PyTorch, HuggingFace Transformers, and lm-evaluation-harness
- Inspired by best practices from Llama, Qwen, and other open-source LLM projects
