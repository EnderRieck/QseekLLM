# LLMTrain

一个可扩展的、生产就绪的大语言模型预训练框架，支持 FSDP/DDP、流式数据管道和全面的可观测性。

## 特性

- **分布式训练**：支持 FSDP/DDP，具备梯度累积和混合精度训练
- **流式数据管道**：基于 parquet 的内存高效数据加载，支持异步分词
- **灵活的调度策略**：WSD（预热-稳定-衰减）、余弦和恒定学习率调度
- **训练中验证**：训练期间自动按数据源评估困惑度
- **检查点管理**：基于 DCP 的分布式检查点，支持里程碑和间隔保存
- **可观测性**：JSONL 指标、心跳监控，可选 W&B/TensorBoard 集成
- **质量过滤**：可选的文本质量分类，用于数据筛选
- **评估工具**：集成 lm-evaluation-harness 进行下游任务评估

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/llmTrain.git
cd llmTrain

# 安装依赖
pip install -e .

# 安装评估支持
pip install -e '.[eval]'

# 配置环境变量
cp .env.example .env
# 编辑 .env 并设置 DATA_DIR 为你的数据目录
```

**重要提示**：在运行任何训练之前，请将 `.env.example` 复制为 `.env` 并更新 `DATA_DIR` 指向你的数据目录：

```bash
# .env
DATA_DIR=/path/to/your/data
```

框架会自动从 `.env` 加载环境变量，并在配置文件中展开它们（例如 `${DATA_DIR}/train_340b/manifest.jsonl`）。

### 训练

```bash
# 单节点训练（8 个 GPU）
python run.py train --config configs/train/stage1_general_300m_v2_wsd_30b.yaml

# 多节点训练
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=... \
    run.py train --config configs/train/your_config.yaml
```

### 评估

```bash
# 评估一个检查点
python run.py eval --config configs/eval/default_300m_v2.yaml \
    --checkpoint runs/your_run/checkpoints/milestone_030000000000

# 批量评估多个检查点
bash scripts/eval_checkpoints.sh
```

## 项目结构

```
llmTrain/
├── src/llmtrain/
│   ├── models/          # 模型架构（Qwen-like 等）
│   ├── data/            # 数据管道（reader、mixer、packer）
│   ├── training/        # Trainer、优化器、调度器
│   ├── evaluation/      # 验证回调、评估工具集成
│   ├── checkpointing/   # DCP 检查点保存/加载
│   ├── distributed/     # FSDP/DDP 设置
│   ├── tokenizer/       # 分词器封装
│   └── observability/   # 指标日志
├── configs/
│   ├── train/           # 训练配置
│   ├── eval/            # 评估配置
│   ├── model/           # 模型架构配置
│   ├── data/            # 数据混合配置
│   └── tokenizer/       # 分词器配置
├── scripts/             # 实用脚本
├── tools/               # 独立工具（验证、预处理）
└── run.py               # 主入口
```

## 配置

训练通过支持继承的 YAML 文件进行配置：

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

## 数据格式

训练数据以 parquet 分片存储，并配有 JSONL 清单文件：

**清单条目：**
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

**Parquet 模式：**
```
id: string
text: string
source: string
domain: string
language: string
metadata: string (JSON)
```

## 监控

训练指标记录到 JSONL 文件：

- `metrics.jsonl`：训练损失、学习率、吞吐量
- `val_metrics.jsonl`：按数据源的验证困惑度
- `data_metrics.jsonl`：数据加载统计
- `events.jsonl`：训练事件（开始、检查点、验证）
- `heartbeat.json`：实时训练状态

## 高级特性

### 训练中验证

在训练期间自动在留出数据上进行评估：

```yaml
validation:
  enabled: true
  val_manifest: path/to/val_manifest.jsonl
  interval_tokens: 1_000_000_000  # 每 1B tokens 评估一次
  max_tokens_per_source: 2_000_000
  run_at_start: true
```

### 质量过滤

在数据加载期间过滤低质量文档：

```yaml
data:
  quality_filter:
    enabled: true
    model_path: models/quality_classifier.bin
    threshold: 0.5
    apply_to_sources: [common_crawl]
```

### 检查点管理

```yaml
checkpoint:
  format: auto  # DCP 或 torch
  save_interval_minutes: 60
  milestone_interval_tokens: 2_000_000_000  # 每 2B tokens 保存一次
  keep_latest: 2  # 保留最近 2 个间隔检查点
```

## 模型架构

当前支持：
- **Qwen-like**：带有 GQA、RoPE、SwiGLU 的 Transformer（提供 300M、700M、1.7B 配置）

添加新架构：
1. 在 `src/llmtrain/models/your_model.py` 中实现模型
2. 在 `src/llmtrain/models/__init__.py` 中注册
3. 在 `configs/model/your_model.yaml` 中添加配置

## 性能建议

- **批量大小**：对于 >1B 参数的模型，使用 2-4M tokens/step
- **激活检查点**：对大模型启用以节省内存
- **异步分词**：设置 `async_tokenization: true` 以加快数据加载
- **FSDP 分片**：使用 `shard_grad_op` 以获得最佳内存/速度权衡
- **混合精度**：推荐使用 bf16 以保证训练稳定性

## 开发

有关架构细节和第一阶段实现说明，请参阅 `DESIGN.md` 和 `DEVELOPMENT.md`。

## 许可证

MIT License - 详见 LICENSE 文件

## 致谢

- 基于 PyTorch、HuggingFace Transformers 和 lm-evaluation-harness 构建
- 受 Llama、Qwen 和其他开源 LLM 项目最佳实践的启发
