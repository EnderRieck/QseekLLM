# llmTrain Design

## 1. Background

本项目目标是从头训练一个中文能力较强、兼顾英文、代码、数学、论文和百科知识的大语言模型。整体计划从小模型验证开始，逐步推进到 1.7B 规模的通用预训练、推理方向继续预训练，以及资源允许情况下的长上下文继续预训练。目前只做预训练阶段，但后续后训练会考虑提高其推理能力

训练框架需要提前设计清楚各模块边界，避免数据处理、tokenizer、模型结构、checkpoint、分布式训练在后期耦合成难以维护的脚本集合。当前阶段优先落地数据处理和 tokenizer 相关闭环，同时为后续手写 PyTorch 模型、训练主循环和分布式训练预留稳定接口。

## 2. User Requirements

### 2.1 Training Plan

训练阶段规划如下：

1. 阶段 0A：数据管线验证
   - 验证 manifest、流式读取、数据混合、过滤、tokenization、packing 和可恢复状态。
   - 完成数据清洗并保留索引，支撑后续流式数据加载

2. 阶段 0B：Tokenizer 训练与验证
   - 从清洗后的多源数据中采样约 200GB 文本。
   - 使用 SentencePiece BPE 训练 150K 词表，规模对齐 Qwen 系列。
   - 检查中英文、代码、数学、论文等不同域的压缩率和 token 分布。

3. 阶段 0C：300M / 700M 小模型验证
   - 300M / 700M 正式训练配置预留 `trainer.max_tokens` 作为总训练量上限，避免默认把全量数据集跑完。
   - 验证数据配比、tokenizer、模型实现和训练稳定性。

4. 阶段 1：1.7B General Pretraining
   - 数据域：中文、英文、代码、数学、论文、百科。
   - 上下文长度：`seq_len=4096`。
   - 起步训练量：20B-50B tokens。

5. 阶段 2：Reasoning Continual Pretraining
   - 提高 STEM、代码、数学、推理、合成数据比例。
   - 训练量：2B-10B tokens。
   - 学习率衰减更快。

6. 阶段 3：Long-context Continual Pretraining
   - 如果资源允许，使用论文、书籍、长百科、长代码文件。
   - 上下文长度逐步扩展：8192 -> 16384 -> 32768。

### 2.2 Data Plan

目标总数据量约 50B tokens，初步配比：

| Data domain | Target ratio |
| --- | --- |
| 中文通用 | 35%-45% |
| 英文通用 | 15%-25% |
| 代码 | 10%-15% |
| 数学推理 | 5%-10% |
| 论文 | 10%-15% |
| 百科 | 5%-10% |

候选数据来源：

| Domain | Sources |
| --- | --- |
| 中文 | CCI3.0-HQ, ChineseWebText 2.0 |
| 英文 | FineWeb-Edu, Dolma |
| 代码 | The Stack v2, CodeNet |
（由于TheStackV2数据量过大，且需要从AWS下载，比较麻烦，因此采用TheStackV1数据集来下载）
| 数学推理 | OpenWebMath, Proof-Pile-2, DeepSeekMath Corpus 思路参考, 从网页数据中额外提取数学数据 |
| 论文 | 自主采样，目标约 5B tokens / 50 万篇 |
| 百科 | 自主采样 |

论文和百科等自建数据统一使用 JSON schema：

```json
{
  "id": "source/xxx",
  "text": "...",
  "source": "arxiv",
  "domain": "paper",
  "language": "en",
  "metadata": {
    "title": "...",
    "year": 2024,
    "license": "...",
    "quality_score": 0.87
  }
}
```

数据加载应采用流式方式，不要求把 50B token 级别的数据完整落盘到本地训练目录。

### 2.3 Tokenizer Requirements

项目决定自训练 tokenizer，而不是直接复用外部模型 tokenizer。

核心要求：

- 从清洗后的训练数据中采样约 200GB tokenizer corpus。
- 使用 SentencePiece BPE。
- 词表大小：150K。
- 目标是兼顾中文、英文、代码、数学和论文文本，并尽量对齐 Qwen 系列的大词表策略。
- tokenizer 训练后需要提供 inspection 工具，评估不同数据域的压缩率、unk/byte fallback 使用情况、长文本稳定性和特殊 token 设计。

### 2.4 Model Requirements

模型结构尽量采用成熟 LLM 的稳定设计，降低收敛风险。1.7B 模型希望与 Qwen3 1.7B 的结构风格保持一致。

实现策略：

- 使用 PyTorch 基础算子手写 TransformerBlock。
- 手写 RMSNorm、RoPE、SwiGLU、Attention、MLP 和 Block。
- 不手写底层 CUDA kernel；在不显著影响性能的前提下，优先使用 PyTorch 高性能原语，例如 `torch.nn.functional.scaled_dot_product_attention`。
- 模型实现应支持 300M / 700M / 1.7B 不同规模，通过配置切换。
- 正式训练入口必须允许通过 `trainer.max_tokens` 明确截断总训练量。

### 2.5 Environment Requirements

训练环境基于已有 `base` conda 环境克隆得到 `llmtrain` 环境，因为 `base` 中包含沐曦定制开发的 Torch 相关包。为了保留 CUDA/后端兼容性，后续依赖管理必须遵守：

- 不重装或升级 `torch`。
- 不安装会隐式替换 Torch、CUDA runtime 或设备后端的包。
- 新依赖优先选择轻量、纯 Python 或与当前 Torch 版本兼容的版本。
- 安装前先检查 `llmtrain` 环境中是否已经存在相关依赖。

Huggingface大数据集加载时不走本地7890端口代理，使用https://hf-mirror.com作为镜像站，另外数据集、模型缓存不放在系统盘默认位置，应放到项目目录中的hf_cache文件夹

## 3. Repository Layout

初步项目结构：

```text
llmTrain/
  DESIGN.md
  README.md
  pyproject.toml

  configs/
    data/
      mixture_50b.yaml
      tokenizer_sample_200gb.yaml
    tokenizer/
      hf_byte_bpe_150k.yaml
    model/
      qwen_like_300m.yaml
      qwen_like_700m.yaml
      qwen_like_1_7b.yaml
    train/
      stage0_smoke.yaml
      stage1_general.yaml
      stage2_reasoning_cpt.yaml
      stage3_long_context.yaml

  src/llmtrain/
    data/
    tokenizer/
    models/
    training/
    checkpointing/
    distributed/
    evaluation/
    observability/
    utils/

  tools/
    build_manifest.py
    sample_tokenizer_corpus.py
    train_tokenizer.py
    inspect_tokenizer.py
    prepare_pretrain_stream.py
    train.py

  examples/
    data/

  tests/
  docs/
```

## 4. Configuration System

配置系统目标：训练只在命令行指定一个 yaml 文件路径即可启动，其余通过 yaml 组合表达。CLI 参数保留作为调试 escape hatch，不是主力。

### 4.1 Selection

- **Pydantic v2** 做 schema 与类型校验，错误信息友好，支持 `extra: forbid` 防止字段拼写错误被静默忽略。
- **自写极简 yaml 合并器**（< 50 行），不引入 Hydra 与 OmegaConf。Hydra 的 defaults list / group 对小项目过重，OmegaConf 的 `${...}` interpolation 长期维护成本高。

### 4.2 CLI Surface

```bash
python tools/train.py --config configs/train/stage1_general.yaml
```

只两个参数：

- `--config <path>`：必需，唯一主力入口。
- `--override k=v`（可重复）：可选 escape hatch，用于扫参或临时调试，不进 yaml。

### 4.3 File Roles

`configs/` 下的 yaml 分两类，铁律：

| 角色 | 路径 | 含义 |
| --- | --- | --- |
| fragment | `configs/{data,tokenizer,model}/*.yaml` | 只描述自己那一段，不可直接训练（缺 trainer 等字段） |
| stage 入口 | `configs/train/*.yaml` | 含 `extends` 引用 fragment + 自己的 trainer/checkpoint 段，可直接训练 |

### 4.4 Composition via `extends`

stage 入口通过 `extends` 列出要拼装的 fragment：

```yaml
# configs/train/stage1_general.yaml
extends:
  - ../data/mixture_50b.yaml
  - ../tokenizer/hf_byte_bpe_150k.yaml
  - ../model/qwen_like_1_7b.yaml

run:
  name: stage1_general_1_7b
  output_dir: runs/stage1_general
  seed: 42

trainer:
  micro_batch_size: 1
  global_batch_size: 1024
  max_tokens: 50_000_000_000
  optimizer: {type: adamw, lr: 3.0e-4, betas: [0.9, 0.95], weight_decay: 0.1}
  scheduler: {type: cosine, warmup_tokens: 1_000_000_000, min_lr_ratio: 0.1}
  precision: bf16
  grad_clip: 1.0

checkpoint:
  save_interval_minutes: 30
  milestone_interval_tokens: 1_000_000_000
  keep_latest: 3

distributed:
  backend: fsdp
  activation_checkpointing: true
  fsdp_sharding_strategy: full_shard
  fsdp_mixed_precision: true
```

### 4.5 Merge Rules

仅五条，全部规则：

| 类型 | 行为 |
| --- | --- |
| dict | 递归合并，子字段覆盖父字段 |
| list | 整段替换，不深合并 |
| 标量 | 覆盖 |
| `extends` 列表 | 后者覆盖前者 |
| 当前文件 | 覆盖所有 `extends` |

CLI `--override` 在以上合并完成后最后一步 apply。

### 4.6 Schema

**分散组织**：每个实现模块在自己包下放一个 `config.py`，定义该模块的 Pydantic `*Config` schema，与实现就近。`utils/config.py` 只负责把它们组装为顶层 `Config` + 提供 yaml 合并器与 loader。

| Schema | 位置 |
| --- | --- |
| `RunConfig` | `src/llmtrain/utils/config.py`（无对应实现模块） |
| `DataConfig` | `src/llmtrain/data/config.py` |
| `TokenizerConfig` | `src/llmtrain/tokenizer/config.py` |
| `ModelConfig` | `src/llmtrain/models/config.py` |
| `TrainerConfig` | `src/llmtrain/training/config.py` |
| `CheckpointConfig` | `src/llmtrain/checkpointing/config.py` |
| `DistributedConfig` | `src/llmtrain/distributed/config.py` |
| `ObservabilityConfig` | `src/llmtrain/observability/config.py` |

顶层 `Config` 在 `utils/config.py` 组装：

```python
# src/llmtrain/utils/config.py
from llmtrain.data.config import DataConfig
from llmtrain.tokenizer.config import TokenizerConfig
from llmtrain.models.config import ModelConfig
from llmtrain.training.config import TrainerConfig
from llmtrain.checkpointing.config import CheckpointConfig
from llmtrain.distributed.config import DistributedConfig
from llmtrain.observability.config import ObservabilityConfig

class Config(BaseModel):
    run: RunConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    trainer: TrainerConfig
    checkpoint: CheckpointConfig
    distributed: DistributedConfig
    observability: ObservabilityConfig

    model_config = {"extra": "forbid"}
```

所有子 schema **必须**同样开启 `extra: forbid`，统一杜绝字段拼错被静默忽略。

### 4.7 Resolved Config Snapshot

合并、校验完成后的最终配置以两种形式持久化，互为冗余：

**A. Run 目录下的 yaml（人读首选）**

每次 trainer 启动时，由 rank 0 写入 `{run.output_dir}/config.resolved.yaml`：

```yaml
# schema_version: 1.0
# generated_at: 2026-05-01T14:30:22+08:00
# chain:
#   - configs/train/stage1_general.yaml         sha256:ab12...
#   - configs/data/mixture_50b.yaml             sha256:cd34...
#   - configs/tokenizer/hf_byte_bpe_150k.yaml  sha256:ef56...
#   - configs/model/qwen_like_1_7b.yaml         sha256:7890...

run:
  name: stage1_general_1_7b
  output_dir: runs/stage1_general
  seed: 42

data: { ... }
tokenizer: { ... }
model: { ... }
trainer: { ... }
checkpoint: { ... }
distributed: { ... }
```

冲突策略：

- 不存在 → 直接写。
- 已存在且**字节级相同** → 跳过。
- 已存在且不同（恢复时改了字段）→ 把旧文件重命名为 `config.resolved.<timestamp>.yaml` 保留，再写新内容到 `config.resolved.yaml`。绝不静默覆盖。

这份 yaml 仅用于人读、grep、diff，**训练不从此处加载**（避免被无意修改）。

**B. Checkpoint 内的 dict（机器恢复源）**

每个 checkpoint 同时保存：

```python
{
    "resolved": cfg.model_dump(),
    "chain": [
        {"path": "configs/train/stage1_general.yaml", "sha256": "..."},
        {"path": "configs/data/mixture_50b.yaml",     "sha256": "..."},
    ],
    "schema_version": "1.0",
}
```

恢复策略：

- 用 ckpt 内 `resolved` 直接重建 `Config`，**不重新读 yaml、不读 run 目录的 resolved.yaml**，避免文件被改动导致恢复行为漂移。
- 启动时 diff 当前命令行 yaml 解析结果与 ckpt 的 `resolved`，不一致仅警告，允许中途调 lr 等字段。
- `schema_version` 不一致拒绝 load。

### 4.8 Implementation Footprint

- `src/llmtrain/<module>/config.py`：定义 `<X>Config` schema（每个实现模块各 ~20–40 行）。
- `src/llmtrain/utils/config.py`：组装顶层 `Config` + yaml 合并器 + `load_config(path, overrides=[])` + `dump_resolved(cfg, output_dir)` + `RunConfig`，预期 ~80 行（schema 已分散，组合层很薄）。
- `tools/train.py`：argparse + `load_config` + `dump_resolved` + 启动 trainer，< 30 行。

## 5. Data Pipeline Design

数据管线是项目第一优先级。设计目标：训练主循环只消费标准 batch，对底层数据来源（本地 JSONL/Parquet、远程对象存储、论文、网页、代码）完全无感。

### 5.1 Responsibility Boundary

本框架的输入是已经清洗、去重、打分完成的 unified shard 集合。下列工作**不在本框架职责内**，由独立的离线 pipeline 完成：

- 跨 shard 的全局去重（MinHash-LSH 或 exact hash）。
- 文档级质量打分（heuristic 或分类器）。
- 原始格式（warc / pdf / latex / git）到 unified record 的转换。

框架内部仅做轻量 record 级处理（schema 校验、语言/长度/质量阈值过滤、字段归一化），不承担重型清洗。这条边界一旦写入设计就不放回来，避免数据模块逐渐膨胀成一个清洗 pipeline。

### 5.2 Data Flow

```text
unified shards (JSONL or Parquet)
  -> manifest (immutable, hashed, versioned)
  -> shard reader (deterministic split per rank/worker)
  -> record pipeline (validate -> normalize -> filter)
  -> weighted mixer (token-weighted, temperature sampling)
  -> async data producer (tokenize -> pack -> numpy batch queue)
  -> trainer consumer (torch batch -> forward/backward)
```

### 5.3 Modules

```text
src/llmtrain/data/
  config.py        # DataConfig (Pydantic schema)
  schemas.py       # Record dataclass、字段校验
  manifest.py      # shard 清单、不变性校验、版本号
  readers.py       # JSONL / Parquet 流式读取，rank/worker 切分
  pipeline.py      # 顺序组合的 record-level stage（validate/normalize/filter）
  mixer.py         # 多源 token-weighted + 温度采样
  packing.py       # 固定 seq_len pack，输出 document_ids
```

精简到 6 个文件。`normalizers` 与 `filters` 合并为 `pipeline.py`，每个 stage 是 `Callable[[Record], Record | None]`，返回 None 即丢弃；可恢复状态由各模块自身实现 `state_dict / load_state_dict`，不再单独抽 `state.py`。

### 5.4 Interfaces

跨模块边界通过 Protocol 显式定义。具体类型放 `src/llmtrain/interfaces.py`：

```python
@dataclass
class Record:
    id: str
    text: str
    source: str
    domain: str
    language: str
    metadata: dict

@dataclass
class Batch:
    input_ids: torch.Tensor       # [B, T]
    document_ids: torch.Tensor    # [B, T]，模型据此构造 block-diagonal mask
    consumed_tokens: int

class Stage(Protocol):
    def __call__(self, r: Record) -> Record | None: ...

class DataIterator(Protocol):
    def __iter__(self) -> Iterator[Batch]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, sd: dict) -> None: ...
```

Trainer 只依赖 `DataIterator`，不直接 import `data/` 下任何具体类。

### 5.5 Manifest

manifest 描述训练数据的确定版本，规则：

- **格式**：`manifest.jsonl`（每行一个 shard 描述）+ `manifest.meta.json`（头部 metadata + 全文件 sha256）。
- **不变性**：写盘后视为 immutable。新增数据走新版本 manifest 或 manifest delta，绝不原地修改。
- **校验**：训练启动时验证 `manifest.meta.json` 的 sha256；恢复 checkpoint 时再校验 manifest hash 与 ckpt 中保存的一致。
- **版本**：metadata 含 `manifest_version`（语义化版本）和 `created_at`。

每个 shard 描述字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | str | shard 唯一 id |
| uri | str | 本地路径或远程 URI |
| source | str | 数据源名（cci3, fineweb, the-stack, ...）|
| domain | str | 中文/英文/代码/数学/论文/百科 |
| language | str | 主语言（zh / en / multi）|
| format | str | jsonl 或 parquet |
| num_records | int | 文档条数 |
| bytes | int | 原始字节 |
| estimated_tokens | int | 用 tokenizer 抽样估算 |
| sha256 | str | 文件内容 hash |
| weight | float | 在 mixer 中的相对权重 |
| license | str | 许可证标识 |
| created_at | str | ISO 8601 |

### 5.6 Storage Format

- **训练流读首选 Parquet + pyarrow**：50B token 规模下，体积约为 JSONL 的 30%-50%，列式读取速度显著更优。
- JSONL 仅作为离线清洗的中间产物以及小规模验证用，不推荐直接喂大规模训练。
- 两种格式由 manifest 的 `format` 字段路由，readers 内部分派。

### 5.7 Sharding across Ranks and Workers

shard 到 (rank, worker) 的映射是**确定性**的，避免重复或漏读：

- 分配规则：`hash(shard.sha256) % (world_size * num_workers) == rank * num_workers + worker_id`。
- 不做动态拉取（动态拉取的状态难以序列化恢复）。
- 每个 worker 自维护 `(current_shard_id, byte_offset, consumed_records)`，checkpoint 时按 (rank, worker) 聚合到 rank-local state，分布式写盘走 `torch.distributed.checkpoint` (DCP)。
- 恢复时要求 `world_size` 与 `num_workers` 不变；若发生变化，必须走 first-class 的 `tools/reshard_data_state.py` 脚本，不在 trainer 内静默 reshard。

### 5.8 Mixer

- 权重以 **token 数**为基准（不是 sample 数），由 `estimated_tokens * weight` 推导。
- 温度采样：`p_i ∝ (tokens_i * weight_i)^α`，`α` 由配置控制，默认 1.0；`α<1` 弱化大源的统治。
- 训练中比例**不在 mixer 内动态调整**。curriculum 由 stage 切换（重启 trainer 换 config）实现，避免 mixer 内部状态复杂化。
- mixer 状态：一个 RNG + 各 source 已消费 tokens 计数器，全部写入 state_dict。

### 5.9 Packing

- 策略：concatenated packing + intra-document attention（block-diagonal mask）。
- 文档之间插入 `<|endoftext|>` 作为分隔 token；mask 阻断跨文档 attention。
- 输出 `document_ids` 张量，**不输出完整 [B,T,T] mask**——mask 由 attention 模块在前向时按 document_ids 构造（FlashAttention / SDPA 都支持变长 segment ids）。
- packing buffer 状态（未满序列的 token 列表与文档 id 列表）写入 state_dict。
- seq_len 支持 4096 / 8192 / 16384 / 32768，由 config 切换。

### 5.10 Resume Contract

`DataIterator.state_dict()` 必含字段：

```python
{
    "manifest_hash": str,
    "world_size": int,
    "num_workers": int,
    "rank_states": [
        {
            "rank": int,
            "worker_states": [
                {
                    "worker_id": int,
                    "current_shard_id": str,
                    "shard_byte_offset": int,
                    "consumed_records": int,
                },
            ],
        },
    ],
    "mixer": {
        "rng": bytes,
        "consumed_tokens_per_source": dict[str, int],
    },
    "packing": {
        "buffer_ids": list[int],
        "buffer_doc_ids": list[int],
    },
    "global_consumed_tokens": int,
}
```

恢复必须满足 `manifest_hash` / `world_size` / `num_workers` 三者一致，否则 `load_state_dict` 抛 `IncompatibleDataState` 并提示走 reshard 流程，绝不静默继续。

### 5.11 Training-Time Async Producer/Consumer

训练阶段的数据准备采用**异步生产者-消费者**，目标是把 CPU 侧的读盘、过滤、tokenize、packing 与 GPU 侧的 forward/backward 重叠起来，而不是提前落一份 token cache。

设计约束：

- 训练外部仍然只暴露 `DataIterator -> Batch` 接口，Trainer 不直接依赖 `reader` / `pipeline` / `tokenizer` 的具体实现。
- 每个 rank 只管理自己的本地 producer pool 和本地 queue，不共享跨 rank 队列。
- producer 采用 `multiprocessing.Process`，不是线程；tokenizer 属于 CPU-bound，线程不能真正吃满多核。
- 队列必须有界，防止 producer 跑赢 trainer 之后把内存堆满。

推荐数据流：

```text
rank-local shards
  -> reader / pipeline workers
  -> tokenizer workers
  -> packing
  -> multiprocessing.Queue(maxsize=N)
  -> trainer loop
```

producer 进程负责：

- 从 `ShardReader` 读取 `Record`
- 跑 `RecordPipeline`
- 调用 tokenizer
- 做 concatenated packing
- 将 batch 以 `numpy.ndarray` 形式写入队列

trainer 进程负责：

- 阻塞等待队列产出
- 将 `numpy.ndarray` 转成 `torch.Tensor`
- 做前向、反向、优化器更新
- 只在 batch 边界上推进 step / token 计数 / checkpoint

队列 payload 建议最少包含：

- `input_ids: np.ndarray`
- `document_ids: np.ndarray`
- `consumed_tokens: int`
- `producer_state: dict`（仅用于边界 checkpoint，不包含队列内飞行中的未消费样本）

恢复语义：

- checkpoint 只保存 producer 的 committed state，不保存 queue 里还没被 trainer 消费的 batch
- restart 后 producer 从最近一次提交点重建，未消费队列数据可以丢弃
- 这意味着恢复不是“精确恢复 queue 内容”，但能保证训练状态在 batch 边界继续前进

实现建议：

- 进程启动方式优先用 `spawn`
- 队列大小、worker 数、每次预取 batch 数都写进 YAML
- 第一版不做 token cache；如果后续需要复用，再单独做离线 cache，不和在线训练路径耦合

### 5.12 Acceptance Criteria

第一阶段交付时必须达成：

- 同时支持 JSONL 与 Parquet 两种格式构建 manifest，通过 sha256 校验。
- 流式读取吞吐 ≥ N tokens/sec/worker（N 在首轮基准后写回本节）。
- mixer 在 1B token 模拟运行后，各 source 实际 token 比例与目标比例偏差 < 1%。
- packing 通过 round-trip 单测：还原后文档边界与输入一致，attention mask 严格 block-diagonal。
- 保存 → kill → 恢复后，前 100 个 batch 与未中断训练 bit-exact 一致。
- 同一 manifest 下，`world_size` 或 `num_workers` 变化时 `load_state_dict` 显式拒绝并报清晰错误。

## 6. Tokenizer Design

Tokenizer 训练独立于模型训练，但依赖清洗后的数据和 manifest。

### 6.1 Tokenizer Corpus Sampling

从清洗后数据中分层采样约 200GB 文本。采样比例可以与最终训练比例接近，但建议略微提高代码、数学和论文占比，因为这些数据域对 tokenizer 质量更敏感。

初始建议：

| Domain | Sampling ratio |
| --- | --- |
| 中文通用 | 35% |
| 英文通用 | 20% |
| 代码 | 15% |
| 数学/推理 | 10% |
| 论文 | 15% |
| 百科 | 5% |

该比例不应写死，应由 `configs/data/tokenizer_sample_200gb.yaml` 控制。

### 6.2 SentencePiece BPE

建议配置方向：

```yaml
algorithm: sentencepiece_bpe
vocab_size: 150000
character_coverage: 0.9995
byte_fallback: true
normalization_rule_name: identity
shuffle_input_sentence: true
```

特殊 token 需要统一定义，并在模型配置、数据 packing 和 checkpoint 中保持一致。初始建议预留：

- `<unk>`
- `<s>`
- `</s>`
- `<pad>`
- `<|system|>`
- `<|user|>`
- `<|assistant|>`
- `<|tool|>`
- `<|endoftext|>`

### 6.3 Tokenizer Modules

```text
src/llmtrain/tokenizer/
  config.py           # TokenizerConfig (Pydantic schema)
  sampler.py          # 从 manifest 分层采样 tokenizer corpus
  trainer.py          # SentencePiece 训练封装（避免与 pip 包 sentencepiece 同名）
  inspector.py        # 压缩率、token 分布、特殊 token 检查（避免与标准库 inspect 同名）
  adapter.py          # 把 SP API 适配成 §5.4 Tokenizer Protocol
```

工具脚本：

```text
tools/sample_tokenizer_corpus.py
tools/train_tokenizer.py
tools/inspect_tokenizer.py
```

## 7. Model Design

模型实现遵循 Qwen/Llama 类 decoder-only Transformer 结构。项目手写 PyTorch 模块，但避免重造低级 kernel。

### 7.1 Block Structure

```text
input
  -> RMSNorm
  -> SelfAttention + RoPE
  -> residual
  -> RMSNorm
  -> SwiGLU MLP
  -> residual
```

### 7.2 Model Modules

```text
src/llmtrain/models/
  __init__.py         # 内含 build_model(cfg) 工厂函数
  config.py           # ModelConfig (Pydantic schema)，由 utils/config.py 组装
  decoder.py          # decoder-only TransformerLM，按 cfg.model.type 路由族系
  init_weights.py     # 参数初始化策略
  layers/
    rmsnorm.py
    rotary.py
    attention.py
    mlp.py
    block.py
```

### 7.3 Key Components

RMSNorm：

```text
y = x * rsqrt(mean(x^2) + eps) * weight
```

SwiGLU MLP：

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Attention：

- 支持 RoPE。
- 支持 GQA/MQA：`num_key_value_heads <= num_attention_heads`。
- 优先使用 `torch.nn.functional.scaled_dot_product_attention`。
- 支持 causal mask。
- 为长上下文阶段预留 rope scaling 或长上下文扩展策略。

### 7.4 Config Driven Scaling

模型规模不应写死在代码中，而应通过 YAML 配置控制：

```yaml
model_type: qwen_like
vocab_size: 150000
hidden_size: 2048
intermediate_size: 11008
num_hidden_layers: 24
num_attention_heads: 16
num_key_value_heads: 8
max_position_embeddings: 4096
rope_theta: 1000000
rms_norm_eps: 1.0e-6
tie_word_embeddings: false
```

具体参数后续再按 300M、700M、1.7B 目标规模校准。

## 8. Training Framework Design

训练框架暂时先做接口和模块边界，避免一开始过早绑定复杂分布式实现。

```text
src/llmtrain/training/
  config.py           # TrainerConfig (Pydantic schema)
  trainer.py          # 训练主循环
  schedule.py         # LR schedule
  optim.py            # optimizer 构建
  callbacks.py        # checkpoint/eval/logging hooks
  state.py            # global step、consumed tokens 等训练状态
```

训练主循环只依赖三类接口：

- `model`: 接收 token batch，返回 loss/logits。
- `data_iterator`: 可流式产出 packed batch，并支持 `state_dict()` / `load_state_dict()`。
- `checkpoint_manager`: 负责保存和恢复完整训练状态。

## 9. Checkpoint Design

Checkpoint 必须支持长时间大规模训练中的可靠恢复。

### 9.1 Save Policy

- 每 30-60 分钟保存一个 latest checkpoint。
- 每 0.5B-1B tokens 保存一个 milestone checkpoint。
- 每次验证前后保存一次。
- latest checkpoint 保留最近 2-3 个。
- milestone checkpoint 长期保留。
- best checkpoint 按 validation loss 保留。

### 9.2 Checkpoint Content

| Content | Required | Purpose |
| --- | --- | --- |
| 模型权重 | 必须 | 恢复模型参数 |
| 优化器状态 | 必须 | 恢复 AdamW 一阶/二阶动量 |
| LR scheduler 状态 | 必须 | 恢复学习率位置 |
| global step | 必须 | 确定训练进度 |
| consumed tokens / samples | 必须 | 确定已训练数据量 |
| dataloader / sampler 状态 | 必须 | 避免数据重复或漏读 |
| RNG 状态 | 强烈建议 | 保证 dropout、shuffle、采样一致 |
| 训练配置 config | 必须 | 防止恢复时配置不一致 |
| tokenizer 信息 | 必须 | 保证 token id 解释一致 |
| dataset manifest | 必须 | 保证数据 shard 顺序一致 |
| 分布式配置 | 必须 | FSDP/ZeRO 恢复需要 |

### 9.3 Logical Structure

按"大状态 sharded / 小 metadata 单文件"二分：

```python
# DCP: 大状态，跨 rank sharded 写入 model/ 与 optim/ 子目录
dcp_state = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}

# meta.pt: 小 metadata，rank 0 单文件 torch.save
meta = {
    "schema_version": "1.0",
    "scheduler": scheduler.state_dict(),
    "trainer_state": {
        "global_step": global_step,
        "consumed_tokens": consumed_tokens,
        "consumed_samples": consumed_samples,
    },
    "data_state": dataloader.state_dict(),     # §5.10 Resume Contract
    "rng_state": rng_state,                     # python / numpy / torch / cuda
    "resolved_config": cfg.model_dump(),        # §4.7
    "chain": [...],                             # fragment 文件 + sha256
    "tokenizer": tokenizer_metadata,
    "manifest": manifest_metadata,              # §5.5，含 manifest_hash
    "distributed": distributed_metadata,        # world_size / num_workers / backend
}
```

scheduler state 体积极小（几个标量），归入 meta 与 trainer_state 邻近，避免单独写一份 sharded。

### 9.4 Storage Format

| 内容 | 库 | 原因 |
| --- | --- | --- |
| `model` / `optimizer` 大状态 | `torch.distributed.checkpoint` (DCP) | FSDP 事实标准；支持 sharded save 与 resharding load（world_size 变化也能恢复） |
| `meta` 小 metadata | `torch.save`（pickle） | rank 0 单文件；适合 Python 对象（RNG bytes、嵌套 dict、Path 等），DCP 处理这些别扭 |
| 推理导出 | `safetensors` | 独立工具产出，不参与训练 ckpt 流程；纯权重，安全且加载快 |

不引入 HF accelerate / lightning，避免与自研 trainer 冲突。

**目录布局**（一个 ckpt 是一个目录，不是单文件）：

```text
checkpoints/milestone_0001000000/
  model/                       # DCP sharded
    __0_0.distcp
    ...
  optim/                       # DCP sharded
    ...
  meta.pt                      # rank 0 单文件
  _SUCCESS                     # 原子性标记，所有文件写完后最后创建
```

**原子性**：

- 写入顺序：`model/` → `optim/` → `meta.pt` → `_SUCCESS`。
- 启动扫 `checkpoints/` 时**仅承认含 `_SUCCESS` 的目录**，没有标记的视作脏 ckpt 直接清理。
- 防止训练崩溃留下半写状态被误用为恢复源。

**Resharding**：

`world_size` 或 `num_workers` 变化时：

- DCP 自动处理 `model` / `optim` 的 reshard。
- `data_state` 在 §5.10 已规定拒绝静默 resume，必须走 `tools/reshard_data_state.py`。
- 两条路径不冲突：DCP 解决张量重分布，data_state reshard 解决 shard ↔ worker 的重新分配。

**早期阶段兼容**：

300M smoke train 单卡 / 纯 DDP 下也走 DCP（fallback 非 sharded 模式自动可用），不维护两套代码路径。

### 9.5 Modules

```text
src/llmtrain/checkpointing/
  config.py           # CheckpointConfig (Pydantic schema)
  manager.py          # CheckpointManager: save / load_latest / list / cleanup
  io.py               # DCP + torch.save 封装，含 _SUCCESS 原子写入
```

## 10. Distributed Training Direction

分布式训练后续再落地，但接口应提前预留。

候选方向：

- PyTorch DDP：用于小模型和早期验证。
- FSDP / ZeRO：用于 1.7B 规模和更大 batch。
- activation checkpointing：降低显存压力。
- bf16 mixed precision：优先使用。
- gradient accumulation：补足全局 batch size。

分布式模块初始结构：

```text
src/llmtrain/distributed/
  config.py           # DistributedConfig (Pydantic schema)
  env.py
  launcher.py
  fsdp.py
  zero.py
  parallel_state.py
```

## 11. Observability and Failure Modes

长训不能瞎跑。本章定义训练过程中必须可见的信号、统一的事故处理策略以及 run 目录约定。

### 11.1 Metrics

三类信号默认采集，按 `global_step` 与 `global_consumed_tokens` 双维度索引：

| 类别 | 指标 |
| --- | --- |
| 标量（每步） | loss, lr, grad_norm, tokens/sec, samples/sec, gpu_mem_used, gpu_mem_reserved |
| 比例（每 N 步） | 各 source / domain 的实际 token 占比 |
| 事件（触发） | checkpoint save/load, loss spike, NaN/Inf, shard switch, restart, quarantine |

可选信号（config 开关，默认关）：参数 / 梯度 norm 直方图、激活值统计、tokenizer 实时压缩率。

### 11.2 Sinks

主循环不直接调 logger，全部通过 trainer callback 注入，避免指标代码污染训练逻辑：

- **`{output_dir}/metrics.jsonl`**：默认开启，ground truth。每行一条 record，事后 grep / pandas 直接用。
- **TensorBoard / W&B**：可选，由 config 切换，仅 rank 0 写入。
- **`{output_dir}/heartbeat.json`**：rank 0 后台线程每 60 秒写一次（含 step / consumed_tokens / loss / timestamp）。外层 watchdog（systemd / k8s / 脚本）据此判 hang。**trainer 内部不做 hang 自愈**。
- **控制台**：rank 0 每 N 步打印核心指标，N 可配置。

### 11.3 Failure Modes

策略基调：**能让训练继续就继续；不能继续就立即 fail**。绝不静默降级或自愈，避免事故被掩盖。

| 故障 | 检测 | 行为 |
| --- | --- | --- |
| Loss spike | 连续 N 步 loss > running_mean + kσ（默认 N=10, k=5） | 回滚最近 milestone ckpt，跳过出问题的 data window，记 event |
| NaN / Inf loss | 单步触发 | 同上，回滚优先 |
| OOM | `torch.cuda.OutOfMemoryError` | 直接 fail；启动时先按 config 估算并打印显存上限 |
| Manifest hash 不一致 | 启动校验 | 拒绝启动 |
| Shard 读异常 | 训练中读 record 抛异常 | 跳过该 shard 加入 `quarantine.json`，继续；累计 > 阈值（默认 5）则 fail |
| Tokenizer 不达标 | 阶段 0B inspection 工具门槛指标不通过 | 不自动恢复，人工判断是否重训 |
| Rank 间不一致 | 启动 barrier + config sha256 校验 | 拒绝启动 |
| Hang / 心跳停滞 | 外层 watchdog 监测 `heartbeat.json` | 杀进程重启，由 ckpt 恢复 |

### 11.4 Run Directory Layout

每个 run 在 `{run.output_dir}` 下统一产物，便于 grep / tar 归档 / 复现：

```text
{output_dir}/
  config.resolved.yaml        # §4.7
  metrics.jsonl
  events.jsonl                # spike / restart / quarantine 事件流
  heartbeat.json
  quarantine.json             # 跳过的坏 shard 列表
  checkpoints/
    latest/
    milestone_0001000000/
    ...
  logs/
    rank0.log
    rank1.log
    ...
```

### 11.5 Modules

```text
src/llmtrain/observability/
  config.py        # ObservabilityConfig (Pydantic schema)
  collectors.py    # 三类 metrics 采集（标量 / 比例 / 事件）
  sinks.py         # jsonl / tensorboard / wandb / console / heartbeat 各 sink
  detectors.py     # loss spike、NaN/Inf 检测器
  recovery.py      # spike 触发的自动回滚到最近 milestone
  callbacks.py     # 把 collectors / sinks / detectors / recovery 组装成 trainer callback
```

trainer 主循环只通过 `callbacks` 与本模块交互，不直接 import 具体 sink 或 detector。坏 shard 的 quarantine 由 `data/readers.py` 实现（读 shard 时最早能发现），仅把事件抛给 observability 写入 `events.jsonl` / `quarantine.json`。

## 12. Development Flow

开发顺序按“先固定契约，再实现流水线，再接入训练”的原则推进。每个阶段都必须产出可运行 CLI、配置样例和最小测试，避免只堆模块而没有端到端闭环。

### 12.1 Phase 1: Foundation + Data/Tokenizer Closed Loop

第一阶段目标是把训练前最容易影响长期稳定性的部分打牢：配置系统、统一接口、数据 manifest、流式读取、混合采样、packing、tokenizer 训练与 tokenizer 质量检查。此阶段不追求大规模分布式训练，也不要求 300M 模型收敛。

交付顺序：

1. 项目基础设施
   - `pyproject.toml`、包结构、基础 CLI、测试入口。
   - 在 `llmtrain` conda 环境中验证不破坏现有 Torch/沐曦后端。
   - 明确可选依赖：Parquet/pyarrow、SentencePiece、TensorBoard/W&B 等不应强制拖入训练核心。

2. 配置系统
   - 实现 `extends` yaml 合并、`--override`、Pydantic v2 schema。
   - 各模块就近维护 `config.py`。
   - 训练或工具启动时输出 `config.resolved.yaml`，并保留 config chain sha256。

3. 跨模块接口
   - 建立 `interfaces.py`，固定 `Record`、`Batch`、`Stage`、`DataIterator` 等协议。
   - Trainer、tokenizer、data pipeline 只依赖接口，不互相 import 具体实现。

4. Manifest 与 reader
   - 支持 JSONL 与 Parquet unified shard。
   - 构建 `manifest.jsonl` + `manifest.meta.json`。
   - 启动时做 manifest sha256 校验。
   - 实现确定性的 rank/worker shard 分配规则，单机先模拟 `world_size` / `num_workers`。

5. Record pipeline 与 mixer
   - 实现 schema validation、轻量 normalize/filter。
   - 实现 token-weighted mixer 与温度采样。
   - 记录各 source/domain 的 consumed tokens，输出比例统计。

6. Tokenizer corpus sampling
   - 基于 manifest 和 domain/source 配比采样文本。
   - 支持按 byte budget 停止，正式配置目标 200GB，小样例可用 MB 级 budget 跑通。
   - 输出采样统计，包含各 domain/source 字节数、记录数、估算 token 数。

7. SentencePiece tokenizer 工具链
   - 封装 SentencePiece BPE 训练，目标词表 150K。
   - 小样例测试允许较小 vocab，但配置结构必须与 150K 正式训练一致。
   - 实现 tokenizer metadata 与特殊 token 校验。

8. Tokenization + packing + resume
   - 使用 tokenizer adapter 将文本转为 token ids。
   - 实现 concatenated packing + `document_ids`。
   - 实现 `DataIterator.state_dict()` / `load_state_dict()`。
   - 实现 train-time async producer/consumer：producer 在后台多进程产出 numpy batch，trainer 从队列消费并训练。
   - 覆盖保存、kill、恢复后的数据一致性测试。

9. Observability 最小闭环
   - 写 `metrics.jsonl`、`events.jsonl`、`heartbeat.json`。
   - 数据比例、吞吐、坏 shard quarantine 事件能被记录。
   - 不在第一阶段实现完整 loss spike rollback，但保留 detector/callback 接口。

10. Checkpoint metadata 最小闭环
   - 先实现 `meta.pt` 中的 config、manifest、tokenizer、data_state、rng_state 保存与恢复。
   - DCP 的 model/optimizer sharded save 可以在 Phase 2 衔接，但目录布局和 `_SUCCESS` 原子标记第一阶段就确定。

### 12.2 Phase 2: Model + Single-node Smoke Training

第二阶段开始接模型与训练主循环，目标是用小模型证明 forward/backward、loss、checkpoint、恢复和 metrics 都能跑通。

交付顺序：

1. 手写 PyTorch 模型层：RMSNorm、RoPE、SwiGLU、Attention、TransformerBlock。
2. `qwen_like` decoder-only LM 与参数初始化。
3. 小配置模型 forward/backward 单测。
4. Trainer 主循环、optimizer、scheduler、gradient accumulation、bf16 开关。
5. DDP 或单机多卡 smoke training。
6. 完整 checkpoint manager：latest/milestone/best、DCP model/optimizer、`meta.pt`、`_SUCCESS`。
7. 300M / 700M 规模前的 tiny model 长跑稳定性验证。

### 12.3 Phase 3: Scale-up Preparation

第三阶段面向 300M/700M 和 1.7B 训练前的工程扩展。

交付顺序：

1. FSDP 或 ZeRO 接入。
2. activation checkpointing、bf16、梯度裁剪、吞吐 profiling。
3. `tools/reshard_data_state.py`，处理 `world_size` / `num_workers` 变化后的数据状态迁移。
4. Parquet reader 吞吐优化和数据预取。
5. tokenizer 200GB 正式训练与 inspection 报告。
6. 300M/700M 数据配比实验。
7. 1.7B stage1 general pretraining 启动前 checklist。

## 13. Phase 1 Deliverables and Acceptance Criteria

第一阶段的成果不是“已经能训练大模型”，而是“数据与 tokenizer 进入可复现、可恢复、可观测的工程状态”。完成后，应能用小样例数据跑完整工具链，也能把同一套工具切换到正式 200GB tokenizer 采样和 50B token manifest。

### 13.1 Required Deliverables

代码与配置：

- `pyproject.toml` 与可导入的 `llmtrain` Python 包。
- `configs/train/stage0_data_tokenizer.yaml`：第一阶段端到端入口配置。
- `configs/data/mixture_50b.yaml`：正式数据混合比例模板。
- `configs/data/tokenizer_sample_200gb.yaml`：tokenizer corpus 采样模板。
- `configs/tokenizer/hf_byte_bpe_150k.yaml`：正式 tokenizer 训练模板。
- 各模块的 Pydantic `config.py`，全部开启 `extra: forbid`。

数据工具：

- `tools/build_manifest.py`
- `tools/validate_manifest.py`
- `tools/sample_tokenizer_corpus.py`
- `tools/train_tokenizer.py`
- `tools/inspect_tokenizer.py`
- `tools/prepare_pretrain_stream.py`

核心模块：

- `src/llmtrain/interfaces.py`
- `src/llmtrain/utils/config.py`
- `src/llmtrain/data/{config.py,schemas.py,manifest.py,readers.py,pipeline.py,mixer.py,packing.py}`
- `src/llmtrain/tokenizer/{config.py,sampler.py,trainer.py,inspector.py,adapter.py}`
- `src/llmtrain/observability/{config.py,sinks.py,callbacks.py}`
- `src/llmtrain/checkpointing/{config.py,io.py}`，至少支持 metadata checkpoint。

样例与测试：

- `examples/data/` 中提供多 domain 小样例，覆盖中文、英文、代码、数学、论文、百科。
- `tests/` 覆盖 config merge、manifest hash、reader resume、mixer ratio、packing boundary、tokenizer adapter。
- `README.md` 给出第一阶段最小运行命令。

### 13.2 Functional Acceptance Criteria

数据与 manifest：

- 能从 JSONL 和 Parquet unified shards 构建 `manifest.jsonl` + `manifest.meta.json`。
- manifest 写入后可校验 sha256；故意修改 shard 或 manifest 时启动必须失败。
- reader 在单机模拟多 rank/worker 时无重复、无漏读。
- shard 分配由确定性 hash 规则决定，恢复时 `world_size` / `num_workers` 不一致必须显式拒绝。

数据混合与 packing：

- mixer 支持按 token 权重和温度采样混合不同 source/domain。
- 小规模模拟运行后，各 source/domain 实际 token 比例与目标比例偏差可统计并写入 metrics。
- packing 输出固定 `seq_len` 的 `input_ids` 与 `document_ids`。
- round-trip 测试能验证文档边界没有丢失，跨文档 attention 被 `document_ids` 阻断。
- packing buffer 可进入 `state_dict()`，恢复后继续产出的 batch 与未中断路径一致。

Tokenizer：

- 能从 manifest 按 domain/source 配比采样 tokenizer corpus，并按 byte budget 停止。
- 小样例能训练 SentencePiece BPE tokenizer；正式配置可直接切换到 150K vocab 和 200GB corpus。
- inspection 输出至少包含：各 domain 压缩率、平均 token/char、特殊 token id、byte fallback 使用情况、异常样例。
- tokenizer metadata 能写入 run 目录和 checkpoint metadata。

配置、可观测性与恢复：

- 所有 CLI 只需 `--config` 即可运行，`--override` 可用于临时调试。
- 每次运行写出 `config.resolved.yaml`、`metrics.jsonl`、`events.jsonl`、`heartbeat.json`。
- metadata checkpoint 能保存并恢复 config、manifest hash、tokenizer metadata、data iterator state、RNG state。
- kill 后从 metadata checkpoint 恢复，前 100 个 batch 与未中断基线一致。

### 13.3 Non-goals for Phase 1

以下内容不作为第一阶段完成标准：

- 300M / 700M 模型训练收敛。
- FSDP / ZeRO 完整接入。
- DCP model/optimizer sharded checkpoint 完整实现。
- loss spike 自动回滚。
- 200GB tokenizer corpus 的真实采样完成。
- 150K tokenizer 的最终质量达标。
- 50B token 数据全集 ready。

这些任务进入 Phase 2 或 Phase 3。第一阶段只要求同一套工程链路已经能在小样例上稳定、可复现地跑通，并且配置切换后能够承接正式规模任务。
