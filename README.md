# ExplicitLM - Qwen3 显式记忆增强语言模型

ExplicitLM 是一个基于 Qwen3-4B 的显式记忆增强语言模型，通过显式记忆库实现知识的透明管理和动态更新。模型采用参数高效训练策略，只训练记忆相关组件（MemoryGate、GatedMemoryFusion、MemoryNorm），冻结 Qwen3 backbone。

## 核心特性

- Qwen3 基础架构：基于 Qwen3-4B 预训练模型
- 显式记忆库：知识以 token 序列形式显式存储（Product Key Memory）
- 参数高效训练：冻结 Qwen3 主模型，只训练记忆相关组件（约 0.208B 可训练参数）
- 多阶段训练：支持分阶段训练 MemoryGate、Fusion 和联合微调
- 分布式训练：可以使用 DeepSpeed ZeRO Stage 2，自动处理多 GPU 训练

## 环境准备

### 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 安装依赖

```bash
git clone <repository-url>
cd ExplicitLM
uv sync
```

### 准备 Qwen3-4B 模型

下载 Qwen3-4B 模型到本地，记录模型路径。支持本地路径或 HuggingFace Hub 模型 ID：

```bash

/Qwen3-4b

# 或使用 HuggingFace Hub
Qwen/Qwen3-4B-Instruct
```

## 快速开始

### 使用便捷脚本

项目提供了便捷的训练脚本，位于 `scripts/` 目录：

**记忆组件训练**：
```bash

bash scripts/run_sft.sh
```

**MemoryGate (Router) 训练**：
```bash
bash scripts/run_router.sh
```

**数据格式转换**：
```bash
uv run python scripts/convert_omcq_to_sft.py \
    --input sft_data/omcq_trex_data.json \
    --output sft_data/omcq_trex_sft.jsonl
```

### 直接使用命令行

**记忆组件训练**（推荐）：
```bash
export CUDA_VISIBLE_DEVICES=4,5

uv run accelerate launch --config_file accelerate_config.yaml train_memory.py \
    model.qwen3_model_path=<YOUR_QWEN3_MODEL_PATH> \
    model.cache_path=data/cache/knowledge_cache.pt \
    model.keys_path=data/keys.pt \
    dataset.sft_dataset_path=sft_data/train.jsonl \
    dataset.sft_val_dataset_path=data/benchmarks/eval.jsonl \
    training.learning_rate=5e-5 \
    training.batch_size=1 \
    training.accumulation_steps=128 \
    training.epochs=3
```

**必需参数**：
- `model.qwen3_model_path`: Qwen3-4B 模型路径（HuggingFace 格式目录）
- `model.cache_path`: 记忆库缓存文件路径
- `model.keys_path`: Product Key Memory 的 keys 文件路径
- `dataset.sft_dataset_path`: 训练数据路径（JSONL 格式）
- `dataset.sft_val_dataset_path`: 验证数据路径（JSONL 格式）

## 数据准备

### 训练数据格式

训练数据必须是 JSONL 格式（每行一个 JSON 对象），包含对话格式：

```jsonl
{"conversations": [{"role": "user", "content": "问题内容"}, {"role": "assistant", "content": "回答内容"}]}
{"conversations": [{"role": "user", "content": "另一个问题"}, {"role": "assistant", "content": "另一个回答"}]}
```

字段要求：
- `conversations`: 对话列表，必须包含 `user` 和 `assistant` 角色
- `role`: 必须是 `"user"` 或 `"assistant"`
- `content`: 对话内容

### 验证数据格式

验证数据格式与训练数据相同，也是 JSONL 格式。

### 记忆库缓存文件

记忆库缓存文件（`knowledge_cache.pt`）包含预训练的记忆库内容。如果不存在，需要先运行预训练或使用已有的缓存文件。

生成缓存文件：
```bash
python train_pretrain.py \
    --qwen3_model_path=<YOUR_QWEN3_MODEL_PATH> \
    --database_init_path=data/knowledge_base/sentence_trex_data.json \
    --output_cache_path=data/cache/knowledge_cache.pt
```

### Keys 文件

Keys 文件（`keys.pt`）是 Product Key Memory 的查询键，通常通过 K-Means 训练生成。

生成 keys 文件：
```bash
python convert_conversations_to_labeled.py \
    --conversations_path=data/train.jsonl \
    --kb_path=data/knowledge_base/sentence_trex_data.json \
    --output_path=data/train_labeled.jsonl \
    --model_name=BAAI/bge-base-en-v1.5 \
    --top_k=32
# 该脚本会生成 keys.pt 和 meta.json
```

## 训练流程

### 阶段 1：MemoryGate 训练（可选）

训练 MemoryGate 组件，学习从查询中检索相关记忆。

使用脚本：
```bash
vim scripts/run_router.sh
# 修改 MODEL_NAME="<YOUR_QWEN3_MODEL_PATH>"
bash scripts/run_router.sh
```

或直接使用命令行：
```bash
export CUDA_VISIBLE_DEVICES=0,1
uv run python train_router.py \
    --data_path=data/train_labeled.jsonl \
    --model_name=<YOUR_QWEN3_MODEL_PATH> \
    --output_dir=checkpoints/router \
    --batch_size=1 \
    --lr=1e-4 \
    --epochs=3 \
    --knowledge_num=65536 \
    --num_candidates=32
```

输出：`checkpoints/router/memory_gate_epoch_X.pth`

### 阶段 2：知识融合训练（可选）

加载预训练的 MemoryGate，训练 GatedMemoryFusion。

```bash
uv run python train_fusion.py \
    --qwen3_model_path=<YOUR_QWEN3_MODEL_PATH> \
    --pretrained_memory_gate_path=checkpoints/router/memory_gate.pth \
    --dataset_path=data/database/merged_pretrain.jsonl \
    --val_dataset_path=data/benchmarks/eval_data.json \
    --knowledge_num=65536 \
    --batch_size=8 \
    --lr=1e-4 \
    --epochs=3
```

### 阶段 3：记忆组件训练（推荐）

直接训练所有记忆相关组件。

使用脚本：
```bash
vim scripts/run_sft.sh
# 修改 QWEN3_MODEL_PATH 和其他参数
bash scripts/run_sft.sh
```

或直接使用命令行：
```bash
export CUDA_VISIBLE_DEVICES=0,1
uv run accelerate launch --config_file accelerate_config.yaml train_memory.py \
    model.qwen3_model_path=<YOUR_QWEN3_MODEL_PATH> \
    model.cache_path=data/cache/knowledge_cache.pt \
    model.keys_path=data/keys.pt \
    model.recompute_cache=False \
    model.max_seq_len=256 \
    model.gate_rank=128 \
    model.fusion_rank=128 \
    dataset.sft_dataset_path=sft_data/train.jsonl \
    dataset.pretrained_router_path=checkpoints/router/memory_gate.pth \
    dataset.pretrained_fusion_path="" \
    dataset.sft_val_dataset_path=data/benchmarks/eval.jsonl \
    training.learning_rate=5e-5 \
    training.batch_size=1 \
    training.accumulation_steps=128 \
    training.epochs=3 \
    training.zero_stage=2 \
    logging.out_dir=out \
    logging.save_dir=out
```

## 配置参数

### 模型参数（model.*）

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `qwen3_model_path` | Qwen3-4B 模型路径 | - | 是 |
| `cache_path` | 记忆库缓存文件路径 | `data/cache/knowledge_cache.pt` | 是 |
| `keys_path` | Product Key Memory keys 文件路径 | `data/keys.pt` | 是 |
| `max_seq_len` | 最大序列长度 | 256 | 否 |
| `knowledge_num` | 记忆库条目数（需为完全平方数） | 1048576 | 否 |
| `knowledge_dim` | 记忆嵌入向量维度 | 1536 | 否 |
| `knowledge_length` | 每个记忆条目的 token 数 | 16 | 否 |
| `gate_rank` | MemoryGate LoRA rank（None=原版，>0=LoRA） | 128 | 否 |
| `fusion_rank` | Fusion LoRA rank（None=原版，>0=LoRA） | 128 | 否 |
| `recompute_cache` | 是否重新计算缓存 | False | 否 |

### 数据集参数（dataset.*）

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `sft_dataset_path` | 训练数据路径（JSONL） | - | 是 |
| `sft_val_dataset_path` | 验证数据路径（JSONL） | - | 是 |
| `pretrained_router_path` | Router 预训练权重路径 | "" | 否 |
| `pretrained_fusion_path` | Fusion 预训练权重路径 | "" | 否 |
| `system_message` | 系统提示消息 | "You are a helpful assistant." | 否 |

### 训练参数（training.*）

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `learning_rate` | 学习率 | 2e-4 | 否 |
| `batch_size` | 批次大小 | 4 | 否 |
| `accumulation_steps` | 梯度累积步数 | 32 | 否 |
| `epochs` | 训练轮数 | 3 | 否 |
| `zero_stage` | DeepSpeed ZeRO 阶段 | 2 | 否 |
| `num_candidates` | 记忆检索候选数 | 16 | 否 |
| `similarity_loss_coef` | 相似度损失系数 | 0.1 | 否 |
| `diversity_loss_coef` | 多样性损失系数 | 0.05 | 否 |
| `eval_interval` | 评估间隔（步数） | 500 | 否 |
| `eval_num_samples` | 评估样本数量 | 200 | 否 |

### 日志参数（logging.*）

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `out_dir` | 输出目录 | `out` | 否 |
| `save_dir` | 检查点保存目录 | `out` | 否 |
| `use_swanlab` | 是否使用 SwanLab | True | 否 |
| `swanlab_project` | SwanLab 项目名 | "explicitlm" | 否 |
| `log_interval` | 日志记录间隔（步数） | 10 | 否 |

## 数据格式

### 训练数据（JSONL）

每行一个 JSON 对象，包含对话格式：

```jsonl
{"conversations": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"conversations": [{"role": "user", "content": "Explain quantum computing."}, {"role": "assistant", "content": "Quantum computing uses quantum mechanical phenomena..."}]}
```

要求：
- 文件必须是 UTF-8 编码
- 每行一个完整的 JSON 对象
- `conversations` 字段必须存在
- 每个对话必须包含 `user` 和 `assistant` 角色
- `role` 字段必须是 `"user"` 或 `"assistant"`

### 验证数据（JSONL）

格式与训练数据完全相同。

### 记忆库缓存文件（.pt）

PyTorch 格式的二进制文件，包含记忆库的 token 序列。

文件结构：
- 类型：`torch.Tensor`
- 形状：`[knowledge_num, knowledge_length, hidden_size]`
- 示例：`[1048576, 16, 2560]`（1048576 个记忆条目，每个 16 个 token，维度 2560）

### Keys 文件（.pt）

PyTorch 格式的二进制文件，包含 Product Key Memory 的查询键。

文件结构：
- 类型：`torch.Tensor`
- 形状：`[knowledge_num, knowledge_dim]`
- 示例：`[1048576, 1536]`（1048576 个键，每个维度 1536）

## 文件路径

### 必需文件

1. **Qwen3-4B 模型**
   - 路径：通过 `model.qwen3_model_path` 指定
   - 格式：HuggingFace 模型目录
   - 示例：`/data/models/Qwen3-4B-Instruct` 或 `Qwen/Qwen3-4B-Instruct`（HuggingFace Hub）

2. **记忆库缓存文件**
   - 路径：通过 `model.cache_path` 指定
   - 格式：`.pt` 文件
   - 示例：`data/cache/knowledge_cache.pt`

3. **Keys 文件**
   - 路径：通过 `model.keys_path` 指定
   - 格式：`.pt` 文件
   - 示例：`data/keys.pt`

4. **训练数据**
   - 路径：通过 `dataset.sft_dataset_path` 指定
   - 格式：`.jsonl` 文件
   - 示例：`sft_data/train.jsonl`

5. **验证数据**
   - 路径：通过 `dataset.sft_val_dataset_path` 指定
   - 格式：`.jsonl` 文件
   - 示例：`data/benchmarks/eval.jsonl`

### 可选文件

1. **预训练 Router 权重**
   - 路径：通过 `dataset.pretrained_router_path` 指定
   - 格式：`.pth` 或 `.pt` 文件
   - 示例：`checkpoints/router/memory_gate.pth`

2. **预训练 Fusion 权重**
   - 路径：通过 `dataset.pretrained_fusion_path` 指定
   - 格式：`.pth` 或 `.pt` 文件
   - 示例：`checkpoints/fusion/fusion_weights.pth`

### 输出文件

训练完成后，会在 `logging.save_dir` 目录下生成：
- `sft_*.pth`: 训练检查点（按步数命名）
- `sft_latest.pth`: 最新检查点

## 使用示例

### 基础训练

```bash
export CUDA_VISIBLE_DEVICES=0,1

uv run accelerate launch --config_file accelerate_config.yaml train_memory.py \
    model.qwen3_model_path=/data/models/Qwen3-4b \
    model.cache_path=/data/cache/knowledge_cache.pt \
    model.keys_path=/data/keys.pt \
    dataset.sft_dataset_path=/data/train.jsonl \
    dataset.sft_val_dataset_path=/data/eval.jsonl \
    training.learning_rate=5e-5 \
    training.batch_size=1 \
    training.accumulation_steps=128 \
    training.epochs=3
```

### 使用预训练权重

```bash
export CUDA_VISIBLE_DEVICES=0,1

uv run accelerate launch --config_file accelerate_config.yaml train_memory.py \
    model.qwen3_model_path=/data/models/Qwen3-4b \
    model.cache_path=/data/cache/knowledge_cache.pt \
    model.keys_path=/data/keys.pt \
    dataset.sft_dataset_path=/data/train.jsonl \
    dataset.sft_val_dataset_path=/data/eval.jsonl \
    dataset.pretrained_router_path=/data/checkpoints/router.pth \
    training.learning_rate=5e-5 \
    training.batch_size=1 \
    training.accumulation_steps=128 \
    training.epochs=3
```

### 自定义配置

```bash
export CUDA_VISIBLE_DEVICES=0,1

uv run accelerate launch --config_file accelerate_config.yaml train_memory.py \
    model.qwen3_model_path=/data/models/Qwen3-4b \
    model.cache_path=/data/cache/knowledge_cache.pt \
    model.keys_path=/data/keys.pt \
    model.max_seq_len=512 \
    model.gate_rank=128 \
    model.fusion_rank=128 \
    dataset.sft_dataset_path=/data/train.jsonl \
    dataset.sft_val_dataset_path=/data/eval.jsonl \
    training.learning_rate=1e-4 \
    training.batch_size=2 \
    training.accumulation_steps=64 \
    training.epochs=5 \
    training.similarity_loss_coef=0.2 \
    training.diversity_loss_coef=0.1 \
    logging.out_dir=/data/output \
    logging.save_dir=/data/checkpoints
```

## 注意事项

### 记忆库缓存文件

如果已有预训练的记忆库，直接使用。否则需要运行预训练脚本生成缓存，或使用已有的缓存文件。

### Keys 文件生成

使用 `convert_conversations_to_labeled.py` 脚本生成：

```bash
python convert_conversations_to_labeled.py \
    --conversations_path=data/train.jsonl \
    --kb_path=data/knowledge_base/sentence_trex_data.json \
    --output_path=data/train_labeled.jsonl \
    --model_name=BAAI/bge-base-en-v1.5
```

### 显存优化

- 减小 `training.batch_size`（如从 2 改为 1）
- 增加 `training.accumulation_steps` 保持有效批次大小
- 减小 `model.max_seq_len`（如从 512 改为 256）
- 使用更多 GPU（增加 `CUDA_VISIBLE_DEVICES`）

### 数据格式验证

```bash
# 检查 JSONL 格式
head -n 1 data/train.jsonl | python -m json.tool

# 检查文件编码
file -bi data/train.jsonl
```

### 参数传递

使用 `key=value` 格式，支持点号访问。参数会自动进行类型转换（布尔值、整数、浮点数，包括科学计数法）：

```bash
model.qwen3_model_path=/data/models/Qwen3-4B-Instruct
dataset.sft_dataset_path=/data/train.jsonl
training.learning_rate=5e-5
```

### 训练日志

- 控制台输出：实时显示训练进度
- SwanLab：如果启用，可通过 Web 界面查看（需要设置 `SWANLAB_API_KEY`）
- 日志文件：保存在 `logging.out_dir` 目录

### 检查点文件

检查点保存在 `logging.save_dir` 目录：
- `sft_*.pth`: 按步数命名的检查点
- `sft_latest.pth`: 最新检查点

### 分布式训练

使用 `accelerate launch` 和 `accelerate_config.yaml`：

```bash
uv run accelerate launch --config_file accelerate_config.yaml train_memory.py ...
```

数据加载器会自动使用 `num_workers=0` 以避免 NCCL 同步问题。

## 相关文档

- [记忆组件训练详细指南](docs/SFT_TRAINING.md)
- [Git 提交指南](GIT_COMMIT_GUIDE.md)

## 项目结构

```
ExplicitLM/
├── models/              # 模型架构
│   ├── core/           # 核心模型
│   └── memory_bank/    # 记忆库组件
├── config/             # 配置模块
├── utils/              # 工具函数
├── scripts/            # 启动脚本
├── train_memory.py     # 记忆组件训练入口（推荐）
├── train_router.py     # MemoryGate 训练
├── train_fusion.py     # Fusion 训练
├── train_joint.py      # 联合微调
└── pyproject.toml      # 项目依赖
```
