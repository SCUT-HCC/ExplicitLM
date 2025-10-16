# ExplicitLM

ExplicitLM是一个创新的语言模型架构，通过引入显式记忆库（Memory Bank）解决传统语言模型知识更新困难和可解释性不足的问题。与传统模型将知识隐式存储在参数中不同，ExplicitLM将知识以token序列的形式显式存储在共享记忆库中，通过可微分的检索和门控机制实现知识的透明管理、动态更新和端到端训练。该架构采用EMA（指数移动平均）更新机制，借鉴VQ-VAE的思想实现记忆库的渐进式优化，在保持模型可解释性的同时显著提升知识存储效率和答案准确性。

## 核心创新

**显式记忆库设计**：ExplicitLM使用一个共享的记忆库替代传统Transformer的FFN层，记忆库包含大量可学习的token序列（记忆条目）。每个记忆条目由固定长度的token组成，并配有对应的嵌入向量用于检索。模型在处理输入时，通过相似度计算从记忆库中检索相关条目，并通过门控机制将检索到的知识融入到表示中。这种显式存储方式使得知识可被直接查看、分析和修改，极大地提升了模型的可解释性。

**EMA动态更新机制**：受VQ-VAE中codebook更新策略的启发，ExplicitLM采用指数移动平均（EMA）机制动态更新记忆库。在训练过程中，被检索到的记忆条目会根据当前输入的表示进行渐进式更新，通过`ema_decay`参数控制新旧知识的融合速度。这种更新方式避免了记忆库的剧烈变化，同时允许模型持续学习新知识。系统还支持记忆冻结策略（`freeze_ratio`），可以保护部分稳定的重要记忆条目不被修改，防止灾难性遗忘。

**Gumbel-Softmax候选选择**：为了优化记忆检索的效率和效果，模型采用两阶段检索策略。第一阶段通过快速相似度计算筛选出`num_candidates`个候选记忆条目，第二阶段使用Gumbel-Softmax技术从候选中选择最相关的条目。Gumbel-Softmax的引入使得选择过程可微分，支持端到端的梯度传播，同时通过温度参数控制选择的确定性与探索性之间的平衡。

**门控知识融合**：检索到的记忆条目通过专门的门控模块与输入表示融合。门控机制动态决定每个位置应该采纳多少记忆知识和保留多少原始信息，这种自适应融合策略使模型能够根据上下文灵活地利用显式知识。

## 项目结构

```
ExplicitLM/                          # 项目根目录
├── 1_pretrain.py                    # 预训练主程序（十阶段训练流程）
├── main.py                          # 简单测试入口
├── models/                          # 模型架构实现
│   ├── configs/
│   │   └── LMConfig.py             # 统一超参数配置类
│   ├── core/
│   │   ├── ExplicitLM.py           # 主模型类（基于PreTrainedModel）
│   │   └── ExplicitLMBlock.py      # Transformer Block实现
│   ├── layers/
│   │   ├── Attention.py            # 多头注意力层和RoPE位置编码
│   │   ├── RMSNorm.py              # RMS归一化层
│   │   └── pos_cis.py              # 位置编码相关工具
│   └── memory_bank/
│       ├── GatedMemoryFusion.py    # 门控记忆融合模块
│       └── MemoryGate.py           # 记忆门控层
├── utils/                           # 工具模块
│   ├── config_utils.py             # 配置管理（自动从LMConfig提取参数）
│   ├── model_initializer.py        # 模型初始化工具
│   ├── pretrain_datasets.py        # 数据加载器（支持训练集和验证集）
│   ├── train_loop.py               # 训练循环实现
│   ├── train_utils.py              # 训练辅助工具
│   └── Logger.py                   # 分布式日志工具
├── data/                            # 数据管理（DVC追踪）
│   ├── database/                    # 预训练数据集
│   │   └── merged_pretrain.jsonl   # 训练数据（10.77 GB）
│   ├── database.dvc                 # database目录DVC追踪文件
│   ├── benchmarks/                  # 验证数据集
│   │   └── eval_data.json          # 评估数据（28 KB）
│   ├── benchmarks.dvc               # benchmarks目录DVC追踪文件
│   ├── knowledge_base/              # 显式知识库数据
│   │   └── sentence_trex_data.json # T-REx知识库（446 MB）
│   ├── knowledge_base.dvc           # knowledge_base目录DVC追踪文件
│   └── raw/                         # 原始数据
├── experiments/                     # 实验管理系统
│   ├── scripts/                     # 实验运行脚本
│   │   ├── _run_experiment_core.sh              # 单机实验核心脚本
│   │   ├── _run_experiment_cluster_pre.sh       # 集群前置阶段脚本
│   │   ├── _run_experiment_cluster_train.sh     # 集群训练阶段脚本
│   │   ├── _run_experiment_cluster_post.sh      # 集群后续阶段脚本
│   │   ├── exp_001.sh                           # 实验001（单机）
│   │   ├── exp_001_cluster.sh                   # 实验001（集群）
│   │   └── ...                                  # 其他实验脚本
│   └── records/                     # 实验记录（JSON格式）
│       ├── README.md                # 实验记录系统说明文档
│       ├── exp_001.json             # 实验001元数据记录
│       └── ...                      # 其他实验记录
├── cache/                           # 缓存数据（DVC追踪）
│   ├── knowledge_cache.pt           # 知识缓存（128 MB）
│   └── cluster_tokens_single_mapping.json  # 聚类token映射（338 MB）
├── cache.dvc                        # cache目录DVC追踪文件
├── checkpoints/                     # 模型检查点（DVC追踪）
│   ├── exp_001/                     # 实验001的模型权重
│   ├── exp_001.dvc                  # DVC版本追踪文件
│   └── ...
├── logs/                            # 训练日志
│   ├── exp_001.log                  # 实验001训练日志
│   └── ...
├── docs/                            # 项目文档
│   ├── experiment_workflow.md       # 实验运行指南（DVC、训练流程）
│   ├── dvc_guide.md                 # DVC数据版本管理指南（常用指令、版本记录）
│   └── uv.md                        # uv包管理器使用指南
├── scripts/                         # 实用脚本
│   ├── data_processing/             # 数据处理脚本
│   ├── model_analysis/              # 模型分析脚本
│   └── deployment/                  # 部署相关脚本
├── evaluation/                      # 评估工具
├── tests/                           # 单元测试
├── visualization/                   # 可视化工具
│   ├── memory_analysis/             # Memory Bank可视化
│   └── attention_maps/              # 注意力机制可视化
├── .dvc/                            # DVC配置目录
├── .gitignore                       # Git忽略规则（包含DVC相关配置）
├── ds_config.json                   # DeepSpeed配置（ZeRO-2优化）
└── pyproject.toml                   # uv项目配置文件
```

## 超参数配置系统

ExplicitLM采用基于优先级的超参数配置系统，所有超参数集中定义在`models/configs/LMConfig.py`中。系统通过`utils/config_utils.py`的`setup_config()`函数自动解析命令行参数并与配置文件合并，实现了"一行代码完成配置"的简洁接口。配置优先级遵循"命令行参数优先于配置文件默认值"的原则，确保实验参数的灵活性和可追溯性。特别地，系统不使用argparse的default参数，避免默认值掩盖潜在的配置错误。

### 主要参数分组

#### 基本模型架构参数

这组参数定义了Transformer的核心结构，控制模型的基础容量和计算复杂度。`dim`（默认512）设置模型的隐藏维度，`n_layers`（默认8）决定Transformer层的数量，`n_heads`（默认16）和`n_kv_heads`（默认8）分别控制查询头和键值头的数量，支持Group Query Attention以提升效率。`vocab_size`（默认6400）定义词汇表大小，`max_seq_len`（默认512）限制最大序列长度。其他参数包括`hidden_dim`（FFN隐藏层维度）、`rope_theta`（RoPE位置编码基频率）、`dropout`（Dropout比率）、`flash_attn`（是否启用Flash Attention）等。

#### 知识库相关配置

这是ExplicitLM的核心创新模块。`knowledge_num`（默认1024×1024=1048576）定义显式记忆库的条目总数，决定了模型的知识存储容量。`knowledge_length`（默认16）指定每个记忆条目包含的token数量，过长会增加计算开销，过短可能无法表达完整知识。`knowledge_dim`（默认128）设置每个记忆条目对应的嵌入向量维度，用于检索时的相似度计算。这三个参数共同决定了记忆库的规模和性能特性。

#### EMA更新相关配置

借鉴VQ-VAE的指数移动平均机制实现记忆库的动态更新。`use_ema_update`（默认True）控制是否启用EMA更新模式，启用后记忆库中的`requires_grad`将被设为False，通过EMA机制而非梯度下降更新。`ema_decay`（默认0.9）是关键的超参数，值越大表示更重视历史知识（更新速度慢），值越小表示更快适应新知识。`ema_update_freq`（默认5）设置EMA更新的频率，每隔几个batch更新一次记忆库。`freeze_ratio`（默认0.2）允许冻结部分记忆条目，例如设为0.2表示20%的记忆条目不会被EMA更新，可以保护重要的稳定知识。`use_token_memory`（默认True）是token-based memory的启用标志。

#### 实验1.4.10配置（Gumbel-Softmax优化）

这组参数控制优化的记忆选择机制，是当前主要研究方向。`num_candidates`（默认16）定义第一阶段检索的候选记忆条目数量，在检索准确性和计算效率之间权衡。`num_selected`（默认1）指定第二阶段选中的记忆条目数量，当前实验中只选择1个最佳条目。`gumbel_temperature`（默认1.0）控制Gumbel-Softmax的温度参数，高温使选择更随机（增加探索），低温使选择更确定（增加利用）。

#### MOE混合专家配置

可选的混合专家架构支持。`use_moe`（默认False）控制是否启用MOE，启用后可以增强模型的表达能力。`num_experts_per_tok`（默认2）指定每个token激活的专家数量，`n_routed_experts`（默认4）定义总的专家数量。其他参数包括`n_shared_experts`（是否使用共享专家）、`scoring_func`（专家选择的评分函数）、`aux_loss_alpha`（辅助损失系数）等。

#### 训练相关配置

控制训练过程的关键参数。`dataset_path`（默认"data/database/merged_pretrain.jsonl"）指定预训练数据集路径，`val_dataset_path`（默认"data/benchmarks/eval_data.json"）指定验证数据集路径。`batch_size`（默认48）设置每个设备上的批次大小，`accumulation_steps`（默认16）控制梯度累积步数，实际的全局批次大小等于`batch_size × accumulation_steps × num_gpus`。`epochs`（默认3）定义训练轮数，`learning_rate`（默认2e-4）设置初始学习率。`out_dir`（默认"out"）指定输出目录，用于保存模型检查点和训练日志。

#### SwanLab实验追踪配置

集成SwanLab进行实验可视化和管理。`use_swanlab`（默认False）控制是否启用SwanLab追踪，`swanlab_online`（默认False）选择在线或离线模式，`swanlab_project`（默认"MiniMind"）设置SwanLab项目名称。启用后，训练过程中的所有指标（损失、学习率、验证准确率等）都会自动上传到SwanLab平台。

#### 模型初始化配置

控制模型的初始化方式。`model_variant`（默认"model_memory"）选择模型变体类型，可选值包括`model`、`model_original`、`model_no_feed`、`model_memory`等，对应不同的架构变种。`pretrained_embedding_path`（默认None）可以指定预训练的嵌入权重文件路径，用于迁移学习。`database_init_path`（默认None）用于指定知识库初始化数据文件路径，可以从外部知识源初始化记忆库。`cache_path`（默认"cache/knowledge_cache.pt"）设置处理后数据的缓存路径，`recompute_cache`（默认False）控制是否强制重新计算缓存。

### 使用示例

在训练脚本中只需一行代码即可完成配置：

```python
from utils.config_utils import setup_config

# 自动加载配置（默认值来自LMConfig）
config = setup_config()
```

通过命令行参数覆盖默认配置：

```bash
# 修改模型规模并启用MOE
python 1_pretrain.py --dim 1024 --n_layers 12 --use_moe

# 调整记忆库大小和EMA参数
python 1_pretrain.py --knowledge_num 2097152 --ema_decay 0.95 --num_candidates 32

# 启用SwanLab在线追踪
python 1_pretrain.py --use_swanlab --swanlab_online --swanlab_project "ExplicitLM-Exp"

# 修改训练参数
python 1_pretrain.py --epochs 10 --batch_size 64 --learning_rate 1e-4
```

查看所有可用参数：

```bash
python 1_pretrain.py --help
```

## 训练流程

ExplicitLM采用十阶段的训练流程，完整实现在`1_pretrain.py`中。整个流程集成了Accelerate和DeepSpeed，支持分布式训练、混合精度和ZeRO优化。

### 阶段一：配置初始化

通过`setup_config()`函数加载超参数配置，该函数会自动从`LMConfig`提取所有参数定义并与命令行参数合并。配置系统采用优先级机制：命令行参数优先于配置文件默认值，确保实验的灵活性和可复现性。

### 阶段二：Accelerator和DeepSpeed初始化

初始化分布式训练环境。配置DDP（Distributed Data Parallel）时允许未使用的参数（`find_unused_parameters=True`），这对于EMA模式下部分参数不参与梯度计算的情况至关重要。DeepSpeed插件配置为ZeRO-2优化级别，具体设置在`ds_config.json`中定义，包括CPU offload（优化器和参数）、bf16混合精度、梯度裁剪等。Accelerator会根据配置自动处理模型分布、混合精度转换和梯度同步。

### 阶段三：随机种子设置

为每个进程设置不同的随机种子（`1337 + process_index`），确保数据采样的多样性和训练的可复现性。不同进程使用不同的种子可以避免数据重复，同时保证整体实验的确定性。

### 阶段四：创建输出目录

仅主进程负责创建输出目录（`out_dir`和`save_dir`），避免多进程竞争导致的文件系统错误。目录用于保存模型检查点、训练日志和实验元数据。

### 阶段五：SwanLab实验追踪配置

如果启用SwanLab（`use_swanlab=True`），主进程会初始化SwanLab实验实例。系统会自动生成实验运行名称，格式为`MiniMind-Pretrain-Epoch-{epochs}-BatchSize-{batch_size}-LearningRate-{lr}`，便于在SwanLab界面中识别和对比不同实验。根据`swanlab_online`参数选择在线或离线模式，离线模式下数据会先保存在本地，等网络恢复后自动上传。所有配置参数都会自动同步到SwanLab，方便后续查询和对比。

### 阶段六：模型初始化

调用`init_model()`函数创建模型实例和tokenizer。该函数根据`model_variant`参数选择对应的模型架构，支持从预训练嵌入或知识库数据初始化。模型创建后，通过`accelerator.prepare(model)`进行包装，Accelerator会根据配置自动应用混合精度、分布式策略和DeepSpeed优化。这一步之后，模型已经分布到各个设备上，并且启用了ZeRO-2优化。

### 阶段七：数据加载器初始化

创建训练数据加载器和验证数据加载器。`create_pretrain_dataloader()`函数负责加载预训练数据集，支持自定义batch size、最大序列长度、数据打乱等参数。`create_validation_dataloader()`函数加载验证数据集，通常采样固定数量的样本（如200个）用于快速评估。两个数据加载器都会被`accelerator.prepare()`包装，Accelerator会自动处理数据分片（每个进程只加载部分数据）和分布式采样。

### 阶段八：优化器和调度器初始化

这是一个关键阶段，需要特别处理EMA更新模式。首先，系统会检查模型配置中的`use_ema_update`标志。如果启用EMA模式，只有`requires_grad=True`的参数会被包含在优化器中（记忆库等EMA更新的参数会被自动排除）；如果是传统模式，所有参数都参与优化。优化器使用AdamW，配置为最佳实践参数：`betas=(0.9, 0.95)`和`weight_decay=0.1`。

学习率调度器采用带warmup的余弦退火策略。系统会自动计算总的优化步数（考虑梯度累积），并将10%的步数用于warmup阶段，剩余步数按余弦曲线衰减学习率。优化器和调度器创建后，必须通过`accelerator.prepare()`包装，这一步让DeepSpeed接管优化器管理，激活ZeRO优化和CPU offload功能。

### 阶段九：训练循环

调用`train_epoch()`函数执行训练循环。该函数实现了完整的训练逻辑，包括前向传播、损失计算、反向传播、梯度累积、梯度裁剪、优化器更新、学习率调度、验证评估、检查点保存等。训练过程中的所有指标会自动记录到SwanLab（如果启用）。每个epoch结束后，系统会执行内存清理（`gc.collect()`和`torch.cuda.empty_cache()`），释放不再需要的缓存，避免内存泄漏。

### 阶段十：SwanLab URL导出和关闭

训练完成后，主进程会从SwanLab运行实例中提取实验URL，并写入到`.swanlab_url`文件中。这个文件会被实验管理脚本读取，记录到实验元数据JSON文件中，便于日后查看和复现。最后，调用`swanlab_run.finish()`优雅地关闭SwanLab运行，确保所有数据都被正确上传和保存。

## 实验管理系统

ExplicitLM集成了完整的实验管理系统，基于Git（代码版本）、DVC（数据和模型权重版本）和SwanLab（指标可视化）三者协同，并通过CI/CD实现标准化的持续交付流程。系统支持单机模式和集群模式，提供细粒度的数据版本控制和自动化的实验记录生成，所有实验版本的结果可通过GitHub进行统一管理和追溯。详细的使用指南请参阅[实验运行指南](docs/experiment_workflow.md)。

### 核心原则

每个实验对应一个shell脚本文件，脚本中定义实验ID、描述、数据版本和训练参数。所有实验配置集中在脚本顶部，避免参数散落在多处导致的不一致。实验执行后会自动生成JSON格式的元数据记录，包含代码版本、数据版本、超参数、结果URL、复现指令等完整信息。系统采用单次提交策略：训练完成后，代码修改、实验记录和DVC元数据会被一次性提交到Git仓库，保证版本的原子性。

### 单机模式

单机模式适用于个人工作站或单个服务器环境。创建实验脚本（如`experiments/scripts/exp_003.sh`），定义实验配置，然后执行脚本即可自动完成数据同步、训练、结果收集、版本提交等全流程。脚本会智能检测数据版本是否需要更新，只同步发生变化的数据集，节省时间。训练完成后，模型权重通过DVC上传到远程存储，实验记录通过Git提交到代码仓库。

```bash
# 创建实验脚本
cat > experiments/scripts/exp_003.sh <<'EOF'
#!/bin/bash
EXP_ID="exp_003"
EXP_DESC="学习率降低至1e-4，增加模型层数至12层"
DATASET_VERSION=""  # 使用当前版本
VAL_DATASET_VERSION=""
TRAIN_ARGS="--epochs 10 --dim 512 --n_layers 12 --learning_rate 1e-4 --use_swanlab"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_experiment_core.sh"
EOF

# 赋予执行权限并运行
chmod +x experiments/scripts/exp_003.sh
./experiments/scripts/exp_003.sh
```

### 集群模式

集群模式解决了登录节点（有网络无GPU）和计算节点（有GPU无网络）分离的问题。实验流程分为三个阶段：

1. **前置阶段（登录节点）**：执行`./exp_001_cluster.sh pre`，完成数据同步和版本记录。系统会智能比较数据版本，只同步发生变化的数据集，避免在拥堵的集群网络中浪费时间。所有配置信息保存到状态文件`.cluster_state_${EXP_ID}`。

2. **训练阶段（计算节点）**：将工作目录同步到计算节点后，执行`./exp_001_cluster.sh train`，加载状态文件并启动训练。训练完成后，SwanLab URL和模型权重会保存在本地。

3. **后续阶段（登录节点）**：将结果同步回登录节点后，执行`./exp_001_cluster.sh post`，完成DVC上传、实验记录生成和Git提交。

三个阶段共享同一个脚本文件，所有参数只需在脚本顶部定义一次，通过命令行参数（pre/train/post）控制执行哪个阶段，避免了多文件维护的复杂性。

### 细粒度数据版本控制

ExplicitLM支持为每个数据集独立指定版本，包括训练数据集（`DATASET_VERSION`）、验证数据集（`VAL_DATASET_VERSION`）、预训练嵌入（`EMBEDDING_VERSION`）、知识库初始化数据（`DATABASE_VERSION`）、缓存数据（`CACHE_VERSION`）等。这种细粒度控制使得可以在保持部分数据不变的情况下更新其他数据，非常适合对比实验和消融研究。

版本可以留空（使用当前版本）或指定Git commit哈希（使用历史版本）。系统会自动切换到对应版本的.dvc文件，执行`dvc checkout`恢复数据，然后切换回当前代码分支。实验记录中会详细记录每个数据集的版本信息，确保实验的可复现性。

### 实验记录系统

每个实验完成后会在`experiments/records/`目录下生成一个JSON文件（如`exp_001.json`），包含以下信息：

- **实验基本信息**：ID、描述、时间戳、执行脚本、训练命令
- **版本信息**：代码commit哈希、各数据集版本、模型权重DVC哈希
- **超参数**：所有训练参数（自动从命令行提取）
- **结果**：SwanLab实验URL、模型权重路径
- **运行环境**：Python版本、CUDA版本、GPU数量
- **复现指令**：Git代码恢复命令、DVC数据恢复命令、完整训练命令

通过实验记录，可以精确复现任何历史实验。系统提供了便捷的查询工具，支持按参数、时间、数据版本等条件检索实验。所有实验版本的结果通过GitHub进行集中管理，每次实验的JSON记录、模型权重DVC追踪文件和训练日志都会提交到代码仓库，形成完整的实验历史轨迹，便于团队协作和长期追溯。

## 工具选择

**包管理器**：项目使用[uv](https://github.com/astral-sh/uv)作为Python包管理器。uv是一个现代化的、极快的包管理工具，比传统的pip和conda快10-100倍。项目配置在`pyproject.toml`中定义，通过`uv sync`即可安装所有依赖。详细的uv使用指南请参阅[uv使用指南](docs/uv.md)。

**版本控制**：代码使用Git进行版本控制，数据和模型权重使用DVC（Data Version Control）。DVC采用与Git类似的工作流程，但专门针对大文件进行了优化。项目配置MinIO作为DVC的远程存储后端，支持团队协作和数据共享。所有数据集的版本信息（包括版本标识、创建日期、数据大小等）请参阅[DVC数据版本管理指南 - 数据集版本记录](docs/dvc_guide.md#二数据集版本记录)。

**实验追踪**：使用SwanLab进行实验可视化和管理。SwanLab提供了丰富的指标记录、可视化、对比分析功能，支持在线和离线模式。训练过程中的所有指标（损失、学习率、验证准确率等）会自动上传到SwanLab平台。

**分布式训练**：使用Accelerate和DeepSpeed进行分布式训练。Accelerate提供了统一的API，简化了分布式训练的配置。DeepSpeed提供了ZeRO优化、梯度累积、混合精度等高级功能，大幅降低了显存占用并提升了训练效率。

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd ExplicitLM

# 安装uv（如果还未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 使用uv安装项目依赖
uv sync

# 初始化DVC（首次使用）
dvc remote list  # 查看远程存储配置
```

### 运行简单测试

```bash
# 查看配置系统
python main.py --dim 1024 --n_layers 12

# 查看所有可配置参数
python 1_pretrain.py --help
```

### 运行完整训练

```bash
# 单机模式：直接运行预训练脚本
python 1_pretrain.py --epochs 3 --batch_size 48 --use_swanlab

# 或使用实验管理系统（推荐）
./experiments/scripts/exp_001.sh
```

### 集群环境训练

```bash
# 登录节点：前置准备
./experiments/scripts/exp_001_cluster.sh pre

# 同步到计算节点并训练
rsync -avz --exclude='.git' . compute-node:/path/to/ExplicitLM/
ssh compute-node "cd /path/to/ExplicitLM && ./experiments/scripts/exp_001_cluster.sh train"

# 同步回登录节点并完成后续处理
rsync -avz compute-node:/path/to/ExplicitLM/checkpoints/exp_001/ ./checkpoints/exp_001/
./experiments/scripts/exp_001_cluster.sh post
```

## 主要文档

- [DVC数据版本管理指南](docs/dvc_guide.md)：DVC常用指令、数据集版本记录、版本切换方法
- [实验运行指南](docs/experiment_workflow.md)：详细介绍DVC数据管理、实验训练流程（单机/集群）、实验记录系统
- [uv使用指南](docs/uv.md)：uv包管理器的使用方法和最佳实践
- [实验记录说明](experiments/records/README.md)：实验记录文件的结构、查询和复现方法

## 项目发展路线图

ExplicitLM的开发按照三个阶段递进式推进，每个阶段聚焦不同的技术重点和研究目标：

### 第一阶段：重构原始项目 & 构建标准运行流

本阶段的核心目标是建立规范化的项目基础设施和完善基线模型实验。主要工作包括：

- **代码重构与规范化**：整理原始代码结构，统一编码规范，建立完整的配置管理系统
- **SFT训练流程集成**：在现有预训练流程基础上，添加监督微调（Supervised Fine-Tuning）模块，支持下游任务适配
- **实验补充与验证**：
  - 补充缺失的评估指标（如困惑度、F1分数、知识检索准确率等）
  - 扩展模型规模实验（从小规模到中等规模的模型对比）
  - 实现可视化工具（记忆库使用热力图、注意力分布、知识检索路径等）
- **基准对比实验**：在通用数据集（如WikiText、LAMBADA、TriviaQA等）上与主流模型进行性能对比，验证ExplicitLM的有效性
- **CI/CD流程建设**：建立自动化测试、代码检查和实验追踪的持续集成/持续交付流程

### 第二阶段：更新模型架构 & 实现三元组提取与更新

本阶段聚焦于知识管理机制的深化，实现动态知识图谱的构建与更新：

- **知识三元组提取**：开发基于神经网络的三元组抽取模块，从文本中自动识别实体关系
- **记忆库动态更新**：将提取的三元组结构化存储到显式记忆库中，以支持知识更新

### 第三阶段：优化技术细节

本阶段针对系统的各个维度进行深度优化和应用拓展：

- **知识提取优化**：提升三元组提取的精度和效率，支持细粒度的知识颗粒度控制
- **应用场景探索**：将ExplicitLM应用于问答系统、对话生成、知识推理等实际任务
- **强化学习集成**：引入人类反馈的强化学习（RLHF），优化知识检索和生成策略
- **持续学习机制**：实现终身学习能力，支持模型在部署后持续从新数据中学习
- **效率与可扩展性**：优化计算效率和内存占用，支持更大规模的记忆库和更长的上下文

## 项目定位

这是一个研究性项目，专注于探索显式记忆增强的语言模型架构。项目遵循研究型代码开发规范：

- **MVP开发准则**：不随意添加测试代码，不使用默认值以避免掩盖潜在错误
- **类型注解规范**：所有方法签名必须包含完整类型注解，复杂类型使用`Union`/`Dict`等明确标注
- **文档规范**：所有模块、类、方法必须包含中文docstring，说明功能、参数、返回值和关键实现细节
- **代码组织模式**：使用阶段化注释组织复杂逻辑，提高可读性（如"第一阶段：配置初始化"）
- **接口设计原则**：返回值包含完整诊断信息，使用条件标志控制返回内容粒度

## 许可证

本项目采用研究性许可，仅供学术研究使用。
