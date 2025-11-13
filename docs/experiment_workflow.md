# ExplicitLM 实验运行指南

**更新**：自 commit `ffc0a80` 起，项目实验配置已全面迁移到 Hydra-Zen 系统。  
旧版基于 `TRAIN_ARGS` 和手动 DVC 版本变量（如 `DATASET_VERSION`）的流程已废弃。 

## 一、DVC 数据版本控制系统介绍以及Hydra-Zen 配置系统架构

### 1.1DVC 数据版本控制系统
Data Version Control (DVC) 是一个专为机器学习项目设计的版本控制系统，它解决了Git在管理大型数据文件和模型权重时的局限性。DVC 采用与Git类似的工作流程，但专门针对大文件进行了优化：它将实际的数据文件存储在远程存储系统（如S3、MinIO等）中，而在Git仓库中只保存轻量级的元数据文件（.dvc文件），这些元数据文件记录了数据的哈希值和存储位置。当团队成员需要获取特定版本的数据时，DVC会根据.dvc文件中的信息从远程存储拉取对应的数据，从而实现了数据的版本控制和团队协作。

在ExplicitLM项目中，DVC与Git和SwanLab共同构成了完整的实验管理体系。Git负责管理代码版本，DVC负责管理数据集和模型权重的版本，SwanLab则负责记录实验过程中的指标和可视化结果。这种三者结合的方式确保了实验的完全可复现性：通过Git commit可以恢复代码状态，通过DVC可以恢复数据和模型状态，通过SwanLab可以查看实验的训练曲线和评估指标。每一个实验都会生成一个JSON格式的记录文件，其中包含了所有必要的版本信息和复现指令，使得任何团队成员都可以精确地重现某次实验的结果。

### DVC 的核心优势

**版本控制的一致性**：DVC使用与Git相同的概念模型，包括add、commit、push、pull等命令，降低了学习成本。但不同于Git直接存储文件内容，DVC将大文件的内容哈希和位置信息存储在.dvc文件中，这些文件体积很小（通常几KB），可以轻松地纳入Git管理。这种设计使得数据版本控制变得像代码版本控制一样简单和高效。

**细粒度的版本管理**：ExplicitLM项目中包含多个独立的数据集，如训练数据集（database）、验证数据集（benchmarks）、预训练嵌入（embeddings）、知识库初始化数据等。DVC允许为每个数据集创建独立的.dvc文件，这意味着可以为不同的数据集指定不同的版本。例如，在进行对比实验时，可以保持训练数据集版本不变，而只更新验证数据集的版本；或者固定验证集版本，而使用不同版本的训练数据进行实验。这种细粒度的版本控制大大增强了实验设计的灵活性。

**高效的存储和传输**：DVC采用内容寻址存储（Content-Addressable Storage）机制，相同内容的文件只会在远程存储中保存一份。当数据集发生小幅修改时，DVC只会上传变化的部分，而不是整个数据集。在集群环境下，这种设计尤为重要：登录节点可以通过DVC的版本比较机制判断本地数据是否需要更新，如果数据版本未变化，就完全跳过数据同步步骤，避免在网络拥堵的集群环境中浪费大量时间进行不必要的数据传输。

### 项目中的 DVC 配置

本项目使用MinIO作为DVC的远程存储后端。MinIO是一个高性能的对象存储系统，兼容Amazon S3 API，在项目配置中，远程存储地址为`s3://192.168.31.231:11900/dvc-storage`。DVC的配置信息存储在`.dvc/config`文件中，包含了远程存储的连接参数和认证信息。团队成员在首次使用DVC时，需要确保自己的环境可以访问这个MinIO服务器，并且已经正确配置了访问凭证。

项目的`.gitignore`文件已经配置了完善的DVC规则：实际的数据目录（如`data/raw/*`、`data/database/*`）会被Git忽略，但对应的.dvc元数据文件（如`data/database.dvc`）会被Git追踪。模型权重目录`checkpoints/`也遵循同样的规则，只有.dvc文件会被提交到Git仓库。DVC的临时文件和缓存目录（如`.dvc/tmp`、`.dvc/cache`）也被正确地排除在Git追踪之外，避免了不必要的文件进入版本控制系统。

### 注意：自 commit `ffc0a80` 起，数据集版本不再通过环境变量指定，而是由 Hydra-Zen 配置系统统一管理（见下文）。

## 1.2Hydra-Zen 配置系统架构

自 commit `ffc0a80` 起，ExplicitLM 项目全面采用 **Hydra-Zen** 作为统一的实验配置管理系统。该系统基于 [Hydra](https://hydra.cc/) 和 [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) 构建，旨在实现：

- **结构化配置**：将数据集、模型、训练、日志等参数模块化组织  
- **可组合性**：通过配置组合（composition）灵活复用和覆盖设置  
- **命令行友好**：支持运行时动态覆盖任意配置字段  
- **版本一致性**：配置与代码、数据一同纳入 Git/DVC 管理，确保实验可复现  

### 配置模块结构

所有配置定义位于项目根目录的 `config/` Python 包中，结构如下：
config/
├── init.py
├── dataset.py      # 数据集路径与 DVC 版本控制
├── model.py        # 模型架构参数（如 dim, n_layers, knowledge_num）
├── training.py     # 训练超参（epochs, batch_size, learning_rate 等）
├── logging.py      # 日志与 SwanLab 集成开关
└── exp_*.yaml      # （可选）实验专属配置组合文件


每个 `.py` 文件通过 `hydra_zen.store` 注册一组命名配置（例如 `"default"`、`"base"`），主训练脚本 `1_pretrain.py` 在启动时自动加载这些配置。

### 核心优势

- **无需修改代码即可切换实验设置**：只需更改配置名称或覆盖参数  
- **避免 shell 脚本参数爆炸**：不再依赖 `TRAIN_ARGS` 或环境变量传递复杂参数  
- **配置即元数据**：实验记录中的 `hyperparameters` 字段直接来自配置树，结构清晰、无歧义  

> **重要**：所有新实验必须基于此配置系统开发。旧版通过命令行或 shell 变量传参的方式已废弃，继续使用将导致配置缺失或训练异常。


## 二、使用 DVC 管理项目数据集

### 2.1 数据集版本控制的基本工作流

在ExplicitLM项目中，数据集的版本控制采用与代码版本控制并行但独立的方式。当需要添加新的数据集或更新已有数据集时，首先将数据文件放置到相应的目录（如`data/database/`），然后使用`dvc add data/database.dvc`命令让DVC追踪这个目录。DVC会计算目录内所有文件的哈希值，并将这些信息保存到`data/database.dvc`文件中。接下来使用`dvc push`将实际的数据文件上传到MinIO远程存储，最后将.dvc文件通过Git提交到代码仓库。这样，数据的实际内容存储在MinIO中，而数据的版本信息和元数据则存储在Git中。

当团队其他成员需要获取数据时，他们首先通过`git pull`获取最新的.dvc文件，然后运行`dvc pull`从MinIO下载对应的数据。DVC会根据.dvc文件中的哈希值检查本地是否已经有相应版本的数据，如果已经存在且哈希值匹配，则跳过下载。如果需要切换到某个历史版本的数据，首先使用Git切换到对应的commit（该commit包含了目标版本的.dvc文件），然后运行`dvc checkout`让DVC根据.dvc文件的内容恢复对应版本的数据文件。

### 2.2 项目中的数据集组织结构

ExplicitLM项目采用细粒度的数据集组织方式，每个独立的数据集都有自己的.dvc追踪文件：

- **训练数据集** (`data/database/`)：包含预训练使用的主要文本数据，通过data/database.dvc文件追踪。这是最大也是最核心的数据集，更新频率相对较低。在实验配置中通过 `config.dataset.train.version` 字段指定其版本。

- **验证数据集** (`data/benchmarks/`)：包含用于模型评估的基准测试数据，通过`data/benchmarks.dvc`文件追踪。验证集的版本控制独立于训练集，这使得可以在保持训练数据不变的情况下更新评估标准，或者在不同的评估集上测试同一个训练配置。实验脚本中通过`config.dataset.train.version`变量指定版本。

- **预训练嵌入** (`data/embeddings/`，可选)：如果使用预训练的词向量或其他嵌入表示，可以单独追踪其版本。通过`config.dataset.embedding.version`变量控制。

- **知识库初始化数据** (`data/knowledge_base/`，可选)：用于初始化知识库的结构化数据，独立版本控制通过`config.dataset.knowledge_base.version`变量。

- **缓存数据** (`data/cache/`，可选)：预处理后的缓存文件，可以通过`config.dataset.cache.version`变量单独指定版本。


这种细粒度的组织方式带来的最大好处是实验设计的灵活性。例如，在进行消融实验时，可以固定所有其他数据集的版本，只改变训练数据集的版本，从而精确地隔离变量影响。或者在评估模型泛化能力时，可以固定训练数据和模型配置，只更换不同的验证数据集版本。

### 2.3 实验中的数据版本指定

在 Hydra-Zen 配置系统中，每个数据集的版本通过 config/dataset.py 中的配置字段声明。
在实验脚本的开头，需要声明每个数据集使用的版本。版本指定有三种方式：

**使用当前版本**：将版本变量设置为空字符串（如`DATASET_VERSION=""`），表示使用当前Git HEAD对应的数据版本。这是最常见的情况，适用于使用最新数据进行新实验的场景。

**指定历史版本**：将版本变量设置为特定的Git commit哈希（如`DATASET_VERSION="abc1234"`），表示使用该commit对应时刻的数据版本。这种方式常用于复现历史实验或进行严格的对比实验。可以从之前的实验记录JSON文件中获取该commit哈希值，确保使用完全相同的数据。

**混合版本策略**：不同数据集可以指定不同的版本。例如，`train.version="abc1234"`指定使用历史版本的训练数据，而`val.version=""`使用当前最新的验证数据。这种策略在逐步更新数据集时非常有用，可以保持部分数据的稳定性，同时测试新数据的效果。

实验脚本会根据这些版本变量自动执行数据同步。如果指定了历史版本，脚本会先使用`git checkout`切换到对应的commit获取该版本的.dvc文件，然后执行`dvc checkout`恢复对应版本的数据，最后再切换回原来的代码分支。如果版本变量为空，则直接使用当前的数据，不进行任何切换操作。

### 2.4 数据版本的查询和管理

查看某个数据集当前的版本信息，可以使用`git log data/database.dvc`命令，这会显示该.dvc文件的提交历史，每次提交都对应一次数据更新。通过`dvc diff`命令可以比较不同版本之间的数据差异，虽然这个命令对于大规模数据集来说可能比较耗时。

在实际工作中，更常用的方式是查看实验记录文件。每个实验的JSON记录中都包含了`versions.data`字段，其中详细记录了该实验使用的所有数据集版本。例如，要复现实验exp_001，可以从`experiments/records/exp_001.json`中读取`versions.data.train`字段获取训练数据的版本哈希，然后在新实验中使用相同的版本。

如果需要批量查询多个实验使用的数据版本，可以使用jq工具：`jq '.versions.data' experiments/records/*.json`会列出所有实验的数据版本信息。这种方式可以快速找到使用特定数据版本的所有实验，或者分析数据更新对实验结果的影响。

## 三、实验训练流程

### 3.1 非集群环境训练（单机模式）

单机模式适用于在个人工作站或具有完整网络和GPU资源的单个服务器上进行实验。在这种模式下，整个实验流程由一个shell脚本自动化完成，无需手动干预多个步骤。

#### 3.1.1 创建实验脚本

在`experiments/scripts/`目录下创建一个新的实验脚本文件，文件名应当与实验ID对应（如`exp_003.sh`）。脚本的核心内容包括以下几个部分：

首先是实验的元数据定义。`EXP_ID`是实验的唯一标识符，应当在整个项目中保持唯一，建议使用有意义的命名或递增的编号。`EXP_DESC`是实验的中文描述，应当简要说明本次实验的主要目的和配置特点，这个描述会被记录到实验元数据文件中，便于日后查询和理解。

接下来是数据版本的声明，数据版本现在由 Hydra-Zen 配置文件管理（见 config/dataset.py），无需在此处手动指定。

最后，训练超参数（如 --epochs、--dim 等）现在通过 Hydra-Zen 配置组合指定，不再通过命令行参数传递。

脚本的最后两行是固定的模板代码：首先获取脚本所在目录的绝对路径，然后source核心执行脚本`_run_experiment_core.sh`。核心脚本包含了实验执行的完整逻辑，包括环境检查、数据同步、训练执行、结果记录等步骤。

一个完整的实验脚本示例如下：

```bash
#!/bin/bash
################################################################################
# 实验003 - 调整学习率和模型深度
################################################################################

# 执行实验（固定模板）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_experiment_core_hydra_zen.sh"
```

#### 3.1.2 执行实验

创建脚本后，首先为其添加可执行权限：`chmod +x experiments/scripts/exp_003.sh`。然后在项目根目录执行该脚本：`./experiments/scripts/exp_003.sh`。

脚本执行后会依次完成以下步骤：

**环境检查阶段**：脚本会首先验证运行环境是否满足要求。检查项包括确认当前是否在Git仓库中、DVC是否已经初始化、必要的环境变量是否已设置等。如果检查发现任何问题，脚本会输出错误信息并终止执行，避免在不完整的环境中运行实验导致结果不可靠。

**代码版本记录**：脚本会获取当前代码的Git commit哈希值，这个哈希值唯一标识了实验使用的代码版本。即使在未提交的代码上运行实验，脚本也会记录当前的commit，并在实验记录中标注代码可能包含未提交的修改，提醒后续复现时需要注意。

**数据同步阶段**：根据 config/dataset.py 中声明的数据版本自动同步，脚本会逐个数据集进行版本切换和同步。对于每个非空的版本变量，脚本会先用`git checkout <version> -- <dvc_file>`切换到指定版本的.dvc元数据文件，然后执行`dvc checkout <dvc_file>`恢复对应版本的数据文件，最后切换回当前的Git分支。如果某个数据集的版本变量为空，则跳过该数据集的同步，直接使用当前已有的数据。整个过程中会输出详细的日志，显示每个数据集的版本切换情况。

**训练执行阶段**：数据准备就绪后，脚本会执行训练命令。执行 python 1_pretrain.py --config-name=<exp_config>，所有超参由 Hydra 自动注入。训练过程的标准输出和标准错误会被同时显示在终端并保存到日志文件`logs/${EXP_ID}.log`中，便于后续查看和调试。如果训练过程中发生错误，脚本会捕获退出码并终止后续步骤。

**结果收集阶段**：训练完成后，脚本会从`.swanlab_url`文件中读取SwanLab实验页面的URL（该文件由修改后的`1_pretrain.py`在训练结束时生成）。然后使用DVC追踪生成的模型权重目录`checkpoints/${EXP_ID}/`，执行`dvc add checkpoints/${EXP_ID}.dvc`创建权重的版本追踪文件，并执行`dvc push`将权重上传到MinIO远程存储。

**元数据生成阶段**：脚本会自动生成一个JSON格式的实验记录文件`experiments/records/${EXP_ID}.json`。这个文件包含了实验的所有关键信息：实验ID和描述、时间戳、代码版本、所有数据集的版本信息、训练参数、SwanLab URL、模型权重路径和版本哈希、运行环境信息（Python版本、CUDA版本、GPU数量）、以及完整的复现指令。记录完整的 Hydra 配置树作为 hyperparameters 字段。

**版本提交阶段**：最后，脚本会将所有变更一次性提交到Git仓库。提交的内容包括新生成的实验记录文件、模型权重的.dvc文件、以及任何在实验过程中修改的代码（如果有未提交的代码修改）。提交信息会清晰地标注实验ID和描述，格式为`chore: 添加实验记录 exp_003 - 学习率降低至1e-4，增加模型层数至12层`。

整个过程是完全自动化的，唯一需要人工介入的是在开始前创建实验脚本，以及在训练过程中可能需要监控训练状态（通过日志文件或SwanLab页面）。实验完成后，所有必要的信息都已经被记录和版本化，可以随时通过实验记录文件复现或对比分析。

### 3.2 集群环境训练（三阶段模式）

集群环境的复杂性在于计算资源和网络资源的分离。通常，登录节点可以访问外部网络和共享存储，但没有GPU资源；而计算节点拥有强大的GPU算力，但出于安全考虑被隔离在内网中，无法访问外部网络。这种架构下，DVC的数据同步、Git的版本管理、SwanLab的结果上传等需要网络的操作都必须在登录节点完成，而实际的模型训练必须在计算节点进行。因此，ExplicitLM的集群训练流程被设计为三个独立的阶段，每个阶段在相应的节点上手动执行。

#### 3.2.1 集群实验脚本的结构

集群模式的实验脚本采用统一脚本的设计，通过命令行参数控制执行哪个阶段。脚本的参数定义部分与单机模式完全相同，包括实验ID、描述、数据版本和训练参数。不同之处在于脚本的执行部分：脚本接受一个位置参数来指定运行阶段，可选值为`pre`（前置阶段）、`train`（训练阶段）、`post`（后续阶段）。根据参数的值，脚本会 source 对应的 _hydra_zen 核心脚本文件。

一个集群实验脚本的典型结构如下：

```bash
#!/bin/bash
################################################################################
# 实验001 - 集群模式统一脚本
#
# 用法：
#   ./exp_001_cluster.sh pre    # 登陆节点：前置工作（数据同步）
#   ./exp_001_cluster.sh train  # 计算节点：训练
#   ./exp_001_cluster.sh post   # 登陆节点：后续工作（Git提交）
################################################################################

# ============================================================================
# 实验配置（只需定义 ID 和描述）
# 数据版本和训练参数请在 config/ 目录下通过 Hydra-Zen 配置管理
# ============================================================================
EXP_ID="exp_001"
EXP_DESC="基线实验 knowledge_num=1M epochs=10"

# 数据版本
DATASET_VERSION=""
VAL_DATASET_VERSION=""
EMBEDDING_VERSION=""
DATABASE_VERSION=""
CACHE_VERSION=""

# 训练参数
TRAIN_ARGS="--epochs 10 --knowledge_num 1048576 --dim 512 --n_layers 8 --batch_size 48 --learning_rate 2e-4 --use_swanlab"

# ============================================================================
# 执行阶段选择
# ============================================================================
STAGE="${1:-pre}"  # 默认为pre阶段

case "$STAGE" in
    pre)
        echo "执行前置阶段（登陆节点）..."
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        source "${SCRIPT_DIR}/_run_experiment_cluster_pre_hydra_zen.sh""
        ;;
    train)
        echo "执行训练阶段（计算节点）..."
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        source "${SCRIPT_DIR}/_run_experiment_cluster_train_hydra_zen.sh"
        ;;
    post)
        echo "执行后续阶段（登陆节点）..."
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        source "${SCRIPT_DIR}/_run_experiment_cluster_post_hydra_zen.sh"
    *)
        echo "错误：未知阶段 '$STAGE'"
        echo ""
        echo "用法："
        echo "  $0 pre    - 登陆节点：数据同步和准备"
        echo "  $0 train  - 计算节点：执行训练"
        echo "  $0 post   - 登陆节点：DVC追踪和Git提交"
        exit 1
        ;;
esac
```

这种设计的优势在于所有实验参数只需要在脚本顶部定义一次，三个阶段共享相同的配置，避免了在多个文件中重复修改参数可能导致的不一致性。

#### 3.2.2 第一阶段：前置准备（登录节点）

在登录节点上执行`./exp_001_cluster.sh pre`启动前置准备阶段。这个阶段的主要任务是准备好训练所需的所有数据，并记录实验的版本信息。

脚本首先会记录当前代码的Git commit哈希值，然后逐个处理实验脚本中声明的数据集版本。这里的关键优化是**智能数据同步机制**：对于每个数据集，脚本会先获取本地当前的数据版本（通过`git log -1 --format="%H" -- <dvc_file>`命令），然后与目标版本进行比较。只有当两者不一致时，才会执行实际的数据同步操作（`git checkout`切换.dvc文件版本，然后`dvc checkout`恢复数据文件）。如果版本一致，脚本会输出提示信息并跳过同步，直接使用已有的数据。

这个机制在集群环境中极为重要，因为登录节点的网络带宽是共享资源，在高峰时段可能非常拥堵。对于大型数据集（如数GB的训练语料），即使网络畅通，完整同步也可能需要数十分钟甚至更长时间。通过版本比较跳过不必要的同步，可以大幅缩短实验准备时间。特别是在进行超参数调优等需要频繁启动实验的场景下，数据版本往往保持不变，智能同步机制可以节省大量等待时间。

数据准备完成后，脚本会将所有关键变量（实验ID、训练参数、各数据集版本哈希等）导出到一个状态文件`.cluster_state_${EXP_ID}`中。这个文件的作用是在阶段之间传递信息：由于三个阶段是独立的shell进程，无法直接共享变量，因此需要通过文件来持久化状态。状态文件本质上是一系列的`export`语句，可以在后续阶段中通过`source`命令加载。

前置阶段完成后，登录节点的工作目录中已经准备好了正确版本的所有数据文件，并且状态文件记录了本次实验的完整配置。下一步需要将整个工作目录同步到计算节点。

#### 3.2.3 第二阶段：训练执行（计算节点）

将工作目录从登录节点同步到计算节点后（通常使用rsync、scp或共享文件系统），在计算节点上执行`./exp_001_cluster.sh train`开始训练阶段。

脚本首先会加载状态文件`.cluster_state_${EXP_ID}`，恢复实验的所有配置变量。然后构建完整的训练命令：`accelerate launch 1_pretrain.py <TRAIN_ARGS>`，并执行训练。训练过程的输出会同时显示在终端和保存到日志文件中。

由于计算节点无法访问外部网络，SwanLab等在线服务虽然无法实时同步数据，但SwanLab客户端会在本地保存实验记录，并在网络恢复后自动上传。训练脚本`1_pretrain.py`在训练结束时会将SwanLab实验URL写入到`.swanlab_url_${EXP_ID}`文件中（即使此时URL可能尚未生效）。

训练完成后，脚本会验证模型权重文件是否已经正确保存到`checkpoints/${EXP_ID}/`目录中。如果验证失败，脚本会输出错误信息，提醒用户检查训练过程是否正常完成。

训练阶段结束后，需要将工作目录（特别是生成的模型权重和SwanLab URL文件）同步回登录节点，为最后的后续处理阶段做准备。

#### 3.2.4 第三阶段：后续处理（登录节点）

在登录节点上执行`./exp_001_cluster.sh post`完成最后的后续处理阶段。

脚本会重新加载状态文件，然后执行一系列需要网络的操作。首先，使用DVC追踪生成的模型权重：`dvc add checkpoints/${EXP_ID}.dvc`会创建权重的版本追踪文件，`dvc push checkpoints/${EXP_ID}.dvc`会将实际的权重文件上传到MinIO远程存储。这一步确保了模型权重像数据集一样被版本化管理，团队成员可以通过DVC拉取任意版本的模型权重。

接下来，脚本会读取`.swanlab_url_${EXP_ID}`文件获取实验URL，并生成完整的实验记录JSON文件。记录文件的生成逻辑与单机模式完全相同，包含了实验的所有元数据和复现指令。

最后，脚本会将所有变更提交到Git仓库：实验记录文件、模型权重的.dvc文件，以及可能在实验过程中修改的代码文件。提交信息会标注实验ID和描述。

至此，一次集群实验的完整流程结束。所有的代码、数据、模型权重都已经被版本化，实验结果已经记录并可复现。

#### 3.2.5 集群训练的完整流程示例

假设要在集群上运行实验exp_002，完整的操作流程如下：

**步骤1：在登录节点准备数据**

```bash
# 登录到集群的登录节点
ssh user@login-node

# 进入项目目录
cd /path/to/ExplicitLM

# 执行前置阶段
./experiments/scripts/exp_002_cluster_hydra_zen.sh pre
```

这一步会输出类似如下的日志：

```
[INFO] 开始前置阶段...
[INFO] 记录代码版本: d2de793a
[INFO] 同步数据集版本:
  - database: 版本未变更 (d2de793a)，跳过同步
  - benchmarks: 版本未变更 (d2de793a)，跳过同步
[INFO] 状态文件已保存: .cluster_state_exp_002
[INFO] 前置阶段完成
```

**步骤2：同步到计算节点**

```bash
# 将整个项目目录同步到计算节点
rsync -avz --exclude='.git' --exclude='.dvc/cache' \
    /path/to/ExplicitLM/ compute-node:/path/to/ExplicitLM/
```

注意排除`.git`和`.dvc/cache`目录以减少传输量，因为这些目录体积很大且在计算节点上不需要。

**步骤3：在计算节点执行训练**

```bash
# 登录到计算节点
ssh user@compute-node

# 进入项目目录
cd /path/to/ExplicitLM

# 执行训练阶段
./experiments/scripts/exp_002_cluster_hydra_zen.sh train
```

训练过程可能持续数小时甚至数天，可以通过日志文件`logs/exp_002.log`监控进度。

**步骤4：同步回登录节点**

训练完成后，将生成的文件同步回登录节点：

```bash
# 在计算节点上执行
rsync -avz /path/to/ExplicitLM/checkpoints/exp_002/ \
    login-node:/path/to/ExplicitLM/checkpoints/exp_002/

rsync -avz /path/to/ExplicitLM/.swanlab_url_exp_002 \
    login-node:/path/to/ExplicitLM/
```

**步骤5：在登录节点完成后续处理**

```bash
# 回到登录节点
ssh user@login-node
cd /path/to/ExplicitLM

# 执行后续阶段
./experiments/scripts/exp_002_cluster_hydra_zen.sh post
```

这一步会输出类似如下的日志：

```
[INFO] 开始后续阶段...
[INFO] 追踪模型权重...
[INFO] 上传权重到DVC远程存储...
[INFO] 生成实验记录文件: experiments/records/exp_002.json
[INFO] 提交变更到Git...
[INFO] 后续阶段完成
[INFO] 实验exp_002已完成！
```

**步骤6：推送到远程仓库（可选）**

```bash
# 将提交推送到远程Git仓库
git push origin main
```

至此，整个集群实验流程完成。团队其他成员可以通过`git pull`获取实验记录，通过`dvc pull`获取模型权重。

### 3.3 实验结果的查询和复现

#### 3.3.1 查看实验记录

每个实验完成后都会在`experiments/records/`目录下生成一个JSON格式的元数据文件。可以使用jq工具或直接查看JSON文件来获取实验信息。

查看某个实验的完整信息：

```bash
cat experiments/records/exp_001.json | jq '.'
```

只查看超参数配置：

```bash
cat experiments/records/exp_001.json | jq '.hyperparameters'
```

对比多个实验的某个参数：

```bash
for f in experiments/records/*.json; do
    exp_id=$(basename $f .json)
    knowledge_num=$(jq -r '.hyperparameters.knowledge_num' $f)
    echo "$exp_id: knowledge_num=$knowledge_num"
done
```

查找使用特定数据版本的所有实验：

```bash
grep -l '"dataset_commit": "d2de793a"' experiments/records/*.json
```

#### 3.3.2 复现历史实验

要完整复现一个历史实验，需要恢复三个方面的状态：代码版本、数据版本、训练配置。实验记录文件的`reproduction`字段提供了所有必要的指令。

**只恢复代码和数据，不重新训练**：

```bash
# 假设要复现exp_001
cd /path/to/ExplicitLM

# 恢复代码版本
git checkout d2de793a

# 恢复训练数据集版本
git checkout d2de793a -- data/database.dvc
dvc checkout data/database.dvc
git checkout -  # 返回当前分支，但保持数据文件不变

# 恢复验证数据集版本
git checkout d2de793a -- data/benchmarks.dvc
dvc checkout data/benchmarks.dvc
git checkout -

# 拉取模型权重
dvc pull checkpoints/exp_001.dvc
```

**完全重新训练**：

除了恢复代码和数据，还需要执行训练命令。可以从实验记录文件的`reproduction.full_command`字段复制完整命令：

```bash

# 恢复代码和数据（同上）
git checkout d2de793a
git checkout d2de793a -- data/database.dvc
dvc checkout data/database.dvc
git checkout -

# 重新训练（使用 Hydra）
python 1_pretrain.py --config-name=exp_001 ++experiment.id=exp_001_repro
```

重新训练得到的模型权重哈希值应当与原实验记录中的`checkpoint_hash`一致（在相同的随机种子和硬件条件下）。如果哈希值不同，可能是由于浮点运算的不确定性或硬件差异导致的轻微数值差异，这在深度学习中是正常现象。

#### 3.3.3 实验结果的可视化分析

实验记录文件中包含了SwanLab实验页面的URL（`results.swanlab_url`字段）。访问这个URL可以查看训练过程中记录的所有指标曲线、日志、系统资源使用情况等。SwanLab提供了丰富的可视化功能，包括对比多个实验的指标曲线、分析超参数对结果的影响等。

对于批量实验分析，可以编写脚本从JSON文件中提取关键指标，然后使用matplotlib等工具绘制自定义的对比图表。例如，提取所有实验的最终验证准确率并绘制条形图：

```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

records_dir = Path("experiments/records")
experiments = []
accuracies = []

for json_file in records_dir.glob("*.json"):
    with open(json_file) as f:
        record = json.load(f)
        experiments.append(record['experiment']['id'])
        # 假设SwanLab中记录了final_accuracy指标
        # 实际使用中需要从SwanLab API或导出的CSV中获取
        accuracies.append(record.get('final_accuracy', 0))

plt.bar(experiments, accuracies)
plt.xlabel('Experiment ID')
plt.ylabel('Validation Accuracy')
plt.title('Experiment Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('experiment_comparison.png')
```

## 四、常见问题和最佳实践

### 4.1 常见问题排查

**问题：DVC提示"cache not found"或"file not in cache"**

这通常意味着本地DVC缓存中没有对应版本的数据文件，而远程存储中也无法找到。可能的原因包括：该版本的数据从未被`dvc push`上传过；远程存储连接失败；.dvc文件的哈希值与实际数据不匹配。解决方法是检查网络连接和DVC远程配置（`dvc remote list -v`），确认远程存储中是否存在对应的数据文件，必要时重新生成数据并`dvc add`和`dvc push`。

**问题：Git提示"large file detected"或仓库体积异常增大**

这说明有大文件被错误地添加到了Git仓库中，而不是通过DVC管理。检查`.gitignore`文件是否正确配置，确保数据目录和权重目录被排除。如果已经误提交，需要使用`git filter-branch`或BFG Repo-Cleaner工具从Git历史中移除大文件，然后重新用DVC追踪这些文件。

**问题：集群训练时SwanLab URL显示为"N/A"**

这可能是因为训练脚本没有在结束时成功写入`.swanlab_url_${EXP_ID}`文件，或者SwanLab在离线环境下无法生成有效的URL。检查训练日志中是否有SwanLab相关的错误信息。如果是离线环境导致的，可以在训练完成后手动从SwanLab本地数据中提取URL，或者在后续阶段手动更新实验记录文件。

**问题：数据版本切换后训练仍然使用旧数据**

这可能是因为数据被缓存在其他位置，或者数据加载代码中使用了绝对路径。确保`dvc checkout`确实执行成功（检查命令输出和数据文件的修改时间）。清除可能存在的缓存目录，如`data/cache/`。检查训练代码中的数据路径是否正确。

### 4.2 最佳实践建议

**实验设计的系统性**：在开始实验之前，应当明确实验的研究问题和假设。遵循单变量原则，每次实验只改变一个关键参数或配置，这样才能清晰地归因结果差异。建立基线实验作为对比的锚点，所有后续实验都应当与基线进行对比分析。对于对比实验，尽量使用相同的数据版本，避免数据差异干扰结论。

**数据版本的管理策略**：对于稳定的数据集（如标准验证集），应当明确标记一个"官方"版本，所有实验都使用这个版本以保证可比性。对于频繁更新的数据集（如训练语料），应当定期打"快照"版本（通过Git tag标记DVC文件的特定commit），便于日后引用。在实验记录中详细注释数据版本的选择原因，特别是使用历史版本时，说明是为了复现、对比还是其他目的。

**资源和时间的优化**：在集群环境中，合理利用智能数据同步机制，避免不必要的网络传输。对于大型实验，可以先在小规模数据上进行pilot实验，验证配置正确后再使用完整数据集。定期清理DVC缓存目录和不再需要的模型权重，释放存储空间。使用GPU资源时，注意batch size和模型大小的配置，避免内存溢出导致的训练中断。

**团队协作的规范性**：统一实验命名规范，建议使用语义化的命名或严格递增的编号。实验完成后及时推送到远程仓库（Git和DVC），让团队成员能够及时获取最新成果。定期组织实验结果的review会议，讨论实验发现和后续方向。维护一个实验索引文档，总结关键实验的结论和最佳配置。

**实验记录的质量保证**：确保实验记录文件的完整性和准确性，避免手动修改JSON文件。在实验描述中提供足够的上下文信息，日后查看时能够快速理解实验动机。对于失败或中断的实验，也应当保留记录并注释原因，避免重复犯错。定期备份实验记录目录，防止意外丢失宝贵的实验数据。

**代码和环境的一致性**：尽量避免在未提交的代码上运行正式实验，如果必须这样做，应当在实验记录中明确标注并尽快提交代码。使用requirements.txt或conda environment文件锁定依赖版本，确保环境的可复现性。在集群环境中，保持登录节点和计算节点的代码同步，避免因版本不一致导致的行为差异。

## 五、附录

### 5.1 相关文件和目录结构

```
ExplicitLM/
├── .dvc/                           # DVC配置和缓存
│   ├── config                      # DVC远程存储配置
│   └── .gitignore                  # DVC内部文件的Git忽略规则
├── .gitignore                      # Git忽略规则（包含DVC相关配置）
├── data/                           # 数据目录
│   ├── database/                   # 训练数据（被Git忽略）
│   ├── database.dvc                # 训练数据的DVC元数据（被Git追踪）
│   ├── benchmarks/                 # 验证数据（被Git忽略）
│   ├── benchmarks.dvc              # 验证数据的DVC元数据（被Git追踪）
│   └── ...                         # 其他数据集及其.dvc文件
├── checkpoints/                    # 模型权重目录
│   ├── exp_001/                    # 实验001的权重（被Git忽略）
│   ├── exp_001.dvc                 # 实验001权重的DVC元数据（被Git追踪）
│   └── ...                         # 其他实验的权重和.dvc文件
├── experiments/                    # 实验管理目录
│   ├── scripts/                    # 实验脚本目录
│   │   ├── _run_experiment_core_hydra_zen.sh           # 单机模式核心脚本
│   │   ├── _run_experiment_cluster_pre_hydra_zen.sh    # 集群模式前置阶段脚本
│   │   ├── _run_experiment_cluster_train_hydra_zen.sh  # 集群模式训练阶段脚本
│   │   ├── _run_experiment_cluster_post_hydra_zen.sh   # 集群模式后续阶段脚本
│   │   ├── exp_001.sh              # 实验001单机脚本
│   │   ├── exp_001_cluster_hydra_zen.sh      # 实验001集群脚本
│   │   └── ...                     # 其他实验脚本
│   └── records/                    # 实验记录目录
│       ├── README.md               # 实验记录文件说明文档
│       ├── exp_001.json            # 实验001的元数据记录
│       └── ...                     # 其他实验的记录文件
├── logs/                           # 训练日志目录
│   ├── exp_001.log                 # 实验001的训练日志
│   └── ...                         # 其他实验的日志
├── 1_pretrain.py                   # 训练脚本（已添加SwanLab URL导出功能）
└── README.md                       # 项目主文档
```

### 5.2 实验记录JSON文件的完整示例

```json
{
  "experiment": {
    "id": "exp_001",
    "description": "基线实验 knowledge_num=1M epochs=10",
    "timestamp": "2024-01-15T08:30:00Z",
    "script": "exp_001.sh",
    "command": "python 1_pretrain.py --config-name=exp_001"
  },
  "versions": {
    "code_commit": "d2de793a1234567890abcdef1234567890abcdef",
    "code_commit_short": "d2de793a",
    "data": {
      "dataset_commit": "d2de793a1234567890abcdef1234567890abcdef",
      "val_dataset_commit": "d2de793a1234567890abcdef1234567890abcdef",
      "embedding_commit": null,
      "database_init_commit": null,
      "cache_commit": null
    },
    "checkpoint_dvc": "checkpoints/exp_001.dvc",
    "checkpoint_hash": "a1b2c3d4e5f67890a1b2c3d4e5f67890",
    "checkpoint_hash_short": "a1b2c3d4"
  },
  "hyperparameters": {
    "epochs": 10,
    "knowledge_num": 1048576,
    "dim": 512,
    "n_layers": 8,
    "batch_size": 48,
    "learning_rate": 0.0002,
    "use_swanlab": true
  },
  "results": {
    "swanlab_url": "https://swanlab.cn/@username/project/runs/exp_001",
    "checkpoint_dir": "checkpoints/exp_001/"
  },
  "environment": {
    "python_version": "3.10.12",
    "cuda_version": "12.1",
    "num_gpus": 8
  },
  "reproduction": {
    "code_checkout": "git checkout d2de793a",
    "data_checkout_steps": [
      "git checkout d2de793a -- data/database.dvc && dvc checkout data/database.dvc && git checkout -",
      "git checkout d2de793a -- data/benchmarks.dvc && dvc checkout data/benchmarks.dvc && git checkout -"
    ],
    "checkpoint_pull": "dvc pull checkpoints/exp_001.dvc",
    "full_command": "git checkout d2de793a && git checkout d2de793a -- data/database.dvc && dvc checkout data/database.dvc && git checkout - && git checkout d2de793a -- data/benchmarks.dvc && dvc checkout data/benchmarks.dvc && git checkout - && python 1_pretrain.py --config-name=exp_001"
  }
}

### 5.3 常用命令速查

**DVC基本操作**：

```bash
# 追踪新的数据文件或目录
dvc add data/new_dataset/

# 上传数据到远程存储
dvc push data/new_dataset.dvc

# 从远程存储下载数据
dvc pull data/new_dataset.dvc

# 切换到历史版本的数据
git checkout <commit> -- data/database.dvc
dvc checkout data/database.dvc

# 查看DVC远程存储配置
dvc remote list -v

# 检查DVC状态
dvc status
```

**实验管理操作**：

```bash
# 单机模式运行实验
./experiments/scripts/exp_001.sh

# 集群模式三阶段执行
./experiments/scripts/exp_001_cluster_hydra_zen.sh pre    # 登录节点
./experiments/scripts/exp_001_cluster_hydra_zen.sh train  # 计算节点
./experiments/scripts/exp_001_cluster_hydra_zen.sh post   # 登录节点

# 查看实验记录
cat experiments/records/exp_001.json | jq '.'

# 对比多个实验的参数
jq '.hyperparameters' experiments/records/*.json
```

**数据版本查询**：

```bash
# 查看数据集的版本历史
git log --oneline data/database.dvc

# 查看某次实验使用的数据版本
jq '.versions.data' experiments/records/exp_001.json

# 查找使用特定数据版本的所有实验
grep -l '"dataset_commit": "d2de793a"' experiments/records/*.json
```

### 5.4 故障排查清单

遇到问题时，可以按照以下清单逐项检查：

- [ ] 确认当前在Git仓库中：`git status`
- [ ] 确认DVC已初始化：`ls -la .dvc/config`
- [ ] 确认DVC远程存储可访问：`dvc remote list -v` 和网络连通性测试
- [ ] 确认必需的环境变量已设置（如果使用MinIO认证）
- [ ] 确认.dvc文件存在且未损坏：`cat data/database.dvc`
- [ ] 确认数据目录权限正确：`ls -ld data/database/`
- [ ] 确认磁盘空间充足：`df -h`
- [ ] 查看详细的错误日志：检查`logs/`目录和终端输出
- [ ] 确认实验脚本中的变量定义正确：`grep -E "EXP_ID" <script>`   (TRAIN_ARGS 已由 Hydra-Zen 配置管理，无需在脚本中定义)
- [ ] 确认Git工作目录干净或已知未提交的修改：`git status`
